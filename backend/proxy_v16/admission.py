"""Shared optional-maintenance admission policy, independent of memory contents.

Call on the serving event loop. There are no await points inside mutations.
This module never cancels native GPU work or releases a running reservation
without a matching terminal acknowledgement from the request owner.
"""
from __future__ import annotations
from dataclasses import dataclass
import heapq
import math
import time


@dataclass
class Ticket:
    request_id: str
    session_id: str
    snapshot_id: str
    deadline: float
    remaining_cost_s: float
    expected_relative_ttft_gain: float
    estimate_source: str
    submitted_at: float
    sequence: int
    initial_wait_budget_s: float
    state: str = 'pending'
    cancellation_requested: bool = False
    dispatched_at: float | None = None
    terminal_reason: str | None = None


class AdmissionQueue:
    """Two indices over the same tickets: expected benefit and waiting slack.

    Slack ratio = (deadline - now - remaining pipeline cost) /
                  (deadline - enqueue time - initial pipeline cost).
    Urgent tickets use the waiting queue; otherwise the benefit queue is used.
    Requests that can no longer fit expire rather than delaying foreground work.
    Optional deadlines are predictions, never extensions of actual tool windows.
    """

    def __init__(self, *, max_running=2, max_pending=64, urgent_ratio=.25,
                 clock=time.monotonic, emit=lambda event, **fields: None):
        if type(max_running) is not int or max_running < 1:
            raise ValueError('positive max_running required')
        if type(max_pending) is not int or max_pending < 1:
            raise ValueError('positive max_pending required')
        if not math.isfinite(urgent_ratio) or not 0 <= urgent_ratio <= 1:
            raise ValueError('urgent_ratio outside [0,1]')
        self.max_running, self.max_pending = max_running, max_pending
        self.urgent_ratio, self.clock, self.emit = urgent_ratio, clock, emit
        self.tickets = {}
        self.sequence = 0
        self.benefit_queue = []
        self.waiting_queue = []

    def submit(self, *, request_id, session_id, snapshot_id, deadline,
               remaining_cost_s, expected_relative_ttft_gain, estimate_source):
        if not all(isinstance(x, str) and x for x in
                   [request_id, session_id, snapshot_id, estimate_source]):
            raise ValueError('identities and estimate provenance required')
        if request_id in self.tickets:
            raise ValueError('duplicate request id')
        if not all(type(x) in (int, float) and math.isfinite(x) for x in
                   [deadline, remaining_cost_s, expected_relative_ttft_gain]):
            raise ValueError('finite timing and gain estimates required')
        # The causal TTFT ratio is signed and intentionally unclipped.  A
        # regression can be below -1 when the first post request is much
        # slower than its explicitly bound pre request.
        if remaining_cost_s <= 0:
            raise ValueError('invalid pipeline cost')
        now = self.clock()
        self._expire(now)
        if sum(t.state == 'pending' for t in self.tickets.values()) >= self.max_pending:
            raise OverflowError('optional admission queue full')
        self.sequence += 1
        ticket = Ticket(request_id, session_id, snapshot_id, float(deadline),
                        float(remaining_cost_s), float(expected_relative_ttft_gain),
                        estimate_source, now, self.sequence,
                        max(0., deadline - now - remaining_cost_s))
        self.tickets[request_id] = ticket
        self.emit('maintenance_queued', **self._fields(ticket, now))
        self._expire(now)
        return ticket

    def _fields(self, t, now):
        slack = t.deadline - now - t.remaining_cost_s
        ratio = slack / max(t.initial_wait_budget_s, 1e-9)
        return dict(request_id=t.request_id, session_id=t.session_id,
                    snapshot_id=t.snapshot_id, deadline=t.deadline,
                    remaining_cost_s=t.remaining_cost_s,
                    expected_relative_ttft_gain=t.expected_relative_ttft_gain,
                    estimate_source=t.estimate_source, state=t.state,
                    queue_wait_s=max(0., now-t.submitted_at), slack_s=slack,
                    waiting_slack_ratio=ratio, sequence=t.sequence)

    def _finish_pending(self, t, reason, now):
        assert t.state == 'pending'
        t.state = 'terminal'
        t.terminal_reason = reason
        self.emit('maintenance_not_dispatched', reason=reason, **self._fields(t, now))

    def _expire(self, now):
        for t in self.tickets.values():
            if t.state == 'pending' and t.deadline - now < t.remaining_cost_s:
                self._finish_pending(t, 'predicted_pipeline_no_longer_fits', now)

    def revise_cost(self, request_id, *, remaining_cost_s, estimate_source):
        t = self.tickets[request_id]
        if t.state != 'pending':
            raise ValueError('only queued cost estimates may be revised')
        if not math.isfinite(remaining_cost_s) or remaining_cost_s <= 0 or not estimate_source:
            raise ValueError('positive finite cost and provenance required')
        t.remaining_cost_s = remaining_cost_s
        t.estimate_source = estimate_source
        # Keep the original waiting allowance: elapsed waiting is not reset.
        self._expire(self.clock())

    def dispatch(self):
        now = self.clock()
        self._expire(now)
        pending = [t for t in self.tickets.values() if t.state == 'pending']
        # Rebuild the bounded index at a single clock sample, since slack changes.
        self.benefit_queue = [(-t.expected_relative_ttft_gain, t.sequence, t.request_id)
                              for t in pending]
        self.waiting_queue = [(self._fields(t, now)['waiting_slack_ratio'],
                               t.deadline-t.remaining_cost_s, t.sequence, t.request_id)
                              for t in pending]
        heapq.heapify(self.benefit_queue)
        heapq.heapify(self.waiting_queue)
        available = self.max_running - sum(t.state == 'running' for t in self.tickets.values())
        selected = []
        for _ in range(max(0, available)):
            while self.waiting_queue and self.tickets[self.waiting_queue[0][-1]].state != 'pending':
                heapq.heappop(self.waiting_queue)
            while self.benefit_queue and self.tickets[self.benefit_queue[0][-1]].state != 'pending':
                heapq.heappop(self.benefit_queue)
            if not self.benefit_queue:
                break
            urgent = self.waiting_queue and self.waiting_queue[0][0] <= self.urgent_ratio
            queue = self.waiting_queue if urgent else self.benefit_queue
            t = self.tickets[heapq.heappop(queue)[-1]]
            t.state = 'running'
            t.dispatched_at = now
            selected.append(t)
            self.emit('maintenance_dispatched', selected_queue='waiting' if urgent else 'benefit',
                      **self._fields(t, now))
        return selected

    def cancel(self, request_id, *, reason='tool_result_ready'):
        t = self.tickets[request_id]
        if t.state == 'pending':
            self._finish_pending(t, reason, self.clock())
        elif t.state == 'running':
            t.cancellation_requested = True
            # A caller must abort its owned native request and acknowledge it.
            self.emit('maintenance_native_cancel_required', reason=reason,
                      **self._fields(t, self.clock()))
        return t.state

    def acknowledge_terminal(self, request_id, *, terminal_request_id,
                             terminal_complete, native_quiescent, reason):
        t = self.tickets[request_id]
        if (t.state != 'running' or terminal_request_id != request_id or
                terminal_complete is not True or native_quiescent is not True):
            raise ValueError('exact owned native terminal proof required')
        t.state = 'terminal'
        t.terminal_reason = reason
        self.emit('maintenance_terminal', reason=reason, **self._fields(t, self.clock()))

    def abandon_before_native_submission(self, request_id, *, native_submitted):
        t = self.tickets[request_id]
        if t.state != 'running' or native_submitted is not False:
            raise ValueError('reserved work must be proved never submitted')
        t.state = 'terminal'
        t.terminal_reason = 'never_submitted_to_native_engine'
        self.emit('maintenance_not_dispatched', reason=t.terminal_reason,
                  **self._fields(t, self.clock()))

    def status(self):
        return {kind: [t.request_id for t in self.tickets.values() if t.state == kind]
                for kind in ['pending', 'running', 'terminal']}
