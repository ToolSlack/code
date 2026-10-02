"""Event-driven coordination around existing native memory and SGLang KV APIs.

All optional work is asynchronous. The result-readiness method is synchronous
and consumes only states already prepared and validated at that boundary.
"""
import asyncio
from contextlib import asynccontextmanager
from copy import deepcopy
from dataclasses import dataclass, field
import time
import math
from typing import Any

from .context import check_scope, materialize, matches_old_prefix
from .langgraph_adapter import closed_prefix_stop
from .types import KVHandle, NativeResult, Plan, ResumeState, Scope, digest


@dataclass
class Window:
    window_id: str
    snapshot: dict
    scopes: list[Scope]
    prediction: Any
    protected_from: int
    required_tools: frozenset[str]
    admitted_at: float
    plan: Plan | None
    returned_tools: set[str] = field(default_factory=set)
    consumed: bool = False
    result: NativeResult | None = None
    candidate_messages: list[dict] | None = None
    kv: KVHandle | None = None
    task_ids: set[str] = field(default_factory=set)
    reason: str = "no_feasible_plan"
    placement_inflight: bool = False
    placement_task: asyncio.Task | None = None
    idle_declared: bool = False


@dataclass
class Job:
    job_id: str
    window: Window
    stage: str
    plan: Plan
    admitted_at: float
    sequence: int
    launched_at: float | None = None


class ToolSlackController:
    def __init__(self, backend, selector, *, max_running=2, max_prefill_tokens=131072,
                 foreground_limit=8, safety_margin_s=.05, urgent_ratio=.2,
                 max_stage_age_s=2.0, scheduler_policy="budget", adaptive_scope=True,
                 enable_kv=True, enable_tiering=False, minimum_offloaded_s=.2,
                 transfer_lead_s=None, idle_prefix_manager=None,
                 event_sink=None, clock=time.monotonic):
        if min(max_running, max_prefill_tokens, foreground_limit) <= 0:
            raise ValueError("Positive resource limits required")
        if scheduler_policy not in ("budget", "fifo"):
            raise ValueError("Scheduler policy must be budget or fifo")
        if safety_margin_s < 0 or not 0 <= urgent_ratio <= 1 or max_stage_age_s < 0:
            raise ValueError("Invalid safety or urgency parameters")
        if enable_tiering and (transfer_lead_s is None or transfer_lead_s <= 0):
            raise ValueError("Tiering requires an independently measured transfer lead")
        self.backend, self.selector, self.clock = backend, selector, clock
        if idle_prefix_manager is not None and idle_prefix_manager.backend is not backend:
            raise ValueError("Idle-prefix and maintenance ownership must share the same backend")
        self.idle_prefix_manager = idle_prefix_manager
        self.max_running, self.max_prefill_tokens = max_running, max_prefill_tokens
        self.foreground_limit, self.foreground_active = foreground_limit, 0
        self.safety_margin, self.urgent_ratio, self.max_stage_age = safety_margin_s, urgent_ratio, max_stage_age_s
        self.scheduler_policy, self.adaptive_scope = scheduler_policy, adaptive_scope
        self.enable_kv, self.enable_tiering = enable_kv, enable_tiering
        self.minimum_offloaded, self.transfer_lead = minimum_offloaded_s, transfer_lead_s
        self.event_sink = event_sink or (lambda row: None)
        self.windows: dict[str, Window] = {}
        self.pending: list[Job] = []
        self.running: dict[str, tuple[Job, asyncio.Task]] = {}
        self.background: set[asyncio.Task] = set()
        self.wake = asyncio.Event()
        self.sequence = 0
        self.pump = None
        self.closed = False
        self.events = []
        self.cleanup_errors = []
        self._close_task = None
        self._release_tasks = {}

    def emit(self, event, **fields):
        row = dict(event=event, monotonic_s=self.clock(), **fields)
        self.events.append(row);self.event_sink(row)

    async def start(self):
        if self.closed:
            raise RuntimeError("A closed controller cannot be restarted")
        if self.pump is None:
            self.pump = asyncio.create_task(self._dispatch_loop())
        return self

    def _select(self, window, now):
        scopes = window.scopes
        if not self.adaptive_scope and scopes:
            scopes = [max(scopes, key=lambda s: (s.stop-s.start, s.input_tokens))]
        lengths = tuple(sorted({s.new_prefix_tokens for s in scopes if s.new_prefix_tokens > 0})) if self.enable_kv else ()
        return self.selector.select(scopes, window.prediction, now=now,
                                    safety_margin=self.safety_margin, kv_lengths=lengths,
                                    queue_memory_s=0, queue_kv_s=self._queue_cost("kv"))

    def on_tool_dispatch(self, window_id: str, snapshot: dict, scopes: list[Scope],
                         prediction, *, protected_from: int, required_tools):
        if self.closed or self.pump is None:
            raise RuntimeError("Controller must be started and open")
        if not window_id or window_id in self.windows:
            raise ValueError("Fresh window identity required")
        required = frozenset(required_tools)
        if not required or any(not isinstance(x, str) or not x for x in required):
            raise ValueError("Required tool-result dependency set cannot be empty")
        copied = deepcopy(snapshot)
        if not isinstance(copied.get("messages"), list):
            raise ValueError("A framework-owned message snapshot is required")
        for scope in scopes:
            check_scope(copied["messages"], scope, protected_from)
        now = self.clock()
        window = Window(window_id, copied, list(scopes), prediction, protected_from,
                        required, now, None)
        window.plan = self._select(window, now)
        self.windows[window_id] = window
        self.emit("tool_dispatch", window_id=window_id, required_tools=sorted(required),
                  snapshot_sha256=digest(copied), scopes=len(scopes),
                  level=window.plan.level if window.plan else 0,
                  deadline=window.plan.deadline if window.plan else None)
        self._declare_idle_prefix(window)
        if window.plan:
            window.reason = "memory_pending"
            self._enqueue(window, "memory", window.plan)
        return window.plan

    def _declare_idle_prefix(self, window):
        manager = self.idle_prefix_manager
        if manager is None:
            return
        try:
            prediction = window.prediction
            deadline = prediction.deadline
            if not prediction.usable or not math.isfinite(deadline) or deadline <= self.clock()+self.safety_margin:
                self.emit("idle_prefix_optional_declined", window_id=window.window_id,
                          reason="uncertain_or_expired_tool_window")
                return
            key = getattr(self.selector, "key", None)
            signature = getattr(prediction, "signature", None)
            if key is None or signature is None or signature.model != key.model or signature.workload_key != key.workload_key:
                self.emit("idle_prefix_optional_declined", window_id=window.window_id,
                          reason="tool_prediction_profile_mismatch")
                return
            stable = deepcopy(window.snapshot)
            stop = closed_prefix_stop(stable["messages"])
            stable["messages"] = stable["messages"][:stop]
            manager.on_tool_dispatch(window.window_id, stable, deepcopy(window.snapshot), deadline)
            window.idle_declared = True
        except Exception as error:
            # Optional cache adoption must not prevent native agent execution.
            self.emit("idle_prefix_optional_declined", window_id=window.window_id,
                      reason=f"{type(error).__name__}: {error}")

    def _enqueue(self, window, stage, plan):
        if window.consumed or self.closed:
            return
        self.sequence += 1
        job = Job(f"{window.window_id}:{stage}:{self.sequence}", window, stage,
                  plan, self.clock(), self.sequence)
        window.task_ids.add(job.job_id);self.pending.append(job)
        self.emit("stage_admitted", job_id=job.job_id, window_id=window.window_id,
                  stage=stage, level=plan.level, scope_stop=plan.scope.stop,
                  kv_tokens=plan.kv_tokens, gain_s=plan.gain_s)
        self.wake.set()

    def _queue_cost(self, stage, exclude_job=None):
        now = self.clock()
        running = sum(max(0, (j.plan.memory_s if j.stage == "memory" else j.plan.kv_s)
                          - (now-(j.launched_at if j.launched_at is not None else now)))
                      for j, _ in self.running.values() if j is not exclude_job)
        waiting = sum(j.plan.kv_s if j.stage == "kv" else j.plan.memory_s
                      for j in self.pending if j.stage == stage)
        return (running+waiting)/self.max_running

    def _service_cost(self, job):
        if job.stage == "kv":
            return job.plan.kv_s
        return job.plan.memory_s + (job.plan.kv_s if job.plan.level == 2 else 0)

    def _path_cost(self, job):
        return self._service_cost(job) + (self._queue_cost("kv", exclude_job=job)
            if job.stage == "memory" and job.plan.level == 2 else 0)

    def _priority(self, job, now):
        if self.scheduler_policy == "fifo":
            return (0, job.sequence)
        allowance = job.plan.deadline-now-self.safety_margin-self._path_cost(job)
        rho = max(0, min(1, allowance/job.plan.initial_budget))
        age = now-job.admitted_at
        if rho <= self.urgent_ratio or age >= self.max_stage_age:
            return (0, rho, -age, job.plan.deadline, job.sequence)
        relative_gain = max(0, min(1, job.plan.gain_s/job.plan.baseline_ttft_s))
        utility = relative_gain/(self._service_cost(job)+1e-9)
        return (1, -utility, self._service_cost(job), job.admitted_at, job.sequence)

    def _refresh(self, job, now):
        window = job.window
        if window.consumed:
            return None
        if job.stage == "memory":
            return self._select(window, now)
        actual_prefix_tokens = window.result.metadata.get("stable_prefix_tokens") if window.result else None
        if type(actual_prefix_tokens) is not int or actual_prefix_tokens <= 0:
            return None
        return self.selector.rebudget_kv(window.plan, actual_prefix_tokens, now=now,
            safety_margin=self.safety_margin, kv_lengths=(job.plan.kv_tokens,), queue_kv_s=0)

    async def _dispatch_loop(self):
        try:
            while not self.closed:
                now = self.clock()
                refreshed = []
                for job in self.pending:
                    plan = self._refresh(job, now)
                    if plan is None:
                        self.emit("stage_retired", job_id=job.job_id, stage=job.stage,
                                  reason="consumed_or_no_longer_feasible")
                    else:
                        job.plan = plan;refreshed.append(job)
                self.pending = refreshed
                ordered = sorted(self.pending, key=lambda j:self._priority(j, now))
                for job in ordered:
                    if len(self.running) >= self.max_running:
                        break
                    reservation = job.plan.scope.input_tokens if job.stage == "memory" else job.plan.kv_tokens
                    active = sum(j.plan.scope.input_tokens if j.stage == "memory" else j.plan.kv_tokens
                                 for j, _ in self.running.values())
                    if active+reservation > self.max_prefill_tokens or self.foreground_active >= self.foreground_limit:
                        continue
                    latest = self.clock()
                    if latest+self._path_cost(job)+self.safety_margin > job.plan.deadline:
                        self.pending.remove(job)
                        self.emit("stage_retired", job_id=job.job_id, stage=job.stage, reason="dispatch_deadline")
                        continue
                    self.pending.remove(job)
                    if job.stage == "memory":
                        job.window.plan = job.plan
                    job.launched_at = latest
                    task = asyncio.create_task(self._execute(job))
                    self.running[job.job_id] = (job, task)
                    self.emit("stage_dispatched", job_id=job.job_id, stage=job.stage,
                              queue_s=latest-job.admitted_at, reserved_tokens=reservation,
                              remaining_s=job.plan.deadline-latest-self.safety_margin,
                              eligible_competitors=len(ordered)-1)
                self.wake.clear()
                # Reconsider expiry and aging even without another admission.
                delay = .05 if self.pending else None
                try:
                    await asyncio.wait_for(self.wake.wait(), timeout=delay)
                except asyncio.TimeoutError:
                    pass
        except asyncio.CancelledError:
            pass

    async def _execute(self, job):
        window = job.window
        try:
            if job.stage == "memory":
                result = await self.backend.compact(deepcopy(window.snapshot), job.plan.scope)
                if not isinstance(result, NativeResult):
                    raise TypeError("Native backend must return a NativeResult")
                candidate = materialize(window.snapshot["messages"], job.plan.scope,
                                        result.replacement_messages)
                self._validate_candidate(window, candidate, result)
                self._observe_cost(job, result.metadata, job.plan.scope.input_tokens)
                if self.closed or window.consumed or self.clock() >= job.plan.deadline:
                    window.reason = "memory_late"
                    self.emit("candidate_discarded", window_id=window.window_id, reason=window.reason)
                    return
                window.result = result;window.candidate_messages = candidate
                window.reason = "validated_l1"
                self.emit("l1_ready", window_id=window.window_id,
                          candidate_sha256=digest(candidate), scope_stop=job.plan.scope.stop)
                actual_tokens = result.metadata.get("stable_prefix_tokens")
                if self.enable_kv and type(actual_tokens) is int and actual_tokens > 0:
                    plan = self.selector.rebudget_kv(job.plan, actual_tokens, now=self.clock(),
                        safety_margin=self.safety_margin, kv_lengths=(actual_tokens,),
                        queue_kv_s=self._queue_cost("kv", exclude_job=job))
                    if plan:
                        self._enqueue(window, "kv", plan)
            else:
                result = await self.backend.prepare_kv(deepcopy(window.result.stable_body),
                    job.plan.kv_tokens, job.plan.deadline,
                    identity=dict(body_sha256=digest(window.result.stable_body),
                                  model_key=self.backend.model_key,
                                  known_future_body=deepcopy(window.result.metadata["known_future_body"]),
                                  window_id=window.window_id))
                if not isinstance(result, KVHandle) or not result.ready or result.location != "hbm":
                    raise ValueError("Engine did not confirm a resident exact-prefix KV handle")
                if result.token_count <= 0 or result.token_count > job.plan.kv_tokens:
                    await self._release_once(result)
                    raise ValueError("Engine prefix exceeds its admitted token scope")
                if result.model_key != self.backend.model_key or not self.backend.device_ready(result):
                    await self._release_once(result)
                    raise ValueError("Engine handle model or confirmed device readiness differs")
                self._observe_cost(job, result.metadata, job.plan.kv_tokens)
                if self.closed or window.consumed or self.clock() >= job.plan.deadline:
                    await self._release_once(result)
                    self.emit("candidate_discarded", window_id=window.window_id, reason="kv_late")
                    return
                window.kv = result;window.reason = "validated_l2"
                self.emit("l2_ready", window_id=window.window_id, kv_handle_id=result.handle_id,
                          engine_prefix_sha256=result.prefix_sha256, kv_tokens=result.token_count)
                if self.enable_tiering:
                    # The flag changes synchronously before creating a task. A
                    # tool-ready event in this gap must never consume tiered KV.
                    window.placement_inflight = True
                    window.placement_task = self._background(self._tier(window))
        except asyncio.CancelledError:
            # No upstream request is forgotten: backend cancellation/terminal
            # drain remains the backend's responsibility, and close() awaits it.
            self.emit("stage_cancelled", job_id=job.job_id, stage=job.stage)
            raise
        except Exception as exc:
            self.emit("stage_failed", job_id=job.job_id, stage=job.stage,
                      error=f"{type(exc).__name__}: {exc}")
        finally:
            elapsed = self.clock()-(job.launched_at if job.launched_at is not None else self.clock())
            self.emit("stage_completed", job_id=job.job_id, stage=job.stage, wall_s=elapsed,
                      consumed=window.consumed)
            self.running.pop(job.job_id, None);self.wake.set()

    def _background(self, awaitable):
        task = asyncio.create_task(awaitable)
        self.background.add(task);task.add_done_callback(self.background.discard)
        return task

    async def _tier(self, window):
        handle = window.kv
        remaining = window.plan.deadline-self.clock()-self.safety_margin
        try:
            if self.closed or window.consumed or remaining < 2*self.transfer_lead+self.minimum_offloaded:
                return
            receipt = await self.backend.place(handle, window.plan.deadline-self.safety_margin)
            self.emit("kv_placement_completed", window_id=window.window_id,
                      handle_id=handle.handle_id, receipt=receipt)
        except Exception as exc:
            handle.ready = False
            self.emit("kv_tiering_failed", window_id=window.window_id, error=str(exc))
        finally:
            window.placement_inflight = False

    @staticmethod
    def _validate_candidate(window, candidate, result):
        if not isinstance(result.stable_body, dict):
            raise ValueError("Native result requires a closed-prefix serialization body")
        prefix = result.stable_body.get("messages")
        if not isinstance(prefix, list) or not prefix or candidate[:len(prefix)] != prefix:
            raise ValueError("Stable KV messages must be an exact leading prefix of the candidate")
        # A native body may stop before pending tool dispatches. It must itself
        # contain every result for any tool dispatch it does include.
        outstanding, seen = set(), set()
        for message in prefix:
            for call in message.get("tool_calls") or []:
                identifier = call.get("id") if isinstance(call, dict) else None
                if not identifier or identifier in seen:
                    raise ValueError("Malformed or duplicate tool identity in stable prefix")
                outstanding.add(identifier);seen.add(identifier)
            if message.get("role") == "tool":
                identifier = message.get("tool_call_id")
                if identifier not in outstanding:
                    raise ValueError("Orphan tool result in stable prefix")
                outstanding.remove(identifier)
        if outstanding:
            raise ValueError("Stable prefix includes an unresolved tool dispatch")
        future = result.metadata.get("known_future_body")
        if not isinstance(future, dict) or future.get("messages") != candidate:
            raise ValueError("Known future serialization must preserve the full candidate")
        scaffold = lambda body: {k: v for k, v in body.items() if k != "messages"}
        if digest(scaffold(result.stable_body)) != digest(scaffold(future)):
            raise ValueError("Stable and future request scaffolds differ")
        if digest(scaffold(future)) != digest(scaffold(window.snapshot)):
            raise ValueError("Maintenance changed model, tools or native template settings")

    def _observe_cost(self, job, metadata, tokens):
        # Backend elapsed RPC time can include native queuing. Only explicit,
        # matching profile-key service/queue measurements may update the fit.
        key = getattr(self.selector, "key", None)
        profiles = getattr(self.selector, "profiles", None)
        measured_key = metadata.get("cost_profile_key")
        service = metadata.get("stage_service_s")
        queue = metadata.get("stage_queue_s", 0)
        if key is None or profiles is None or measured_key not in (key, vars(key)):
            return
        if key.model != self.backend.model_key:
            return
        if not all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in (service, queue)):
            return
        profiles.observe_cost(job.stage, tokens, service, key, queue_s=queue)
        self.emit("cost_profile_updated", job_id=job.job_id, stage=job.stage,
                  service_s=service, native_queue_s=queue, tokens=tokens)

    async def _release_once(self, handle):
        # Store ownership before awaiting. Concurrent cleanup paths share this
        # operation; the backend still owns confirmation of DMA and consumer drain.
        task = self._release_tasks.get(handle.handle_id)
        if task is None:
            task = asyncio.create_task(self.backend.release(handle))
            self._release_tasks[handle.handle_id] = task
        return await asyncio.shield(task)

    async def _retire_unused(self, window):
        try:
            if window.placement_task:
                await asyncio.shield(window.placement_task)
            if window.kv:
                await self._release_once(window.kv)
        except Exception as exc:
            self.cleanup_errors.append(str(exc))
            self.emit("kv_cleanup_failed", window_id=window.window_id, error=str(exc))

    def on_tool_result_ready(self, window_id, tool_id, current_messages):
        window = self.windows[window_id]
        if tool_id not in window.required_tools:
            raise ValueError("Unexpected tool result")
        if window.consumed:
            raise RuntimeError("A tool window can be consumed only once")
        if tool_id in window.returned_tools:
            raise ValueError("Duplicate tool result")
        window.returned_tools.add(tool_id)
        if window.returned_tools != set(window.required_tools):
            self.emit("tool_result_partial", window_id=window_id, tool_id=tool_id)
            return None
        window.consumed = True
        self.pending = [j for j in self.pending if j.window is not window]
        self.wake.set()
        if window.kv:
            self.backend.notify_tool_ready(window.kv)
        if window.result is None or not matches_old_prefix(window.snapshot["messages"], current_messages, window.plan.scope):
            state = ResumeState(0, deepcopy(current_messages), reason="unready_or_changed_prefix")
        else:
            updated = materialize(current_messages, window.plan.scope, window.result.replacement_messages)
            handle = window.kv
            prefix = window.result.stable_body["messages"]
            ready = (handle is not None and not window.placement_inflight
                     and updated[:len(prefix)] == prefix and self.backend.device_ready(handle))
            state = ResumeState(2 if ready else 1, updated, handle if ready else None,
                                reason="resident_prefix" if ready else "validated_text")
            if ready:
                state.request_metadata = {"headers": self.backend.consumer_headers(handle)}
        if window.kv and state.level != 2:
            self._background(self._retire_unused(window))
        original_handle = None
        if window.idle_declared:
            current_body = deepcopy(window.snapshot)
            current_body["messages"] = deepcopy(state.messages)
            try:
                original_handle = self.idle_prefix_manager.on_tool_result_ready(
                    window_id, current_body, state.level)
                if original_handle is not None and state.level == 0:
                    headers = self.backend.consumer_headers(original_handle)
                    state.kv_handle = original_handle
                    state.request_metadata = {"headers": headers}
                    state.reason = "original_resident_prefix"
            except Exception as error:
                original_handle = None
                self.emit("idle_prefix_optional_declined", window_id=window_id,
                          reason=f"{type(error).__name__}: {error}")
                self._background(self.idle_prefix_manager.retire(window_id))
        self.emit("tool_window_consumed", window_id=window_id, level=state.level,
                  reason=state.reason, optional_wait_s=0.0,
                  kv_origin="original" if original_handle is not None else "compacted" if state.level == 2 else None,
                  result_ready_s=self.clock())
        return state

    @asynccontextmanager
    async def foreground(self):
        self.foreground_active += 1;self.wake.set()
        try:
            yield
        finally:
            self.foreground_active -= 1;self.wake.set()

    async def close(self):
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._close())
        return await asyncio.shield(self._close_task)

    async def _close(self):
        self.closed = True;self.pending.clear();self.wake.set()
        for window in self.windows.values():
            window.consumed = True
            if window.kv:
                self.backend.notify_tool_ready(window.kv)
        if self.idle_prefix_manager is not None:
            try:
                await self.idle_prefix_manager.close(drain_backend=False)
            except Exception as error:
                self.cleanup_errors.append(str(error))
        if self.pump:
            await self.pump
        # Charging complete drain prevents late GPU work disappearing from QPS.
        if self.running:
            await asyncio.gather(*(t for _, t in list(self.running.values())), return_exceptions=True)
        # Tool-ready notifications wake tiering timers. Never cancel a placement
        # RPC while the engine may still own a DMA operation.
        if self.background:
            await asyncio.gather(*list(self.background), return_exceptions=True)
        if self._release_tasks:
            results = await asyncio.gather(*self._release_tasks.values(), return_exceptions=True)
            self.cleanup_errors.extend(str(r) for r in results if isinstance(r, Exception))
        # The adapter owns all adopted handles, including rejected receipts and
        # one-shot consumers. Its drain confirms their native terminal state.
        drain = getattr(self.backend, "drain", None)
        if drain is None:
            raise RuntimeError("Backend must provide native ownership drainage")
        await drain()
        self.emit("controller_drained", windows=len(self.windows), cleanup_errors=self.cleanup_errors)
        if self.cleanup_errors:
            raise RuntimeError("Unconfirmed native KV cleanup: "+"; ".join(self.cleanup_errors))
