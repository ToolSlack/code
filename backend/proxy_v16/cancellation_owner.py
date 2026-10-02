"""Own a native generation until its validated terminal response is consumed.

Downstream disconnection requests cancellation, but does not close the native
HTTP stream. An abort acknowledgement is deliberately not a terminal proof.
"""
from __future__ import annotations

import asyncio
import time

import aiohttp


class CancellationOwner:
    def __init__(self, request, client, upstream, common, event, *, timeout=30.0):
        self.transport = request.transport
        self.client = client
        self.upstream = upstream.rstrip("/")
        self.common = common
        self.event = event
        self.timeout = timeout
        self.cancelled_at = None
        self.submitted = False
        self.terminal = False
        self.worker = None
        self.watcher = None
        self.abort_attempts = 0
        self.cleanup_timed_out = False

    def disconnected(self):
        return self.transport is None or self.transport.is_closing()

    def cancel(self, reason):
        if self.cancelled_at is None:
            self.cancelled_at = time.monotonic()
            self.event("request_cancel_requested", **self.common, reason=reason)

    def start(self):
        self.worker = asyncio.current_task()
        self.watcher = asyncio.create_task(self._watch())

    async def _watch(self):
        while not self.terminal:
            if self.disconnected():
                self.cancel("downstream_disconnected")
            if self.cancelled_at is not None and self.submitted:
                if time.monotonic() - self.cancelled_at >= self.timeout:
                    self.cleanup_timed_out = True
                    self.event("native_abort_unconfirmed", **self.common,
                               attempts=self.abort_attempts)
                    # The worker's finally path poisons the run; a guard must
                    # close the service. This never releases an acquired KV ref.
                    self.worker.cancel()
                    return
                self.abort_attempts += 1
                try:
                    async with self.client.post(
                        self.upstream + "/abort_request",
                        json={"rid": self.common["request_id"], "abort_all": False},
                        timeout=aiohttp.ClientTimeout(total=2),
                    ) as response:
                        await response.read()
                        self.event("native_abort_ack", **self.common,
                                   status=response.status, attempt=self.abort_attempts,
                                   terminal_proof=False)
                except Exception as exc:
                    self.event("native_abort_error", **self.common,
                               error_type=type(exc).__name__, attempt=self.abort_attempts)
            # Retry handles a request that enters the tokenizer after the
            # initial abort. Admission may have awaited a pooled connection.
            await asyncio.sleep(0.05 if not self.submitted else 0.1)

    def confirm_terminal(self):
        self.terminal = True
        if self.cancelled_at is not None:
            self.event("native_cancel_terminal", **self.common,
                       cancel_to_terminal_seconds=time.monotonic() - self.cancelled_at,
                       abort_attempts=self.abort_attempts)

    async def prepare(self, response, request):
        if self.disconnected():
            self.cancel("downstream_disconnected_before_headers")
        if self.cancelled_at is not None:
            return
        try:
            await response.prepare(request)
        except (ConnectionError, RuntimeError):
            self.cancel("downstream_headers_failed")

    async def write(self, response, data, *, eof=False):
        if self.disconnected():
            self.cancel("downstream_disconnected_before_write")
        if self.cancelled_at is not None:
            return
        try:
            if eof:
                await response.write_eof()
            else:
                await response.write(data)
        except (ConnectionError, RuntimeError):
            self.cancel("downstream_write_failed")

    async def finish(self):
        if self.watcher:
            self.watcher.cancel()
            await asyncio.gather(self.watcher, return_exceptions=True)


async def shield_owned(worker, owner):
    """A cancelled web handler must not cancel its upstream-owning worker."""
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError:
        owner.cancel("web_handler_cancelled")
        raise
