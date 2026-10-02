"""Optional tiering of a waiting agent's already-computed ORIGINAL prefix KV.

Registration may only adopt an existing native cache hit. This manager never
prefills, generates tokens, or changes semantic memory. A ready original handle
can accompany L0 only; any compacted context supersedes it.
"""
from __future__ import annotations

import asyncio
from copy import deepcopy
from dataclasses import dataclass
import math
import time

from .langgraph_adapter import closed_prefix_stop
from .types import KVHandle, digest


@dataclass
class IdlePrefix:
    window_id: str
    stable_body: dict
    known_future_body: dict
    deadline: float
    identity: dict
    consumed: bool = False
    handle: KVHandle | None = None
    registration_task: asyncio.Task | None = None
    placement_task: asyncio.Task | None = None
    placement_inflight: bool = False
    retire_task: asyncio.Task | None = None


class IdlePrefixManager:
    def __init__(self, backend, *, enable_tiering=True, transfer_lead_s=None,
                 minimum_offloaded_s=.2, minimum_window_s=.05,
                 safety_margin_s=.05, event_sink=None, clock=time.monotonic):
        lead = transfer_lead_s if transfer_lead_s is not None else getattr(backend, "prefetch_lead_s", None)
        for value in (minimum_offloaded_s, minimum_window_s, safety_margin_s):
            if not math.isfinite(value) or value < 0:
                raise ValueError("Finite nonnegative idle-prefix timing required")
        if lead is not None and (not math.isfinite(lead) or lead <= 0):
            raise ValueError("Transfer lead must come from a positive measured estimate")
        self.backend, self.clock = backend, clock
        self.enable_tiering, self.transfer_lead_s = enable_tiering, lead
        self.minimum_offloaded_s, self.minimum_window_s = minimum_offloaded_s, minimum_window_s
        self.safety_margin_s = safety_margin_s
        self.event_sink = event_sink or (lambda event: None)
        self.windows: dict[str, IdlePrefix] = {}
        self.events = []
        self.closed = False
        self._close_task = None
        self._release_tasks = {}
        self.cleanup_errors = []

    def emit(self, event, **fields):
        row = dict(event=event, monotonic_s=self.clock(), **fields)
        self.events.append(row);self.event_sink(row)

    @staticmethod
    def _scaffold(body):
        return {key: value for key, value in body.items() if key != "messages"}

    def on_tool_dispatch(self, window_id, original_stable_body, known_future_body,
                         deadline, identity=None):
        """Declare a closed snapshot and launch cache registration in background."""
        if self.closed:
            raise RuntimeError("Idle-prefix manager is closed")
        if not window_id or window_id in self.windows:
            raise ValueError("A fresh idle-prefix window identity is required")
        stable, future = deepcopy(original_stable_body), deepcopy(known_future_body)
        if not isinstance(stable, dict) or not isinstance(future, dict):
            raise ValueError("Native request bodies are required")
        messages = stable.get("messages")
        if (not isinstance(messages, list) or not messages
                or closed_prefix_stop(messages) != len(messages)):
            raise ValueError("Original stable prefix must be closed")
        if future.get("messages", [])[:len(messages)] != messages:
            raise ValueError("Known tool dispatch does not preserve the original stable prefix")
        if digest(self._scaffold(stable)) != digest(self._scaffold(future)):
            raise ValueError("Original stable and future scaffolds differ")
        if not math.isfinite(deadline):
            raise ValueError("Finite predicted tool deadline required")
        expected = dict(model_key=self.backend.model_key, body_sha256=digest(stable), window_id=window_id)
        supplied = deepcopy(identity or {})
        for field in ("model_key", "body_sha256"):
            if field in supplied and supplied[field] != expected[field]:
                raise ValueError("Original prefix model or body identity differs")
        supplied.update(expected)
        record = IdlePrefix(window_id, stable, future, deadline, supplied)
        self.windows[window_id] = record
        record.registration_task = asyncio.create_task(self._register(record))
        self.emit("idle_prefix_declared", window_id=window_id, original_body_sha256=digest(stable),
                  deadline=deadline, original_messages=len(messages))
        return record

    async def _register(self, record):
        if self.closed or record.consumed:
            return
        if record.deadline-self.clock()-self.safety_margin_s < self.minimum_window_s:
            self.emit("idle_prefix_declined", window_id=record.window_id, reason="insufficient_window")
            return
        register = getattr(self.backend, "register_existing", None)
        if register is None:
            self.emit("idle_prefix_declined", window_id=record.window_id, reason="unsupported_native_cache_adoption")
            return
        try:
            handle = await register(deepcopy(record.stable_body), deepcopy(record.known_future_body),
                                    record.deadline-self.safety_margin_s, deepcopy(record.identity))
            if handle is None:
                self.emit("idle_prefix_declined", window_id=record.window_id, reason="original_prefix_not_resident")
                return
            if not isinstance(handle, KVHandle):
                raise TypeError("Existing-prefix registration must return a native handle or None")
            record.handle = handle
            if (handle.model_key != self.backend.model_key or handle.token_count <= 0
                    or not self.backend.device_ready(handle)):
                await self._release_once(handle)
                raise ValueError("Native original-prefix receipt is not device-ready for this model")
            if self.closed or record.consumed or self.clock() >= record.deadline:
                await self._release_once(handle)
                self.emit("idle_prefix_retired", window_id=record.window_id, reason="late_registration")
                return
            self.emit("idle_prefix_registered", window_id=record.window_id, handle_id=handle.handle_id,
                      token_count=handle.token_count, prefix_sha256=handle.prefix_sha256,
                      native_metadata=deepcopy(handle.metadata))
            remaining = record.deadline-self.clock()-self.safety_margin_s
            enough = (self.transfer_lead_s is not None
                      and remaining >= 2*self.transfer_lead_s+self.minimum_offloaded_s)
            if self.enable_tiering and enough:
                record.placement_inflight = True
                record.placement_task = asyncio.create_task(self._place(record))
            else:
                self.emit("idle_prefix_tiering_skipped", window_id=record.window_id,
                          reason="disabled_or_no_measured_transfer_budget")
        except Exception as error:
            self.emit("idle_prefix_registration_failed", window_id=record.window_id,
                      error=f"{type(error).__name__}: {error}")

    async def _place(self, record):
        try:
            if self.closed or record.consumed:
                return
            receipt = await self.backend.place(record.handle, record.deadline-self.safety_margin_s)
            self.emit("idle_prefix_placement_completed", window_id=record.window_id,
                      handle_id=record.handle.handle_id, receipt=receipt,
                      native_metadata=deepcopy(record.handle.metadata))
        except Exception as error:
            record.handle.ready = False
            self.emit("idle_prefix_placement_failed", window_id=record.window_id,
                      error=f"{type(error).__name__}: {error}")
        finally:
            record.placement_inflight = False

    def on_tool_result_ready(self, window_id, current_body, selected_level):
        """Return a confirmed original KV handle or None, without awaiting work.

        Returning an original handle does not change L0 into a memory-maintained
        level. The caller supplies its native consumer headers through the adapter.
        """
        if selected_level not in (0, 1, 2):
            raise ValueError("Selected semantic-maintenance level must be L0/L1/L2")
        record = self.windows[window_id]
        if record.consumed:
            raise RuntimeError("An idle-prefix window can be consumed only once")
        record.consumed = True
        handle = record.handle
        if handle is not None:
            self.backend.notify_tool_ready(handle)
        prefix = record.stable_body["messages"]
        matches = (isinstance(current_body, dict)
                   and isinstance(current_body.get("messages"), list)
                   and current_body["messages"][:len(prefix)] == prefix
                   and digest(self._scaffold(current_body)) == digest(self._scaffold(record.stable_body)))
        ready = (selected_level == 0 and matches and handle is not None
                 and not record.placement_inflight and self.backend.device_ready(handle))
        self.emit("idle_prefix_tool_ready", window_id=window_id, semantic_level=selected_level,
                  original_handle_usable=ready, optional_wait_s=0.0)
        if ready:
            return handle
        self._begin_retire(record)
        return None

    def _begin_retire(self, record):
        record.consumed = True
        if record.handle is not None:
            self.backend.notify_tool_ready(record.handle)
        if record.retire_task is None:
            record.retire_task = asyncio.create_task(self._retire(record))
        return record.retire_task

    async def _release_once(self, handle):
        task = self._release_tasks.get(handle.handle_id)
        if task is None:
            task = asyncio.create_task(self.backend.release(handle))
            self._release_tasks[handle.handle_id] = task
        return await asyncio.shield(task)

    async def _retire(self, record):
        try:
            if record.registration_task:
                await asyncio.shield(record.registration_task)
            if record.placement_task:
                await asyncio.shield(record.placement_task)
            if record.handle:
                await self._release_once(record.handle)
                self.emit("idle_prefix_retired", window_id=record.window_id, reason="superseded_or_consumed")
        except Exception as error:
            self.cleanup_errors.append(str(error))
            self.emit("idle_prefix_cleanup_failed", window_id=record.window_id, error=str(error))

    async def retire(self, window_id):
        """Call after the foreground consumer has finished using an L0 handle."""
        await asyncio.shield(self._begin_retire(self.windows[window_id]))

    async def close(self, *, drain_backend=True):
        """Shared-controller callers pass False and perform one final backend drain."""
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._close(drain_backend))
        await asyncio.shield(self._close_task)

    async def _close(self, drain_backend):
        self.closed = True
        await asyncio.gather(*(self._begin_retire(record) for record in self.windows.values()))
        if self._release_tasks:
            results = await asyncio.gather(*self._release_tasks.values(), return_exceptions=True)
            self.cleanup_errors.extend(str(r) for r in results if isinstance(r, Exception))
        if drain_backend:
            await self.backend.drain()
        if self.cleanup_errors:
            raise RuntimeError("Original-prefix native ownership could not drain: "+"; ".join(self.cleanup_errors))
        self.emit("idle_prefix_drained", windows=len(self.windows))
