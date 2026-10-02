"""Deadline cancellation for optional native prefix-prefill requests.

An HTTP response from ``/abort_request`` only acknowledges delivery of an
abort request.  It is never treated as proof that the exact model request has
stopped.  A slot is reusable only after either the exact native HTTP response
is validated, the engine supplies a terminal proof, or the same engine epoch
has reported the exact request active and then inactive.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
import time
import uuid
from typing import Any, Awaitable, Callable

import aiohttp

from transport_terminal import StatusProtocolError, validate_status


@dataclass(frozen=True)
class PrefillOutcome:
    status: int | None
    body: dict[str, Any] | None
    deadline_expired: bool
    terminal_confirmed: bool
    proof: str
    engine_epoch: str | None
    status_attempts: int
    status_errors: int
    abort_attempts: int


class PrefillTerminalUnconfirmed(RuntimeError):
    """The native request may still execute; its concurrency slot stays held."""

    def __init__(self, message: str, *, native_rid: str,
                 native_task: asyncio.Task, audit: dict[str, Any]):
        super().__init__(message)
        self.native_rid = native_rid
        self.native_task = native_task
        self.audit = audit


def _remaining(deadline: float) -> float:
    return max(0.0, deadline - time.monotonic())


async def run_deadline_bound_prefill(
    *,
    client: aiohttp.ClientSession,
    upstream: str,
    native: dict[str, Any],
    deadline_mono: float | None,
    terminal_wait_seconds: float,
    engine_control: Callable[[dict[str, Any]], Awaitable[dict[str, Any]]],
    emit: Callable[..., Any],
    validate_http_terminal: Callable[[int, Any], bool],
) -> PrefillOutcome:
    """Submit one exact native request and abort it after its tool deadline.

    The native HTTP task is independently owned.  Cancelling the caller does
    not constitute terminal proof.  If cleanup cannot be proven within the
    bounded confirmation interval, ``PrefillTerminalUnconfirmed`` carries the
    still-owned task back to the proxy, which must retain its prefill slot and
    fail closed.
    """
    native_rid = native.get("rid")
    if not isinstance(native_rid, str) or not native_rid:
        raise ValueError("native prefix-prefill request ID is required")
    if (type(terminal_wait_seconds) not in (int, float)
            or not 0.05 <= float(terminal_wait_seconds) <= 1800):
        raise ValueError("prefill terminal wait must be in [0.05,1800] seconds")

    async def native_post() -> tuple[int, Any, float]:
        async with client.post(upstream.rstrip("/") + "/generate", json=native) as response:
            try:
                body = await response.json()
            except Exception:
                # A malformed or truncated body is not terminal proof.  Keep
                # the transport result for the status/abort recovery path.
                body = None
            return response.status, body, time.monotonic()

    native_task = asyncio.create_task(native_post())

    async def exact_http_terminal() -> tuple[int, Any] | None:
        if not native_task.done() or native_task.cancelled():
            return None
        try:
            status, body, _ = native_task.result()
        except BaseException:
            return None
        return (status, body) if validate_http_terminal(status, body) else None

    # An unbudgeted legacy call retains the v14 behavior.  ToolSlack's optional
    # L2 path supplies an absolute deadline and therefore uses the branch below.
    if deadline_mono is None:
        status, body, _ = await native_task
        return PrefillOutcome(
            status=status,
            body=body,
            deadline_expired=False,
            terminal_confirmed=validate_http_terminal(status, body),
            proof="exact_native_http_terminal" if validate_http_terminal(status, body)
                  else "unconfirmed_http_response",
            engine_epoch=None,
            status_attempts=0,
            status_errors=0,
            abort_attempts=0,
        )

    delay = deadline_mono - time.monotonic()
    if delay <= 0:
        native_task.cancel()
        await asyncio.gather(native_task, return_exceptions=True)
        raise ValueError("tool deadline expired before native prefix-prefill submission")
    done, _ = await asyncio.wait({native_task}, timeout=delay)
    deadline_expired = not bool(done)
    if done:
        status, body, completed_mono = await native_task
        terminal = validate_http_terminal(status, body)
        if terminal:
            return PrefillOutcome(
                status=status,
                body=body,
                deadline_expired=completed_mono > deadline_mono,
                terminal_confirmed=True,
                proof="exact_native_http_terminal",
                engine_epoch=None,
                status_attempts=0,
                status_errors=0,
                abort_attempts=0,
            )
        # A non-terminal/malformed response after native submission is handled
        # by the same exact-rid recovery below; it is never a free slot.

    emit("kv_prefill_deadline_expired" if deadline_expired
         else "kv_prefill_http_terminal_unconfirmed",
         request_id=native_rid,
         deadline_monotonic=deadline_mono,
         deadline_overrun_seconds=max(0.0, time.monotonic() - deadline_mono),
         abort_ack_is_terminal_proof=False)

    cleanup_deadline = time.monotonic() + float(terminal_wait_seconds)
    observed_active = False
    engine_epoch = None
    status_attempts = 0
    status_errors = 0
    abort_attempts = 0
    fatal_protocol_error = None
    next_abort_at = time.monotonic()

    async def observe_status(max_wait: float) -> tuple[bool, bool] | None:
        nonlocal observed_active, engine_epoch, status_attempts, status_errors
        nonlocal fatal_protocol_error
        remaining = _remaining(cleanup_deadline)
        if remaining <= 0:
            return None
        status_attempts += 1
        operation_id = (
            f"{native_rid}:deadline-status:{status_attempts}:"
            f"{uuid.uuid4().hex[:8]}"
        )
        try:
            receipt = await asyncio.wait_for(
                engine_control({
                    "request_id": operation_id,
                    "action": "request_status",
                    "consumer_request_id": native_rid,
                }),
                timeout=min(remaining, max_wait),
            )
            active, terminal_proof, engine_epoch = validate_status(
                receipt, rid=native_rid, expected_epoch=engine_epoch
            )
            if active:
                observed_active = True
            emit("kv_prefill_terminal_status", request_id=native_rid,
                 active=active, terminal_proof=terminal_proof,
                 engine_epoch=engine_epoch, status_attempt=status_attempts,
                 abort_attempts=abort_attempts)
            return active, terminal_proof
        except StatusProtocolError as exc:
            fatal_protocol_error = f"{type(exc).__name__}: {exc}"
            status_errors += 1
            emit("kv_prefill_terminal_status_error", request_id=native_rid,
                 error_type=type(exc).__name__, error=str(exc)[:500],
                 status_attempt=status_attempts,
                 terminal_proof_disqualified=True)
        except Exception as exc:
            status_errors += 1
            emit("kv_prefill_terminal_status_error", request_id=native_rid,
                 error_type=type(exc).__name__, error=str(exc)[:500],
                 status_attempt=status_attempts)
        return None

    async def abort_once() -> None:
        nonlocal abort_attempts
        abort_attempts += 1
        remaining = _remaining(cleanup_deadline)
        if remaining <= 0:
            return
        try:
            async with client.post(
                upstream.rstrip("/") + "/abort_request",
                json={"rid": native_rid, "abort_all": False},
                timeout=aiohttp.ClientTimeout(total=min(2.0, remaining)),
            ) as response:
                await response.read()
                emit("kv_prefill_abort_ack", request_id=native_rid,
                     status=response.status, abort_attempt=abort_attempts,
                     terminal_proof=False)
        except Exception as exc:
            emit("kv_prefill_abort_error", request_id=native_rid,
                 error_type=type(exc).__name__, abort_attempt=abort_attempts)

    # Observe active before the first abort whenever the real control plane can
    # answer promptly.  Initial inactive without a durable proof is ambiguous.
    initial = await observe_status(max_wait=min(0.20, terminal_wait_seconds / 3))
    if fatal_protocol_error is None and initial is not None and initial[1]:
        if not native_task.done():
            native_task.cancel()
            await asyncio.gather(native_task, return_exceptions=True)
        return PrefillOutcome(None, None, deadline_expired, True,
                              "engine_terminal_proof", engine_epoch,
                              status_attempts, status_errors, abort_attempts)

    await abort_once()
    next_abort_at = time.monotonic() + 0.1

    while time.monotonic() < cleanup_deadline:
        terminal_http = await exact_http_terminal()
        if terminal_http is not None:
            status, body = terminal_http
            return PrefillOutcome(status, body, deadline_expired, True,
                                  "exact_native_http_terminal", engine_epoch,
                                  status_attempts, status_errors, abort_attempts)

        observed = await observe_status(max_wait=0.20)
        if fatal_protocol_error is None and observed is not None:
            active, terminal_proof = observed
            if not active and (terminal_proof or observed_active):
                if not native_task.done():
                    native_task.cancel()
                    await asyncio.gather(native_task, return_exceptions=True)
                return PrefillOutcome(
                    None, None, deadline_expired, True,
                    "engine_terminal_proof" if terminal_proof
                    else "observed_active_then_inactive",
                    engine_epoch, status_attempts, status_errors, abort_attempts,
                )
        if time.monotonic() >= next_abort_at:
            await abort_once()
            next_abort_at = time.monotonic() + 0.1
        await asyncio.sleep(min(0.02, _remaining(cleanup_deadline)))

    audit = {
        "terminal_confirmed": False,
        "deadline_expired": deadline_expired,
        "observed_active": observed_active,
        "engine_epoch": engine_epoch,
        "status_attempts": status_attempts,
        "status_errors": status_errors,
        "abort_attempts": abort_attempts,
        "protocol_error": fatal_protocol_error,
        "abort_ack_is_terminal_proof": False,
    }
    emit("kv_prefill_terminal_unconfirmed", request_id=native_rid, audit=audit,
         prefill_slot_released=False)
    raise PrefillTerminalUnconfirmed(
        "native prefix-prefill terminal state is unconfirmed",
        native_rid=native_rid,
        native_task=native_task,
        audit=audit,
    )
