"""Fail-closed native-terminal recovery after a foreground transport loss.

This module never retries model generation and never turns the failed HTTP
request into a successful task.  It can only prove that the exact native
request has stopped, allowing unrelated work to continue safely.
"""
from __future__ import annotations

import asyncio
import math
import time
import uuid
from typing import Any, Awaitable, Callable


class StatusProtocolError(RuntimeError):
    pass


def validate_wait_seconds(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("transport terminal wait must be numeric")
    value = float(value)
    if not math.isfinite(value) or not 0.05 <= value <= 30.0:
        raise ValueError("transport terminal wait outside [0.05,30] seconds")
    return value


def validate_status(receipt: dict[str, Any], *, rid: str,
                    expected_epoch: str | None) -> tuple[bool, bool, str]:
    if receipt.get("ok") is not True:
        raise StatusProtocolError("native request_status was rejected")
    if receipt.get("consumer_request_id") != rid:
        raise StatusProtocolError("native request_status consumer identity differs")
    active = receipt.get("active")
    terminal_proof = receipt.get("terminal_proof")
    epoch = receipt.get("engine_epoch")
    if type(active) is not bool or type(terminal_proof) is not bool:
        raise StatusProtocolError("native request_status booleans are malformed")
    if not isinstance(epoch, str) or not epoch:
        raise StatusProtocolError("native request_status service epoch is missing")
    if expected_epoch is not None and epoch != expected_epoch:
        raise StatusProtocolError("native request_status service epoch changed")
    if terminal_proof and active:
        raise StatusProtocolError("terminal proof conflicts with active request state")
    return active, terminal_proof, epoch


async def confirm_foreground_terminal(
    *,
    rid: str,
    common: dict[str, Any],
    owner: Any,
    engine_control: Callable[[dict[str, Any]], Awaitable[dict[str, Any]]],
    emit: Callable[..., Any],
    wait_seconds: float,
    poll_seconds: float = 0.02,
) -> dict[str, Any]:
    """Confirm terminal only from an exact service-bound status history.

    An inactive response without an engine-provided terminal proof is
    ambiguous until this recovery loop has observed the same request active in
    the same service epoch.  This prevents a pre-registration `active=false`
    race from freeing resources for work that can still start later.
    """
    wait_seconds = validate_wait_seconds(wait_seconds)
    if not isinstance(rid, str) or not rid:
        raise ValueError("exact native request identity is required")
    deadline = time.monotonic() + wait_seconds
    epoch = None
    observed_active = False
    status_attempts = 0
    status_errors = 0
    fatal_protocol_error = None

    async def observe_status(*, max_wait: float | None = None) -> tuple[bool, bool] | None:
        nonlocal epoch, observed_active, status_attempts, status_errors
        nonlocal fatal_protocol_error
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None
        status_attempts += 1
        control_id = f"{rid}:transport-status:{status_attempts}:{uuid.uuid4().hex[:8]}"
        try:
            # The overall recovery budget, not the control client's longer
            # transport timeout, bounds every status call.
            timeout = remaining if max_wait is None else min(remaining, max_wait)
            receipt = await asyncio.wait_for(
                engine_control({
                    "request_id": control_id,
                    "action": "request_status",
                    "consumer_request_id": rid,
                }),
                timeout=timeout,
            )
            active, terminal_proof, epoch = validate_status(
                receipt, rid=rid, expected_epoch=epoch
            )
            emit("native_transport_status", **common, active=active,
                 terminal_proof=terminal_proof, engine_epoch=epoch,
                 status_attempt=status_attempts,
                 abort_attempts=owner.abort_attempts)
            if active:
                observed_active = True
            return active, terminal_proof
        except StatusProtocolError as exc:
            # An identity/epoch/protocol violation is permanently unsafe for
            # this recovery.  Still send the exact abort below, then keep the
            # bounded status audit without accepting a later receipt.
            fatal_protocol_error = f"{type(exc).__name__}: {exc}"
            status_errors += 1
            emit("native_transport_status_error", **common,
                 error_type=type(exc).__name__, error=str(exc)[:500],
                 status_attempt=status_attempts,
                 abort_attempts=owner.abort_attempts,
                 terminal_proof_disqualified=True)
        except Exception as exc:
            status_errors += 1
            emit("native_transport_status_error", **common,
                 error_type=type(exc).__name__, error=str(exc)[:500],
                 status_attempt=status_attempts,
                 abort_attempts=owner.abort_attempts)
        return None

    # Query before abort so a real engine whose request_status has no durable
    # terminal ledger can expose the only unambiguous active observation.
    # Reserve enough of the advertised budget for CancellationOwner's exact
    # abort loop even when the control endpoint stalls.  Its submitted-request
    # cadence is 100 ms; half the budget is reserved for very small test/run
    # budgets and 150 ms for normal ones.
    abort_reserve = min(0.15, wait_seconds / 2)
    initial = await observe_status(max_wait=max(0.001, wait_seconds - abort_reserve))
    owner.cancel("upstream_transport_failure")
    while owner.abort_attempts == 0 and time.monotonic() < deadline:
        await asyncio.sleep(min(0.005, max(0.0, deadline - time.monotonic())))
    if (
        fatal_protocol_error is None
        and initial is not None
        and initial[1] is True
    ):
        return {
            "terminal_confirmed": True,
            "proof": "engine_terminal_proof",
            "engine_epoch": epoch,
            "status_attempts": status_attempts,
            "status_errors": status_errors,
            "abort_attempts": owner.abort_attempts,
        }

    while time.monotonic() < deadline:
        observed = await observe_status()
        if fatal_protocol_error is None and observed is not None:
            active, terminal_proof = observed
            if not active and (terminal_proof or observed_active):
                return {
                    "terminal_confirmed": True,
                    "proof": "engine_terminal_proof" if terminal_proof
                             else "observed_active_then_inactive",
                    "engine_epoch": epoch,
                    "status_attempts": status_attempts,
                    "status_errors": status_errors,
                    "abort_attempts": owner.abort_attempts,
                }
        remaining = deadline - time.monotonic()
        if remaining > 0:
            await asyncio.sleep(min(poll_seconds, remaining))
    return {
        "terminal_confirmed": False,
        "proof": None,
        "engine_epoch": epoch,
        "observed_active": observed_active,
        "status_attempts": status_attempts,
        "status_errors": status_errors,
        "abort_attempts": owner.abort_attempts,
        "protocol_error": fatal_protocol_error,
    }
