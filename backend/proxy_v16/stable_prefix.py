"""Token-stable scope from two exact native request bodies."""
from __future__ import annotations

import json
from typing import Any, Callable


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _call_ids(message: dict[str, Any]) -> list[str]:
    result = []
    for call in message.get("tool_calls") or []:
        if isinstance(call, dict):
            value = call.get("id") or call.get("tool_call_id")
            if isinstance(value, str) and value:
                result.append(value)
    return result


def common_prefix(left: list[int], right: list[int]) -> list[int]:
    boundary = 0
    for first, second in zip(left, right):
        if first != second:
            break
        boundary += 1
    return left[:boundary]


def stable_known_request_prefix(
    closed_body: dict[str, Any],
    known_future_body: dict[str, Any],
    closed_ids: list[int],
    tokenize_closed: Callable[[dict[str, Any]], list[int]],
    future_position_prefix: Callable[[dict[str, Any], dict[str, Any]], list[int] | None] | None = None,
) -> tuple[list[int], dict[str, Any]]:
    """Cache only the LCP of closed and exact known-dispatch serializations."""

    closed_messages = closed_body.get("messages")
    future_messages = known_future_body.get("messages")
    if not isinstance(closed_messages, list) or not isinstance(future_messages, list):
        raise ValueError("both native request bodies require message lists")
    closed_scaffold = {key: value for key, value in closed_body.items() if key != "messages"}
    future_scaffold = {key: value for key, value in known_future_body.items() if key != "messages"}
    if _canonical(closed_scaffold) != _canonical(future_scaffold):
        raise ValueError("known future changed non-message request fields")
    if future_messages[: len(closed_messages)] != closed_messages:
        raise ValueError("closed messages are not an exact prefix of the known future")
    suffix = future_messages[len(closed_messages) :]
    if len(suffix) != 1 or suffix[0].get("role") != "assistant" or not _call_ids(suffix[0]):
        raise ValueError("known future must add exactly one assistant tool dispatch")
    known_future_ids = tokenize_closed(known_future_body)
    positioned = (future_position_prefix(closed_body, known_future_body)
                  if future_position_prefix is not None else None)
    if positioned is not None:
        if (not positioned or any(type(token) is not int for token in positioned)
                or known_future_ids[:len(positioned)] != positioned):
            raise ValueError("future-position scope is not an exact native prefix")
        stable = positioned
        mode = "native_future_position_completed_messages"
        tail_tokens = len(known_future_ids) - len(stable)
    else:
        stable = common_prefix(closed_ids, known_future_ids)
        mode = "known_future_body_common_prefix"
        tail_tokens = len(closed_ids) - len(stable)
    if not stable:
        raise ValueError("known dispatch leaves no token-stable prefix")
    return stable, {
        "mode": mode,
        "closed_snapshot_tokens": len(closed_ids),
        "known_future_tokens": len(known_future_ids),
        "stable_prefix_tokens": len(stable),
        "uncached_known_tail_tokens": tail_tokens,
        "tail_token_reference": "known_future" if positioned is not None else "standalone_closed",
        "unresolved_dispatch_tokens_cached": 0 if positioned is not None else None,
    }
