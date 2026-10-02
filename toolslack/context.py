"""Message-level closure checks; no token deletion or summarization."""
from copy import deepcopy
from .types import Scope, digest


def check_scope(messages: list[dict], scope: Scope, protected_from: int) -> None:
    if scope.stop > min(protected_from, len(messages)):
        raise ValueError("Maintenance crosses protected or unresolved history")
    selected = messages[scope.start:scope.stop]
    if not selected or any(m.get("role") in ("system", "developer") for m in selected):
        raise ValueError("Instructions must remain outside the native maintenance scope")
    pending = set()
    seen = set()
    for message in selected:
        for call in message.get("tool_calls") or []:
            identifier = call.get("id")
            if not identifier or identifier in seen:
                raise ValueError("Missing or duplicate tool-call identity")
            pending.add(identifier);seen.add(identifier)
        if message.get("role") == "tool":
            identifier = message.get("tool_call_id")
            if identifier not in pending:
                raise ValueError("Scope contains an orphan tool result")
            pending.remove(identifier)
    if pending:
        raise ValueError("Scope contains unresolved tool calls")
    if scope.source_sha256 and digest(selected) != scope.source_sha256:
        raise ValueError("Pre-counted native scope does not match the snapshot")


def materialize(original: list[dict], scope: Scope, replacement: list[dict]) -> list[dict]:
    if not replacement or any(not isinstance(m, dict) or not m.get("role") for m in replacement):
        raise ValueError("Native memory returned no valid replacement")
    return deepcopy(original[:scope.start] + replacement + original[scope.stop:])


def matches_old_prefix(snapshot: list[dict], current: list[dict], scope: Scope) -> bool:
    # The preceding instructions and entire replaced prefix must still match.
    # Appended tool results are allowed and are preserved by materialize().
    return len(current) >= scope.stop and digest(snapshot[:scope.stop]) == digest(current[:scope.stop])
