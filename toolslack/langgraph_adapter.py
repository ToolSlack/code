"""Adapter for the installed official LangMem short-term memory algorithm.

Only closed selected messages enter LangMem. Its returned messages replace that
scope; instructions, unselected history and unresolved tool dispatches survive.
Optional KV sees a closed body plus the known future dispatch body separately.
Dependencies are imported lazily so closure/identity tests run without a GPU.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import inspect
import hashlib
import json
from typing import Any, Callable

from .context import check_scope, materialize
from .types import NativeResult, Scope, digest


class NativeMemoryNoop(RuntimeError):
    """Official LangMem did not produce a semantic-memory update."""


def wire_to_langchain(messages):
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
    result = []
    for index, message in enumerate(messages):
        role, content = message.get("role"), message.get("content") or ""
        identity = message.get("id") or f"toolslack-wire-{index}-{digest(message)[:12]}"
        if role == "system":
            value = SystemMessage(content=content, id=identity)
        elif role == "user":
            value = HumanMessage(content=content, id=identity)
        elif role == "assistant":
            calls = []
            for call in message.get("tool_calls") or []:
                function = call["function"]
                arguments = function.get("arguments", {})
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                calls.append(dict(id=call["id"], name=function["name"], args=arguments,
                                  type="tool_call"))
            value = AIMessage(content=content, tool_calls=calls, id=identity)
        elif role == "tool":
            value = ToolMessage(content=content, tool_call_id=message["tool_call_id"], id=identity)
        else:
            raise ValueError(f"Unsupported native memory role: {role}")
        result.append(value)
    return result


def langchain_to_wire(messages):
    roles = {"human": "user", "ai": "assistant", "system": "system", "tool": "tool"}
    result = []
    for message in messages:
        role = roles[message.type]
        row = dict(role=role, content=message.content)
        if role == "tool":
            row["tool_call_id"] = message.tool_call_id
        elif role == "assistant" and getattr(message, "tool_calls", None):
            row["tool_calls"] = [dict(id=call["id"], type="function", function=dict(
                name=call["name"], arguments=json.dumps(call["args"], ensure_ascii=False)))
                for call in message.tool_calls]
        result.append(row)
    return result


def closed_prefix_stop(messages: list[dict]) -> int:
    """Last message boundary whose complete tool dependencies are resolved."""
    outstanding, seen = set(), set()
    last = 0
    for index, message in enumerate(messages):
        for call in message.get("tool_calls") or []:
            identifier = call.get("id")
            if not identifier or identifier in seen:
                raise ValueError("Malformed or duplicate native tool dispatch")
            outstanding.add(identifier);seen.add(identifier)
        if message.get("role") == "tool":
            identifier = message.get("tool_call_id")
            if identifier not in outstanding:
                raise ValueError("Orphan native tool result")
            outstanding.remove(identifier)
        if not outstanding:
            last = index + 1
    return last


def legal_scope_boundaries(messages: list[dict], protected_from: int,
                           max_scopes: int = 3) -> list[tuple[int, int]]:
    """Bounded complete-message scopes; never divide summary generation."""
    if max_scopes < 1:
        raise ValueError("At least one candidate scope is required")
    start = 0
    while start < protected_from and messages[start].get("role") in ("system", "developer"):
        start += 1
    legal = []
    for stop in range(start + 1, protected_from + 1):
        if stop - start < 2:
            continue
        trial = Scope(stop, 0, 0, 0., 0., start=start)
        try:
            check_scope(messages, trial, protected_from)
        except ValueError:
            continue
        legal.append((start, stop))
    if len(legal) <= max_scopes:
        return legal
    indices = {round((len(legal)-1)*i/(max_scopes-1)) for i in range(max_scopes)} if max_scopes > 1 else {len(legal)-1}
    return [legal[index] for index in sorted(indices)]


@dataclass(frozen=True)
class LangMemSettings:
    max_tokens: int = 8192
    trigger_tokens: int = 4096
    summary_tokens: int = 384

    def __post_init__(self):
        if min(self.max_tokens, self.trigger_tokens, self.summary_tokens) <= 0:
            raise ValueError("Native LangMem thresholds must be positive")
        if self.summary_tokens >= self.max_tokens:
            raise ValueError("Summary token reserve must fit the native limit")


class LangGraphLangMemAdapter:
    def __init__(self, model_key: str, settings: LangMemSettings, token_counter: Callable,
                 model_call: Callable, *, body_settings: dict | None = None,
                 exact_body_counter: Callable | None = None, event_sink=None,
                 native_summarizer=None, to_native=wire_to_langchain,
                 from_native=langchain_to_wire, runnable_factory=None):
        self.model_key, self.settings = model_key, settings
        self.token_counter, self.model_call = token_counter, model_call
        self.body_settings = deepcopy(body_settings or {})
        self.body_settings.setdefault("model", model_key)
        if self.body_settings["model"] != model_key or "messages" in self.body_settings:
            raise ValueError("Body scaffold must have the exact model and no messages")
        self.exact_body_counter = exact_body_counter
        self.event_sink = event_sink or (lambda row: None)
        self.native_summarizer = native_summarizer
        self.to_native, self.from_native = to_native, from_native
        self.runnable_factory = runnable_factory
        self._snapshot_contexts = {}

    def body(self, messages):
        return dict(deepcopy(self.body_settings), messages=deepcopy(messages))

    def bind_snapshot(self, snapshot, *, task_id, protected_from, required_memory=False):
        """Bind adapter bookkeeping outside the exact native request scaffold."""
        self._snapshot_contexts[digest(snapshot)] = dict(task_id=task_id,
            protected_from=protected_from,required_memory=required_memory)

    async def compact(self, snapshot: dict, scope: Scope) -> NativeResult:
        messages = deepcopy(snapshot["messages"])
        context = self._snapshot_contexts.get(digest(snapshot),{})
        protected = int(context.get("protected_from",snapshot.get("protected_from",closed_prefix_stop(messages))))
        task_id = context.get("task_id",snapshot.get("task_id"))
        call_context = dict(snapshot,task_id=task_id,
                            required_memory=context.get("required_memory",snapshot.get("required_memory",False)))
        check_scope(messages, scope, protected)
        selected = messages[scope.start:scope.stop]
        summarizer = self.native_summarizer
        if summarizer is None:
            from langmem.short_term import asummarize_messages
            summarizer = asummarize_messages
        runnable_factory = self.runnable_factory
        if runnable_factory is None:
            from langchain_core.runnables import RunnableLambda
            runnable_factory = RunnableLambda

        async def invoke(actual_messages):
            wire = self.from_native(actual_messages)
            text = await self.model_call(wire, call_context, self.settings.summary_tokens)
            if self.native_summarizer is not None:
                return text
            from langchain_core.messages import AIMessage
            return AIMessage(content=text)

        def count(actual_messages):
            return self.token_counter(self.from_native(actual_messages))

        self.event_sink(dict(event="native_memory_input", task_id=task_id,
                             scope_start=scope.start, scope_stop=scope.stop,
                             selected_sha256=digest(selected), native_algorithm="langmem.short_term.asummarize_messages"))
        result = await summarizer(self.to_native(selected), running_summary=None,
            model=runnable_factory(invoke), max_tokens=self.settings.max_tokens,
            max_tokens_before_summary=self.settings.trigger_tokens,
            max_summary_tokens=self.settings.summary_tokens, token_counter=count)
        if getattr(result, "running_summary", None) is None:
            raise NativeMemoryNoop("Official LangMem returned no summary; original context retained")
        replacement = self.from_native(result.messages)
        if closed_prefix_stop(replacement) != len(replacement):
            raise ValueError("Native memory produced an unresolved tool dependency")
        candidate = materialize(messages, scope, replacement)
        stop = closed_prefix_stop(candidate)
        stable_body = self.body(candidate[:stop])
        known_future = self.body(candidate)
        if self.exact_body_counter is None:
            token_count = self.token_counter(candidate[:stop])
        else:
            token_count = await self.exact_body_counter(stable_body)
        if type(token_count) is not int or token_count <= 0:
            raise ValueError("Exact closed native prefix count required")
        try:
            source_hash = hashlib.sha256(inspect.getsource(summarizer).encode()).hexdigest()
        except (OSError, TypeError):
            source_hash = None
        metadata = dict(stable_prefix_tokens=token_count, known_future_body=known_future,
                        model_key=self.model_key, closed_prefix_messages=stop,
                        native_algorithm="langmem.short_term.asummarize_messages",
                        native_algorithm_source_sha256=source_hash, generated=True,
                        source_scope_sha256=digest(selected))
        self.event_sink(dict(event="native_memory_output", task_id=task_id,
                             replacement_sha256=digest(replacement), stable_prefix_tokens=token_count,
                             stable_body_sha256=digest(stable_body), known_future_sha256=digest(known_future)))
        return NativeResult(replacement, stable_body, metadata)
