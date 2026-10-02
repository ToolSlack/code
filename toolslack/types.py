"""Shared state; candidate text and engine KV identities are distinct."""
from dataclasses import dataclass, field
from typing import Any
import hashlib
import json
import math


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class Scope:
    stop: int
    input_tokens: int
    new_prefix_tokens: int
    gain_l1_s: float
    gain_l2_s: float
    start: int = 0
    source_sha256: str = ""

    def __post_init__(self):
        if not (type(self.start) is int and type(self.stop) is int and 0 <= self.start < self.stop):
            raise ValueError("Scope must cover complete, ordered message indices")
        for name in ("input_tokens", "new_prefix_tokens"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError("Token counts must be nonnegative exact integers")
        for name in ("gain_l1_s", "gain_l2_s"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError("Gain must be finite; negative observations are retained")


@dataclass(frozen=True)
class Plan:
    level: int
    scope: Scope
    kv_tokens: int
    memory_s: float
    kv_s: float
    gain_s: float
    deadline: float
    initial_budget: float
    baseline_ttft_s: float = 1.0

    def __post_init__(self):
        if self.level not in (1, 2):
            raise ValueError("An admitted optional plan is L1 or L2; no plan represents L0")
        if type(self.kv_tokens) is not int or self.kv_tokens < 0:
            raise ValueError("KV token count must be nonnegative")
        for name in ("memory_s", "kv_s", "gain_s", "deadline", "initial_budget", "baseline_ttft_s"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError("Plan costs, times, and benefit must be finite and nonnegative")
        if self.initial_budget <= 0 or self.baseline_ttft_s <= 0:
            raise ValueError("Initial effective budget and baseline TTFT must be positive")


@dataclass
class NativeResult:
    # Only the selected old scope is replaced. The controller retains all
    # protected instructions, unselected history, and arriving tool results.
    replacement_messages: list[dict]
    stable_body: dict
    metadata: dict = field(default_factory=dict)


@dataclass
class KVHandle:
    prefix_sha256: str
    token_count: int
    handle_id: str
    model_key: str
    epoch: str
    location: str = "hbm"
    ready: bool = True
    metadata: dict = field(default_factory=dict)


@dataclass
class ResumeState:
    level: int
    messages: list[dict]
    kv_handle: KVHandle | None = None
    reason: str = "original_context"
    request_metadata: dict = field(default_factory=dict)
