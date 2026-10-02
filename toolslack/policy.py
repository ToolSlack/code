"""Cached, bounded ToolSlack planning and ready-stage scheduling.

Selection never executes a model or performs calibration. All estimates come
from completed observations with the same model, workload, load and cache key.
Predicted tool deadlines are optional-work deadlines, never promises to an agent.
"""

from __future__ import annotations

import bisect
import math
from collections import defaultdict, deque
from dataclasses import dataclass, replace
from typing import Callable, Iterable, Sequence

from .types import Plan, Scope


def _nonnegative(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


def _stage(stage: str) -> str:
    if stage == "mem":
        return "memory"
    if stage not in {"memory", "kv"}:
        raise ValueError("stage must be memory or kv")
    return stage


def _quantile(values: Sequence[float], q: float) -> float:
    """An observed order statistic; the upper index is conservative for cost."""
    if not values:
        raise ValueError("quantile of empty observations")
    ordered = sorted(values)
    return ordered[math.ceil((len(ordered) - 1) * q)]


@dataclass(frozen=True)
class ProfileKey:
    model: str
    memory_algorithm: str
    output_settings: str
    load_key: str
    cache_key: str
    workload_key: str

    def __post_init__(self) -> None:
        if any(not isinstance(v, str) or not v for v in self.__dict__.values()):
            raise ValueError("all profile isolation fields must be nonempty strings")


@dataclass(frozen=True)
class StageCost:
    service_s: float
    queue_s: float = 0.0

    def __post_init__(self) -> None:
        _nonnegative(self.service_s, "service_s")
        _nonnegative(self.queue_s, "queue_s")

    @property
    def total_s(self) -> float:
        return self.service_s + self.queue_s


@dataclass(frozen=True)
class ToolSignature:
    model: str
    tool_name: str
    argument_bucket: str
    environment_key: str
    workload_key: str

    def __post_init__(self) -> None:
        if any(not isinstance(v, str) or not v for v in self.__dict__.values()):
            raise ValueError("tool history requires model, arguments, environment and workload")


@dataclass(frozen=True)
class WindowPrediction:
    signature: ToolSignature
    dispatch_time: float
    duration_s: float
    samples: int
    explicit_calibration: bool = False

    def __post_init__(self) -> None:
        if not math.isfinite(self.dispatch_time):
            raise ValueError("dispatch time must be finite")
        _nonnegative(self.duration_s, "duration_s")
        if self.samples < 0:
            raise ValueError("samples must be nonnegative")

    @property
    def deadline(self) -> float:
        return self.dispatch_time + self.duration_s

    @property
    def usable(self) -> bool:
        return self.duration_s > 0 and (self.samples >= 3 or self.explicit_calibration)

    def remaining(self, now: float, safety_margin: float = 0.0) -> float:
        return max(0.0, self.deadline - now - _nonnegative(safety_margin, "safety margin"))


class WindowPredictor:
    """Lower duration quantiles per full signature, with no tool-name pooling."""

    def __init__(self, lower_quantile: float = 0.2, max_samples: int = 128) -> None:
        if not 0 <= lower_quantile <= 0.5 or max_samples < 3:
            raise ValueError("lower quantile must be <= .5 and history must hold >= 3 samples")
        self.lower_quantile = lower_quantile
        self.max_samples = max_samples
        self._durations: dict[ToolSignature, deque[float]] = {}

    def observe(self, signature: ToolSignature, duration_s: float) -> None:
        duration = _nonnegative(duration_s, "tool duration")
        self._durations.setdefault(signature, deque(maxlen=self.max_samples)).append(duration)

    def predict(self, signature: ToolSignature, dispatch_time: float,
                explicit_calibration_s: float | None = None) -> WindowPrediction:
        durations = self._durations.get(signature, ())
        # Lower *observed* quantile uses the lower index; never an upper cost quantile.
        duration = 0.0
        if len(durations) >= 3:
            ordered = sorted(durations)
            duration = ordered[math.floor((len(ordered) - 1) * self.lower_quantile)]
        explicit = explicit_calibration_s is not None
        if explicit:
            calibrated = _nonnegative(explicit_calibration_s, "calibration duration")
            duration = min(duration, calibrated) if len(durations) >= 3 else calibrated
        return WindowPrediction(signature, dispatch_time, duration, len(durations), explicit)


@dataclass(frozen=True)
class _Curve:
    tokens: tuple[int, ...]
    costs: tuple[float, ...]
    queue_s: float


class ProfileStore:
    """Conservative p90 bucket profiles, fitted eagerly on observation.

    No zero-intercept fit: the first measured anchor retains launch/output cost.
    Prefix maxima enforce monotonicity without lowering a measured p90. The final
    segment extrapolates with at least the last anchor's average cost per token.
    Service and queue observations stay separate so elapsed waiting is not charged
    a second time at dispatch. Callers may override future queue estimates.
    """

    def __init__(self, bucket_size: int = 256, max_samples: int = 64,
                 min_samples: int = 1, cost_quantile: float = 0.9) -> None:
        if bucket_size < 1 or max_samples < min_samples or min_samples < 1:
            raise ValueError("invalid profile bucket/sample limits")
        if not 0.5 <= cost_quantile <= 1:
            raise ValueError("cost quantile must be conservative")
        self.bucket_size = bucket_size
        self.max_samples = max_samples
        self.min_samples = min_samples
        self.cost_quantile = cost_quantile
        self._samples: dict[tuple[ProfileKey, str], dict[int, deque[float]]] = defaultdict(dict)
        self._queues: dict[tuple[ProfileKey, str], deque[float]] = {}
        self._curves: dict[tuple[ProfileKey, str], _Curve] = {}

    def _bucket(self, tokens: int) -> int:
        if type(tokens) is not int or tokens < 0:
            raise ValueError("tokens must be a nonnegative integer")
        return math.ceil(tokens / self.bucket_size) * self.bucket_size

    def observe_cost(self, stage: str, tokens: int, service_s: float,
                     key: ProfileKey, queue_s: float = 0.0) -> None:
        bucket = self._bucket(tokens)
        if bucket == 0:
            raise ValueError("profile observations must contain positive tokens")
        service = _nonnegative(service_s, "service_s")
        queue = _nonnegative(queue_s, "queue_s")
        profile = (key, _stage(stage))
        self._samples[profile].setdefault(bucket, deque(maxlen=self.max_samples)).append(service)
        self._queues.setdefault(profile, deque(maxlen=self.max_samples)).append(queue)
        points = [(t, _quantile(values, self.cost_quantile))
                  for t, values in sorted(self._samples[profile].items())
                  if len(values) >= self.min_samples]
        costs: list[float] = []
        running = 0.0
        for _, cost in points:
            running = max(running, cost)
            costs.append(running)
        self._curves[profile] = _Curve(tuple(t for t, _ in points), tuple(costs),
                                       _quantile(self._queues[profile], self.cost_quantile))

    def predict_cost(self, stage: str, tokens: int, key: ProfileKey) -> StageCost | None:
        x = self._bucket(tokens)
        if x == 0:
            return StageCost(0.0)
        curve = self._curves.get((key, _stage(stage)))
        if curve is None or not curve.tokens:
            return None
        xs, ys = curve.tokens, curve.costs
        i = bisect.bisect_left(xs, x)
        if i == 0:
            cost = ys[0]
        elif i < len(xs):
            slope = max(0.0, (ys[i] - ys[i - 1]) / (xs[i] - xs[i - 1]))
            cost = ys[i - 1] + slope * (x - xs[i - 1])
        else:
            slope = ys[-1] / xs[-1]
            if len(xs) > 1:
                slope = max(slope, (ys[-1] - ys[-2]) / (xs[-1] - xs[-2]))
            cost = ys[-1] + slope * (x - xs[-1])
        return StageCost(cost, curve.queue_s)

    # Convenient aliases for clients that store stage observations directly.
    observe = observe_cost
    estimate = predict_cost


@dataclass(frozen=True)
class ResourceCaps:
    max_memory_tokens: int | None = None
    max_kv_tokens: int | None = None
    foreground_available: bool = True

    def allows(self, scope: Scope, kv_tokens: int = 0) -> bool:
        return (self.foreground_available
                and (self.max_memory_tokens is None or scope.input_tokens <= self.max_memory_tokens)
                and (self.max_kv_tokens is None or kv_tokens <= self.max_kv_tokens))


def _lengths(supported: Iterable[int], limit: int) -> list[int]:
    return sorted({k for k in supported if type(k) is int and 0 < k <= limit})


class LevelSelector:
    def __init__(self, profiles: ProfileStore, key: ProfileKey, *,
                 no_selector: bool = False, no_kv: bool = False) -> None:
        self.profiles = profiles
        self.key = key
        self.no_selector = no_selector
        self.no_kv = no_kv

    def select(self, scopes: Sequence[Scope], prediction: WindowPrediction,
               now: float, safety_margin: float = 0.0, *,
               kv_lengths: Iterable[int] = (), queue_memory_s: float | None = None,
               queue_kv_s: float | None = None, caps: ResourceCaps | None = None,
               baseline_ttft_s: float = 1.0) -> Plan | None:
        if not math.isfinite(now):
            raise ValueError("selection time must be finite")
        if now < prediction.dispatch_time or not prediction.usable or prediction.signature.model != self.key.model:
            return None
        if prediction.signature.workload_key != self.key.workload_key:
            return None
        remaining = prediction.remaining(now, safety_margin)
        initial = prediction.remaining(prediction.dispatch_time, safety_margin)
        if remaining <= 0 or initial <= 0:
            return None
        baseline = _nonnegative(baseline_ttft_s, "baseline TTFT")
        if baseline == 0:
            raise ValueError("baseline TTFT must be positive")
        caps = caps or ResourceCaps()
        legal = [s for s in scopes if s.input_tokens > 0 and s.stop > s.start]
        if self.no_selector and legal:
            legal = [max(legal, key=lambda s: (s.stop - s.start, s.input_tokens))]
        candidates: list[tuple[Plan, float]] = []
        kv_lengths = tuple(kv_lengths)
        for scope in legal:
            if not caps.allows(scope):
                continue
            memory = self.profiles.predict_cost("memory", scope.input_tokens, self.key)
            if memory is None:
                continue
            qm = memory.queue_s if queue_memory_s is None else _nonnegative(queue_memory_s, "memory queue")
            cost_l1 = memory.service_s + qm
            if scope.gain_l1_s > 0 and cost_l1 <= remaining:
                candidates.append((Plan(1, scope, 0, memory.service_s, 0.0,
                                        scope.gain_l1_s, prediction.deadline, initial, baseline), cost_l1))
            if self.no_kv or scope.new_prefix_tokens <= 0:
                continue
            marginal = scope.gain_l2_s - scope.gain_l1_s
            if marginal <= 0:
                continue
            for k in _lengths(kv_lengths, scope.new_prefix_tokens):
                if not caps.allows(scope, k):
                    continue
                kv = self.profiles.predict_cost("kv", k, self.key)
                if kv is None:
                    continue
                qk = kv.queue_s if queue_kv_s is None else _nonnegative(queue_kv_s, "KV queue")
                gain = scope.gain_l1_s + marginal * k / scope.new_prefix_tokens
                cost = cost_l1 + kv.service_s + qk
                if gain > 0 and cost <= remaining:
                    candidates.append((Plan(2, scope, k, memory.service_s, kv.service_s,
                                            gain, prediction.deadline, initial, baseline), cost))
        if not candidates:
            return None
        # Gains may be nonmonotonic in scope; never stop at the first oversized plan.
        return max(candidates, key=lambda candidate: (candidate[0].gain_s,
                                                       -candidate[1], -candidate[0].level))[0]

    def rebudget_kv(self, plan: Plan, actual_prefix_tokens: int, now: float,
                    safety_margin: float = 0.0, *, kv_lengths: Iterable[int] = (),
                    queue_kv_s: float | None = None,
                    caps: ResourceCaps | None = None) -> Plan | None:
        """Return a KV-only proposal; None leaves validated L1 text available.

        The deadline and initial budget survive memory completion. Neither memory
        service nor elapsed memory/queue time is included in the fresh KV charge.
        """
        if self.no_kv or plan.level != 2 or actual_prefix_tokens <= 0:
            return None
        remaining = max(0.0, plan.deadline - now - _nonnegative(safety_margin, "safety margin"))
        marginal = plan.scope.gain_l2_s - plan.scope.gain_l1_s
        if remaining <= 0 or marginal <= 0:
            return None
        caps = caps or ResourceCaps()
        # Native output length may differ from the dispatch-time estimate.
        scope = replace(plan.scope, new_prefix_tokens=actual_prefix_tokens)
        candidates: list[tuple[Plan, float]] = []
        for k in _lengths(kv_lengths, actual_prefix_tokens):
            if not caps.foreground_available or (caps.max_kv_tokens is not None and k > caps.max_kv_tokens):
                continue
            kv = self.profiles.predict_cost("kv", k, self.key)
            if kv is None:
                continue
            queue = kv.queue_s if queue_kv_s is None else _nonnegative(queue_kv_s, "KV queue")
            cost = kv.service_s + queue
            gain = marginal * k / actual_prefix_tokens
            if cost <= remaining and gain > 0:
                candidates.append((Plan(2, scope, k, 0.0, kv.service_s, gain,
                                        plan.deadline, plan.initial_budget,
                                        max(1e-9, plan.baseline_ttft_s - plan.scope.gain_l1_s)), cost))
        return max(candidates, key=lambda a: (a[0].gain_s, -a[1]))[0] if candidates else None


Selector = LevelSelector


@dataclass(frozen=True)
class ReadyTask:
    task_id: str
    stage: str
    plan: Plan
    admitted_at: float
    downstream_queue_s: float = 0.0
    valid: bool = True
    consumed: bool = False

    def __post_init__(self) -> None:
        _stage(self.stage)
        _nonnegative(self.downstream_queue_s, "downstream queue")
        if not math.isfinite(self.admitted_at):
            raise ValueError("admission time must be finite")

    @property
    def remaining_service_s(self) -> float:
        if _stage(self.stage) == "kv":
            return self.plan.kv_s
        return self.plan.memory_s + (self.plan.kv_s if self.plan.level == 2 else 0.0)

    @property
    def remaining_path_s(self) -> float:
        downstream = self.downstream_queue_s if _stage(self.stage) == "memory" and self.plan.level == 2 else 0.0
        return self.remaining_service_s + downstream


class TwoQueueScheduler:
    """Finite snapshot scheduling; foreground admission is a separate guard."""

    def __init__(self, *, fifo: bool = False, urgency_fraction: float = 0.2,
                 max_age_s: float = 2.0, epsilon: float = 1e-9) -> None:
        if not 0 <= urgency_fraction <= 1 or max_age_s < 0 or epsilon <= 0:
            raise ValueError("invalid scheduler parameters")
        self.fifo = fifo
        self.urgency_fraction = urgency_fraction
        self.max_age_s = max_age_s
        self.epsilon = epsilon

    def feasible(self, task: ReadyTask, now: float, safety_margin: float = 0.0) -> bool:
        _nonnegative(safety_margin, "safety margin")
        if not math.isfinite(now):
            raise ValueError("scheduling time must be finite")
        p = task.plan
        return (task.valid and not task.consumed and task.admitted_at <= now and p.gain_s > 0
                and p.initial_budget > 0 and p.level in {1, 2}
                and (_stage(task.stage) != "kv" or p.level == 2)
                and task.remaining_path_s <= max(0.0, p.deadline - now - safety_margin)
                and p.deadline - now - safety_margin > 0)

    def order_key(self, task: ReadyTask, now: float, safety_margin: float = 0.0) -> tuple:
        _nonnegative(safety_margin, "safety margin")
        if self.fifo:
            return (task.admitted_at, task.task_id)
        remaining = max(0.0, task.plan.deadline - now - safety_margin)
        allowance = remaining - task.remaining_path_s
        rho = max(0.0, min(1.0, allowance / task.plan.initial_budget))
        age = max(0.0, now - task.admitted_at)
        if rho <= self.urgency_fraction or age >= self.max_age_s:
            return (0, rho, -age, task.plan.deadline, task.task_id)
        relative_gain = min(1.0, max(0.0, task.plan.gain_s / max(self.epsilon, task.plan.baseline_ttft_s)))
        benefit = relative_gain / (task.remaining_service_s + self.epsilon)
        return (1, -benefit, task.remaining_service_s, task.admitted_at, task.task_id)

    def ordered(self, tasks: Iterable[ReadyTask], now: float,
                safety_margin: float = 0.0) -> list[ReadyTask]:
        _nonnegative(safety_margin, "safety margin")
        return sorted((t for t in tasks if self.feasible(t, now, safety_margin)),
                      key=lambda t: self.order_key(t, now, safety_margin))

    def choose(self, tasks: Iterable[ReadyTask], now: float,
               safety_margin: float = 0.0, *,
               can_dispatch: Callable[[ReadyTask], bool] | None = None) -> ReadyTask | None:
        for task in self.ordered(tasks, now, safety_margin):
            # A blocked request must not block another feasible ready request.
            if can_dispatch is None or can_dispatch(task):
                return task
        return None


Scheduler = TwoQueueScheduler
