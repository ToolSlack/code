"""Strict server-side history for ToolSlack's relative-TTFT benefit signal.

The measured TTFT is proxy admission to the first semantic token:
``proxy_arrival_to_first_token_seconds`` from the existing proxy event schema.
This includes proxy/control-plane work and upstream queueing.  The module never
infers a pre/post pair from timestamps alone; the controller must bind requests
to an explicit committed context epoch.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import statistics
from typing import Any, Literal


TTFT_FIELD = "proxy_arrival_to_first_token_seconds"
PRE_MEMORY = "PRE_MEMORY"
POST_MEMORY = "POST_MEMORY"
Phase = Literal["PRE_MEMORY", "POST_MEMORY"]


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def _receipt(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _required_text(name: str, value: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def retained_ratio_bucket(value: float) -> str:
    """Bucket a completed native operation using only commit-time information."""
    if not math.isfinite(value) or value < 0:
        raise ValueError("actual retained ratio must be finite and nonnegative")
    if value <= 0.25:
        return "0.00-0.25"
    if value <= 0.50:
        return "0.25-0.50"
    if value <= 0.75:
        return "0.50-0.75"
    if value <= 1.00:
        return "0.75-1.00"
    return ">1.00"


@dataclass(frozen=True)
class RequestBinding:
    request_id: str
    session_id: str
    run_id: str
    arm: str
    model_profile_hash: str
    memory_algorithm: str
    context_epoch: str
    snapshot_id: str
    queue_bucket: str
    cache_state: str
    input_length_bucket: str
    phase: Phase
    commit_id: str | None = None

    def validate(self) -> None:
        for name, value in asdict(self).items():
            if name == "commit_id":
                if self.phase == POST_MEMORY:
                    _required_text(name, value)  # type: ignore[arg-type]
                continue
            _required_text(name, value)
        if self.phase not in (PRE_MEMORY, POST_MEMORY):
            raise ValueError("phase must be PRE_MEMORY or POST_MEMORY")
        if self.phase == PRE_MEMORY and self.commit_id is not None:
            raise ValueError("a PRE_MEMORY request cannot claim a future commit")


@dataclass(frozen=True)
class CommitBinding:
    commit_id: str
    session_id: str
    run_id: str
    arm: str
    model_profile_hash: str
    memory_algorithm: str
    pre_request_id: str
    pre_context_epoch: str
    post_context_epoch: str
    post_snapshot_id: str
    queue_bucket: str
    cache_state: str
    pre_input_length_bucket: str
    post_input_length_bucket: str
    original_scope_tokens: int
    materialized_scope_tokens: int
    committed_monotonic_ns: int

    def validate(self) -> None:
        for name, value in asdict(self).items():
            if name in ("original_scope_tokens", "materialized_scope_tokens", "committed_monotonic_ns"):
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise ValueError(f"{name} must be a nonnegative integer")
                continue
            _required_text(name, value)
        if self.pre_context_epoch == self.post_context_epoch:
            raise ValueError("a commit must advance the context epoch")
        if self.original_scope_tokens <= 0:
            raise ValueError("original_scope_tokens must be positive")
        if self.committed_monotonic_ns <= 0:
            raise ValueError("committed_monotonic_ns must be positive")

    @property
    def actual_retained_ratio(self) -> float:
        return self.materialized_scope_tokens / self.original_scope_tokens

    @property
    def compression_ratio_bucket(self) -> str:
        return retained_ratio_bucket(self.actual_retained_ratio)


@dataclass(frozen=True)
class TTFTObservation:
    request_id: str
    session_id: str
    run_id: str
    arm: str
    model_profile_hash: str
    memory_algorithm: str
    context_epoch: str
    snapshot_id: str
    queue_bucket: str
    cache_state: str
    input_length_bucket: str
    phase: Phase
    ttft_seconds: float
    observed_monotonic_ns: int
    source_event_receipt_sha256: str


@dataclass(frozen=True)
class PairedGain:
    commit_id: str
    session_id: str
    run_id: str
    arm: str
    model_profile_hash: str
    memory_algorithm: str
    pre_request_id: str
    post_request_id: str
    pre_context_epoch: str
    post_context_epoch: str
    post_snapshot_id: str
    queue_bucket: str
    cache_state: str
    pre_input_length_bucket: str
    post_input_length_bucket: str
    original_scope_tokens: int
    materialized_scope_tokens: int
    committed_monotonic_ns: int
    actual_retained_ratio: float
    compression_ratio_bucket: str
    pre_ttft_seconds: float
    post_ttft_seconds: float
    post_observed_monotonic_ns: int
    relative_ttft_gain: float
    receipt_sha256: str


class TTFTGainHistory:
    """In-memory reference implementation for later proxy integration."""

    def __init__(self) -> None:
        self._bindings: dict[str, RequestBinding] = {}
        self._observations: dict[str, TTFTObservation] = {}
        self._commits: dict[str, CommitBinding] = {}
        self._consumed_commits: set[str] = set()
        self._pairs: list[PairedGain] = []
        self._audit: list[dict[str, Any]] = []

    @property
    def pairs(self) -> tuple[PairedGain, ...]:
        return tuple(self._pairs)

    @property
    def audit(self) -> tuple[dict[str, Any], ...]:
        return tuple(self._audit)

    def has_commit(self, commit_id: str) -> bool:
        """Return whether an explicit commit identity is registered."""
        return commit_id in self._commits

    def _result(self, status: str, reason: str, **extra: Any) -> dict[str, Any]:
        result = {"status": status, "reason": reason, **extra}
        self._audit.append(result)
        return result

    def register_request(self, binding: RequestBinding) -> None:
        binding.validate()
        if binding.request_id in self._bindings:
            raise ValueError("request_id is already bound")
        self._bindings[binding.request_id] = binding

    def register_commit(self, commit: CommitBinding) -> None:
        commit.validate()
        if commit.commit_id in self._commits:
            raise ValueError("commit_id is already registered")
        pre = self._observations.get(commit.pre_request_id)
        if pre is None:
            raise ValueError("the explicitly bound pre request has no TTFT observation")
        expected = {
            "session_id": commit.session_id,
            "run_id": commit.run_id,
            "arm": commit.arm,
            "model_profile_hash": commit.model_profile_hash,
            "memory_algorithm": commit.memory_algorithm,
            "context_epoch": commit.pre_context_epoch,
            "queue_bucket": commit.queue_bucket,
            "cache_state": commit.cache_state,
            "input_length_bucket": commit.pre_input_length_bucket,
            "phase": PRE_MEMORY,
        }
        mismatches = [name for name, value in expected.items() if getattr(pre, name) != value]
        if mismatches:
            raise ValueError("pre request does not match commit: " + ",".join(mismatches))
        if commit.committed_monotonic_ns <= pre.observed_monotonic_ns:
            raise ValueError("commit must occur after the pre-memory TTFT observation")
        self._commits[commit.commit_id] = commit

    def observe_proxy_event(self, event: dict[str, Any]) -> dict[str, Any]:
        if event.get("event") != "first_token":
            return self._result("ignored", "not_first_token")
        if event.get("request_kind") != "foreground":
            return self._result("ignored", "not_foreground")
        request_id = event.get("request_id")
        binding = self._bindings.get(request_id)
        if binding is None:
            return self._result("rejected", "unbound_request", request_id=request_id)
        if event.get("session_id") != binding.session_id:
            return self._result("rejected", "event_session_mismatch", request_id=request_id)
        if event.get("arm") != binding.arm:
            return self._result("rejected", "event_arm_mismatch", request_id=request_id)
        if event.get("cohort_id") != f"{binding.run_id}:{binding.arm}":
            return self._result("rejected", "event_run_mismatch", request_id=request_id)
        value = event.get(TTFT_FIELD)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            return self._result("rejected", "invalid_agent_visible_ttft", request_id=request_id)
        timestamp = event.get("received_monotonic_ns", event.get("monotonic_ns"))
        if isinstance(timestamp, bool) or not isinstance(timestamp, int) or timestamp <= 0:
            return self._result("rejected", "invalid_monotonic_timestamp", request_id=request_id)
        if request_id in self._observations:
            return self._result("rejected", "request_already_observed", request_id=request_id)
        observation = TTFTObservation(
            **{name: getattr(binding, name) for name in (
                "request_id", "session_id", "run_id", "arm", "model_profile_hash", "memory_algorithm",
                "context_epoch", "snapshot_id", "queue_bucket", "cache_state",
                "input_length_bucket", "phase")},
            ttft_seconds=float(value),
            observed_monotonic_ns=timestamp,
            source_event_receipt_sha256=_receipt(event),
        )
        if binding.phase == PRE_MEMORY:
            self._observations[request_id] = observation
            return self._result("accepted", "pre_observation_recorded", request_id=request_id)

        commit_id = binding.commit_id
        if commit_id in self._consumed_commits:
            return self._result("rejected", "commit_already_consumed", request_id=request_id, commit_id=commit_id)
        commit = self._commits.get(commit_id or "")
        if commit is None:
            return self._result("rejected", "unknown_commit", request_id=request_id, commit_id=commit_id)
        expected = {
            "session_id": commit.session_id,
            "run_id": commit.run_id,
            "arm": commit.arm,
            "model_profile_hash": commit.model_profile_hash,
            "memory_algorithm": commit.memory_algorithm,
            "context_epoch": commit.post_context_epoch,
            "snapshot_id": commit.post_snapshot_id,
            "queue_bucket": commit.queue_bucket,
            "cache_state": commit.cache_state,
            "input_length_bucket": commit.post_input_length_bucket,
        }
        mismatches = [name for name, expected_value in expected.items()
                      if getattr(observation, name) != expected_value]
        if mismatches:
            return self._result(
                "rejected", "post_binding_mismatch", request_id=request_id,
                commit_id=commit_id, fields=mismatches,
            )
        pre = self._observations[commit.pre_request_id]
        if observation.observed_monotonic_ns <= pre.observed_monotonic_ns:
            return self._result("rejected", "post_not_after_pre", request_id=request_id, commit_id=commit_id)
        if observation.observed_monotonic_ns <= commit.committed_monotonic_ns:
            return self._result("rejected", "post_not_after_commit", request_id=request_id, commit_id=commit_id)

        # Preserve the signed ratio exactly; do not clamp a severe regression to -1.
        gain = (pre.ttft_seconds - observation.ttft_seconds) / pre.ttft_seconds
        pair_payload = {
            "commit": asdict(commit),
            "pre": asdict(pre),
            "post": asdict(observation),
            "relative_ttft_gain": gain,
            "metric": TTFT_FIELD,
        }
        pair = PairedGain(
            commit_id=commit.commit_id,
            session_id=commit.session_id,
            run_id=commit.run_id,
            arm=commit.arm,
            model_profile_hash=commit.model_profile_hash,
            memory_algorithm=commit.memory_algorithm,
            pre_request_id=pre.request_id,
            post_request_id=observation.request_id,
            pre_context_epoch=commit.pre_context_epoch,
            post_context_epoch=commit.post_context_epoch,
            post_snapshot_id=commit.post_snapshot_id,
            queue_bucket=commit.queue_bucket,
            cache_state=commit.cache_state,
            pre_input_length_bucket=commit.pre_input_length_bucket,
            post_input_length_bucket=commit.post_input_length_bucket,
            original_scope_tokens=commit.original_scope_tokens,
            materialized_scope_tokens=commit.materialized_scope_tokens,
            committed_monotonic_ns=commit.committed_monotonic_ns,
            actual_retained_ratio=commit.actual_retained_ratio,
            compression_ratio_bucket=commit.compression_ratio_bucket,
            pre_ttft_seconds=pre.ttft_seconds,
            post_ttft_seconds=observation.ttft_seconds,
            post_observed_monotonic_ns=observation.observed_monotonic_ns,
            relative_ttft_gain=gain,
            receipt_sha256=_receipt(pair_payload),
        )
        self._observations[request_id] = observation
        self._pairs.append(pair)
        self._consumed_commits.add(commit.commit_id)
        return self._result(
            "accepted", "paired_gain_recorded", request_id=request_id,
            commit_id=commit.commit_id, relative_ttft_gain=gain,
            receipt_sha256=pair.receipt_sha256,
        )

    def estimate_relative_ttft_gain(
        self,
        *,
        session_id: str,
        run_id: str,
        arm: str,
        model_profile_hash: str,
        memory_algorithm: str,
        queue_bucket: str,
        cache_state: str,
        pre_input_length_bucket: str,
        as_of_monotonic_ns: int,
        known_compression_ratio_bucket: str | None = None,
        min_cohort_samples: int = 3,
    ) -> dict[str, Any]:
        if min_cohort_samples <= 0:
            raise ValueError("min_cohort_samples must be positive")
        dimensions = {
            "run_id": run_id,
            "arm": arm,
            "model_profile_hash": model_profile_hash,
            "memory_algorithm": memory_algorithm,
            "queue_bucket": queue_bucket,
            "cache_state": cache_state,
            "pre_input_length_bucket": pre_input_length_bucket,
        }
        if isinstance(as_of_monotonic_ns, bool) or not isinstance(as_of_monotonic_ns, int) or as_of_monotonic_ns <= 0:
            raise ValueError("as_of_monotonic_ns must be a positive integer")
        eligible = [
            pair for pair in self._pairs
            if pair.post_observed_monotonic_ns <= as_of_monotonic_ns
            and all(getattr(pair, name) == value for name, value in dimensions.items())
        ]
        same_session = [pair for pair in eligible if pair.session_id == session_id]
        selected: list[PairedGain]
        confidence: str
        if same_session:
            selected = same_session
            confidence = "direct_session_history"
        elif known_compression_ratio_bucket is not None:
            cohort = [pair for pair in eligible
                      if pair.compression_ratio_bucket == known_compression_ratio_bucket]
            if len(cohort) < min_cohort_samples:
                return {
                    "value": None, "confidence": "unknown",
                    "sample_count": len(cohort), "receipt_sha256": None,
                    "metric": TTFT_FIELD,
                }
            selected = cohort
            confidence = "matched_cohort_history"
        else:
            return {
                "value": None,
                "confidence": "unknown",
                "sample_count": len(eligible),
                "receipt_sha256": None,
                "metric": TTFT_FIELD,
            }
        values = [pair.relative_ttft_gain for pair in selected]
        receipts = sorted(pair.receipt_sha256 for pair in selected)
        return {
            "value": statistics.median(values),
            "confidence": confidence,
            "sample_count": len(selected),
            "receipt_sha256": _receipt(receipts),
            "metric": TTFT_FIELD,
        }
