"""Explicit HTTP lifecycle for causal relative-TTFT history.

The proxy, rather than a caller timestamp heuristic, owns the first-token
observation and the commit clock.  A controller names the exact pre request,
commit, and first post request.  Admission queries only pairs complete at the
native request's proxy-arrival timestamp.
"""
from __future__ import annotations

from dataclasses import asdict
import math
import time
from typing import Any, Callable

from ttft_gain_history import (
    CommitBinding,
    POST_MEMORY,
    PRE_MEMORY,
    RequestBinding,
    TTFTGainHistory,
)


class TTFTGainHTTP:
    def __init__(
        self,
        *,
        emit: Callable[..., Any] = lambda *args, **kwargs: None,
        request_started: Callable[[str], bool] = lambda request_id: False,
        clock_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        self.history = TTFTGainHistory()
        self.emit = emit
        self.request_started = request_started
        self.clock_ns = clock_ns
        self.post_claims: dict[str, str] = {}

    @staticmethod
    def _request_binding(body: dict[str, Any], phase: str) -> RequestBinding:
        allowed = {
            "request_id", "session_id", "run_id", "arm", "model_profile_hash",
            "memory_algorithm", "context_epoch", "snapshot_id", "queue_bucket",
            "cache_state", "input_length_bucket", "commit_id",
        }
        extra = set(body) - allowed
        if extra:
            raise ValueError("unknown request-binding fields: " + ",".join(sorted(extra)))
        values = {name: body.get(name) for name in allowed}
        if phase == PRE_MEMORY:
            if values.get("commit_id") is not None:
                raise ValueError("pre request cannot reference a commit")
            values["commit_id"] = None
        return RequestBinding(phase=phase, **values)

    def register_pre(self, body: dict[str, Any]) -> dict[str, Any]:
        binding = self._request_binding(body, PRE_MEMORY)
        if self.request_started(binding.request_id):
            raise ValueError("pre binding must precede proxy request arrival")
        self.history.register_request(binding)
        result = {"status": "registered", "phase": PRE_MEMORY, "request_id": binding.request_id}
        self.emit("ttft_history_pre_registered", **result)
        return result

    def register_commit(self, body: dict[str, Any]) -> dict[str, Any]:
        allowed = {
            "commit_id", "session_id", "run_id", "arm", "model_profile_hash",
            "memory_algorithm", "pre_request_id", "pre_context_epoch",
            "post_context_epoch", "post_snapshot_id", "queue_bucket", "cache_state",
            "pre_input_length_bucket", "post_input_length_bucket",
            "original_scope_tokens", "materialized_scope_tokens",
        }
        extra = set(body) - allowed
        if extra:
            raise ValueError("unknown commit fields: " + ",".join(sorted(extra)))
        # The server owns the monotonic commit timestamp.  A client cannot move
        # a commit backward or forward to manufacture a causal pair.
        commit = CommitBinding(committed_monotonic_ns=self.clock_ns(), **{
            name: body.get(name) for name in allowed
        })
        self.history.register_commit(commit)
        result = {
            "status": "registered",
            "commit_id": commit.commit_id,
            "pre_request_id": commit.pre_request_id,
            "committed_monotonic_ns": commit.committed_monotonic_ns,
            "actual_retained_ratio": commit.actual_retained_ratio,
            "compression_ratio_bucket": commit.compression_ratio_bucket,
        }
        self.emit("ttft_history_commit_registered", **result)
        return result

    def register_post(self, body: dict[str, Any]) -> dict[str, Any]:
        binding = self._request_binding(body, POST_MEMORY)
        if self.request_started(binding.request_id):
            raise ValueError("post binding must precede proxy request arrival")
        if not self.history.has_commit(binding.commit_id or ""):
            raise ValueError("post request references an unknown commit")
        prior = self.post_claims.get(binding.commit_id or "")
        if prior is not None:
            raise ValueError("commit already has an explicitly bound first post request")
        self.history.register_request(binding)
        self.post_claims[binding.commit_id or ""] = binding.request_id
        result = {
            "status": "registered",
            "phase": POST_MEMORY,
            "request_id": binding.request_id,
            "commit_id": binding.commit_id,
        }
        self.emit("ttft_history_post_registered", **result)
        return result

    def observe_first_token(
        self,
        *,
        common: dict[str, Any],
        received_monotonic_ns: int,
        proxy_arrival_to_first_token_seconds: float,
    ) -> dict[str, Any]:
        result = self.history.observe_proxy_event({
            "event": "first_token",
            "request_id": common.get("request_id"),
            "session_id": common.get("session_id"),
            "request_kind": common.get("request_kind"),
            "received_monotonic_ns": received_monotonic_ns,
            "arm": common.get("arm"),
            "cohort_id": common.get("cohort_id"),
            "proxy_arrival_to_first_token_seconds": proxy_arrival_to_first_token_seconds,
        })
        self.emit("ttft_history_observation", **result)
        return result

    def estimate_for_maintenance(
        self,
        *,
        raw_query: Any,
        common: dict[str, Any],
        as_of_monotonic_ns: int,
        actual_model_profile_hash: str,
    ) -> dict[str, Any]:
        if not isinstance(raw_query, dict):
            raise ValueError("ttft_gain_query object is required")
        allowed = {
            "run_id", "model_profile_hash", "memory_algorithm", "queue_bucket",
            "cache_state", "pre_input_length_bucket", "known_compression_ratio_bucket",
            "min_cohort_samples",
        }
        extra = set(raw_query) - allowed
        if extra:
            raise ValueError("unknown TTFT-query fields: " + ",".join(sorted(extra)))
        session_id = common.get("session_id")
        arm = common.get("arm")
        cohort_id = common.get("cohort_id")
        run_id = raw_query.get("run_id")
        required_query_fields = (
            "run_id", "model_profile_hash", "memory_algorithm", "queue_bucket",
            "cache_state", "pre_input_length_bucket",
        )
        if not all(
            isinstance(raw_query.get(name), str) and raw_query.get(name)
            for name in required_query_fields
        ):
            raise ValueError("all TTFT-history query identities and buckets are required")
        if not all(isinstance(value, str) and value for value in (session_id, arm)):
            raise ValueError("session and arm identities are required")
        if cohort_id != f"{run_id}:{arm}":
            raise ValueError("foreground cohort does not match the explicit TTFT-history run/arm")
        if raw_query.get("model_profile_hash") != actual_model_profile_hash:
            raise ValueError("TTFT-history model profile differs from the active proxy profile")
        estimate = self.history.estimate_relative_ttft_gain(
            session_id=session_id,
            run_id=run_id,
            arm=arm,
            model_profile_hash=actual_model_profile_hash,
            memory_algorithm=raw_query.get("memory_algorithm"),
            queue_bucket=raw_query.get("queue_bucket"),
            cache_state=raw_query.get("cache_state"),
            pre_input_length_bucket=raw_query.get("pre_input_length_bucket"),
            as_of_monotonic_ns=as_of_monotonic_ns,
            known_compression_ratio_bucket=raw_query.get("known_compression_ratio_bucket"),
            min_cohort_samples=raw_query.get("min_cohort_samples", 3),
        )
        value = estimate["value"]
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
        ):
            raise ValueError("history produced a non-finite gain")
        # Unknown is neutral for queue ordering, but remains JSON null in the
        # evidence.  A negative measured benefit is passed through unchanged.
        result = {
            "scheduler_value": 0.0 if value is None else float(value),
            "history_value": value,
            "confidence": estimate["confidence"],
            "sample_count": estimate["sample_count"],
            "receipt_sha256": estimate["receipt_sha256"],
            "metric": estimate["metric"],
            "as_of_monotonic_ns": as_of_monotonic_ns,
            "unknown_is_neutral": value is None,
        }
        self.emit("ttft_history_estimate", session_id=session_id, run_id=run_id, arm=arm, **result)
        return result

    def install(self, app: Any) -> None:
        from aiohttp import web

        async def body(request):
            value = await request.json()
            if not isinstance(value, dict):
                raise ValueError("JSON object required")
            return value

        def error(exc: Exception):
            status = 409 if "already" in str(exc) or "unknown commit" in str(exc) else 400
            return web.json_response({"error": type(exc).__name__, "message": str(exc)}, status=status)

        async def pre(request):
            try:
                return web.json_response(self.register_pre(await body(request)), status=201)
            except (TypeError, ValueError) as exc:
                return error(exc)

        async def commit(request):
            try:
                return web.json_response(self.register_commit(await body(request)), status=201)
            except (TypeError, ValueError) as exc:
                return error(exc)

        async def post(request):
            try:
                return web.json_response(self.register_post(await body(request)), status=201)
            except (TypeError, ValueError) as exc:
                return error(exc)

        async def estimate(request):
            try:
                payload = await body(request)
                common = payload.pop("common")
                actual = payload.pop("actual_model_profile_hash")
                if payload:
                    raise ValueError("unknown estimate-envelope fields: " + ",".join(sorted(payload)))
                result = self.estimate_for_maintenance(
                    raw_query=common.pop("ttft_gain_query"),
                    common=common,
                    as_of_monotonic_ns=self.clock_ns(),
                    actual_model_profile_hash=actual,
                )
                return web.json_response(result)
            except (KeyError, TypeError, ValueError) as exc:
                return error(exc)

        app.router.add_post("/toolslack/v1/ttft/pre-requests", pre)
        app.router.add_post("/toolslack/v1/ttft/commits", commit)
        app.router.add_post("/toolslack/v1/ttft/post-requests", post)
        app.router.add_post("/toolslack/v1/ttft/estimate", estimate)
