"""Budget exact KV preparation without changing the future model request.

The profile is a same-service calibration, not a statistical deadline guarantee.
Queueing and serialization consume the absolute tool deadline. Unmeasured
prefix sizes and unavailable profiles decline optional work before submission.
"""
import hashlib
import json
import math
import time
from pathlib import Path


class BudgetDeclined(ValueError):
    pass


def finite_number(value, name, positive=False):
    if (type(value) not in (int, float) or not math.isfinite(value)
            or (value <= 0 if positive else value < 0)):
        raise ValueError('invalid ' + name)
    return float(value)


class PrefixCost:
    def __init__(self, raw, service_sha, source_sha=None):
        if raw.get('schema') != 'toolslack.native-prefix-prefill-cost.v1':
            raise ValueError('unknown prefix calibration schema')
        if not service_sha or raw.get('service_profile_sha256') != service_sha:
            raise ValueError('prefix calibration service identity differs')
        if raw.get('extrapolation_allowed') is not False:
            raise ValueError('prefix cost extrapolation must be disabled')
        previous_tokens, previous_ms = 0, 0.
        self.knots = []
        for row in raw.get('knots', []):
            n = row.get('tokens')
            ms = finite_number(row.get('monotone_upper_ms'), 'prefix cost', True)
            samples = row.get('samples_ms')
            if (type(n) is not int or n <= previous_tokens or ms < previous_ms
                    or not isinstance(samples, list) or not samples
                    or row.get('sample_count') != len(samples)):
                raise ValueError('invalid monotone calibration knots')
            if any(finite_number(v, 'calibration sample', True) > ms for v in samples):
                raise ValueError('cost does not cover observed samples')
            self.knots.append((n, ms))
            previous_tokens, previous_ms = n, ms
        if not self.knots:
            raise ValueError('empty prefix calibration')
        self.source_sha = source_sha
        self.service_sha = service_sha

    @classmethod
    def load(cls, path, service_sha):
        raw = Path(path).read_bytes()
        return cls(json.loads(raw), service_sha, hashlib.sha256(raw).hexdigest())


class PrefixBudget:
    def __init__(self, spec, profile, now_wall=None, now_mono=None):
        if profile is None:
            raise BudgetDeclined('same-service prefix cost profile is unavailable')
        self.profile = profile
        self.deadline_unix_ms = finite_number(spec['deadline_unix_ms'], 'deadline', True)
        self.safety_ms = finite_number(spec.get('safety_ms', 0), 'safety')
        wall = time.time() if now_wall is None else now_wall
        mono = time.monotonic() if now_mono is None else now_mono
        # Bind once at HTTP admission; subsequent wall-clock changes cannot
        # lengthen this request's tool window.
        self.deadline_mono = mono + (self.deadline_unix_ms / 1000 - wall)

    def select(self, full_stable_tokens=None, now_mono=None):
        now = time.monotonic() if now_mono is None else now_mono
        remaining_ms = (self.deadline_mono - now) * 1000
        candidates = [(n, ms) for n, ms in self.profile.knots
                      if (full_stable_tokens is None or n <= full_stable_tokens)
                      and ms + self.safety_ms <= remaining_ms]
        if not candidates:
            raise BudgetDeclined('remaining tool budget cannot cover a calibrated prefix')
        n, ms = candidates[-1]
        return dict(selected_prefix_tokens=n, estimated_device_ready_ms=ms,
                    remaining_at_selection_ms=remaining_ms, safety_ms=self.safety_ms,
                    deadline_unix_ms=self.deadline_unix_ms,
                    cost_profile_sha256=self.profile.source_sha,
                    full_stable_prefix_tokens=full_stable_tokens,
                    whole_future_request_preserved=True,
                    estimate_kind='same_service_empirical_calibration_not_deadline_guarantee')
