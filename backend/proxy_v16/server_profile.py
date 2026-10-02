"""Validated Qwen3-8B service profile; import and validation do not touch GPUs."""
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class ServerProfile:
    model_path: str = '/models/Qwen3-8B'
    model_name: str = 'Qwen3-8B'
    server_python: str = '/opt/venv/bin/python'
    server_cwd: str = '/opt/sglang'
    context_length: int = 131072
    # The Qwen3-8B Agent workload uses the 8K completion allowance that was
    # frozen for both arms after the earlier 2K cap truncated tool-following
    # answers.  The 90% trigger is derived from the resulting usable input.
    output_reserve: int = 8192
    enable_thinking: bool = False
    trigger_fraction: float = .90
    mem_fraction_static: float = .75
    max_running_requests: int = 16
    max_queued_requests: int = 256
    cuda_graph_max_bs: int = 16
    chunked_prefill_size: int = 8192
    max_prefill_tokens: int = 16384
    request_timeout_seconds: int = 1800
    random_seed: int = 20260916

    def __post_init__(self):
        # This runtime is only verified for this installed model/engine pairing.
        if self.model_name != 'Qwen3-8B' or self.context_length != 131072:
            raise ValueError('This profile is verified only for Qwen3-8B + 131072-token YaRN')
        if not all(Path(x).is_absolute() for x in (self.model_path, self.server_python, self.server_cwd)):
            raise ValueError('Model and runtime paths must be absolute')
        for name in ('context_length', 'output_reserve', 'max_running_requests',
                     'max_queued_requests', 'cuda_graph_max_bs', 'chunked_prefill_size',
                     'max_prefill_tokens', 'request_timeout_seconds', 'random_seed'):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f'{name} must be a positive integer')
        if type(self.enable_thinking) is not bool:
            raise ValueError('enable_thinking must be boolean')
        if not 1 <= self.output_reserve <= 32768:
            raise ValueError('Output reserve outside validated range')
        if self.trigger_fraction != .90:
            raise ValueError('User-specified trigger remains 90% of model usable input')
        if not .5 <= self.mem_fraction_static <= .80:
            raise ValueError('Static allocation must retain tested activation headroom')
        if self.max_running_requests > 16 or self.cuda_graph_max_bs != self.max_running_requests:
            raise ValueError('At most 16 active requests, with matching CUDA graph capacity')
        if self.max_queued_requests < 16:
            raise ValueError('Queue must admit the requested Agent concurrency')
        if not 512 <= self.chunked_prefill_size <= 8192:
            raise ValueError('Chunked prefill outside conservative supported range')
        if self.max_prefill_tokens < self.chunked_prefill_size:
            raise ValueError('max_prefill_tokens must cover a prefill chunk')

    @property
    def max_input_tokens(self):
        # The pinned SGLang backend uses a strict shared input/output boundary.
        return self.context_length - self.output_reserve - 1

    @property
    def trigger_tokens(self):
        return math.ceil(self.trigger_fraction * self.max_input_tokens)

    @property
    def rope_overrides(self):
        return {'max_position_embeddings': self.context_length,
                'rope_scaling': {'rope_type': 'yarn', 'factor': 4.0,
                                 'original_max_position_embeddings': 32768}}

    def capacity(self):
        return dict(context_window=self.context_length, output_reserve=self.output_reserve,
                    max_input_tokens=self.max_input_tokens,
                    nominal_model_input_budget=self.context_length-self.output_reserve,
                    backend_boundary_reserve=1, trigger_fraction=self.trigger_fraction,
                    trigger_tokens=self.trigger_tokens, enable_thinking=self.enable_thinking,
                    capacity_mode='official_yarn_4x', output_and_input_share_context=True)

    def server_argv(self, port):
        if type(port) is not int or not 1024 <= port <= 65535:
            raise ValueError('Unprivileged TCP port required')
        return [self.server_python, '-m', 'sglang.launch_server',
                '--model-path', self.model_path, '--served-model-name', self.model_name,
                '--host', '127.0.0.1', '--port', str(port), '--dtype', 'bfloat16',
                '--context-length', str(self.context_length),
                '--json-model-override-args', json.dumps(self.rope_overrides, separators=(',', ':')),
                '--mem-fraction-static', str(self.mem_fraction_static),
                '--max-running-requests', str(self.max_running_requests),
                '--max-queued-requests', str(self.max_queued_requests),
                '--chunked-prefill-size', str(self.chunked_prefill_size),
                '--max-prefill-tokens', str(self.max_prefill_tokens),
                '--attention-backend', 'triton', '--reasoning-parser', 'qwen3',
                '--tool-call-parser', 'qwen', '--enable-metrics', '--enable-cache-report',
                '--cuda-graph-max-bs', str(self.cuda_graph_max_bs),
                '--random-seed', str(self.random_seed)]


def load_profile(path=None):
    return ServerProfile(**json.loads(Path(path).read_text())) if path else ServerProfile()


if __name__ == '__main__':
    print(json.dumps(asdict(ServerProfile()), indent=2))
