"""CPU-only, explicit model/request contract for the Qwen 27B pilot."""
from __future__ import annotations
from copy import deepcopy
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from pathlib import Path


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


@dataclass(frozen=True)
class ModelProfile:
    model_path: str = '/models/Qwen3.8-27B'
    model_name: str = 'Qwen3.8-27B'
    architecture: str = 'Qwen3_5ForConditionalGeneration'
    context_length: int = 262144
    output_reserve: int = 2048
    backend_boundary_reserve: int = 1
    trigger_fraction: float = 0.90
    enable_thinking: bool = False
    # Sampling values are from this checkpoint's generation_config.json.
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 20
    is_multimodal_architecture: bool = True
    tool_call_parser: str | None = None
    reasoning_parser: str | None = None
    request_timeout_seconds: int = 1800
    sglang_version: str = '0.5.10'
    transformers_version: str = '5.3.0'
    model_config_sha256: str = '191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab'
    tokenizer_json_sha256: str = '0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3'
    tokenizer_config_sha256: str = 'b11349aafa7cdc6a320767cf7ceb29ed82f7eda5d65e8e0819e76f0ce947bf27'
    chat_template_sha256: str = 'c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041'

    def __post_init__(self):
        if not Path(self.model_path).is_absolute() or not self.model_name:
            raise ValueError('Absolute local model path and served model name required')
        if self.architecture != 'Qwen3_5ForConditionalGeneration' or self.context_length != 262144:
            raise ValueError('This adapter is audited only for the actual Qwen3_5 dense checkpoint')
        for name in ('context_length', 'output_reserve', 'backend_boundary_reserve', 'request_timeout_seconds'):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(name + ' must be a positive integer')
        if self.output_reserve >= self.context_length - self.backend_boundary_reserve:
            raise ValueError('Output reserve leaves no input capacity')
        if self.trigger_fraction != 0.90 or self.enable_thinking is not False:
            raise ValueError('This independent pilot freezes 90% usable input and non-thinking deployment mode')
        if (self.temperature, self.top_p, self.top_k) != (1.0, .95, 20):
            raise ValueError('Foreground checkpoint generation defaults changed')
        if self.is_multimodal_architecture is not True:
            raise ValueError('Pure-text requests still use the native conditional model preprocessing route')
        if self.sglang_version != '0.5.10' or self.transformers_version != '5.3.0':
            raise ValueError('Exact counter requires the pinned preprocessing implementation')

    @property
    def max_input_tokens(self):
        return self.context_length - self.output_reserve - self.backend_boundary_reserve

    @property
    def trigger_tokens(self):
        return math.ceil(self.trigger_fraction * self.max_input_tokens)

    @property
    def digest(self):
        return sha256(canonical(asdict(self)).encode()).hexdigest()

    def capacity(self):
        return dict(context_window=self.context_length, output_reserve=self.output_reserve,
                    backend_boundary_reserve=self.backend_boundary_reserve,
                    max_input_tokens=self.max_input_tokens, trigger_tokens=self.trigger_tokens,
                    trigger_fraction=self.trigger_fraction, enable_thinking=self.enable_thinking,
                    capacity_mode='native_checkpoint_context', model_name=self.model_name,
                    model_profile_sha256=self.digest, output_and_input_share_context=True)

    def generation_profile(self):
        return dict(temperature=self.temperature, top_p=self.top_p, top_k=self.top_k,
                    enable_thinking=self.enable_thinking, max_output=self.output_reserve,
                    seed_policy='sha256(20260917,task_id,stage) mod 2**31; identical across arms/concurrency',
                    source=self.model_path + '/generation_config.json',
                    deployment_overrides=['enable_thinking=false', 'output_reserve', 'paired deterministic stage seed'],
                    scope='foreground and caller-supplied LangMem LLM only; never override Mem0/LightMem native generation parameters')

    def body(self, messages, max_tokens=None, task_id='', stage=''):
        maximum = self.output_reserve if max_tokens is None else max_tokens
        if type(maximum) is not int or not 0 < maximum <= self.output_reserve:
            raise ValueError('Output exceeds the frozen request reserve')
        seed = int(sha256(canonical([20260917, task_id, stage]).encode()).hexdigest()[:16], 16) % (2**31)
        return dict(model=self.model_name, messages=deepcopy(messages),
                    temperature=self.temperature, top_p=self.top_p, top_k=self.top_k,
                    max_tokens=maximum, seed=seed, stream=True,
                    stream_options={'include_usage': True},
                    chat_template_kwargs={'enable_thinking': self.enable_thinking})

    def verify_model_files(self):
        pairs = {'config.json': self.model_config_sha256, 'tokenizer.json': self.tokenizer_json_sha256,
                 'tokenizer_config.json': self.tokenizer_config_sha256, 'chat_template.jinja': self.chat_template_sha256}
        for name, expected in pairs.items():
            if sha256((Path(self.model_path) / name).read_bytes()).hexdigest() != expected:
                raise ValueError('Frozen model metadata changed: ' + name)
        generation = json.loads((Path(self.model_path) / 'generation_config.json').read_text())
        if tuple(generation[k] for k in ('temperature', 'top_p', 'top_k')) != (self.temperature, self.top_p, self.top_k):
            raise ValueError('Actual checkpoint generation defaults changed')


def load_profile(path=None):
    return ModelProfile(**json.loads(Path(path).read_text())) if path else ModelProfile()
