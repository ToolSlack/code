"""Explicit target-only DeepSeek-V4 deployment binding; no GPU on import."""
from copy import deepcopy
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from pathlib import Path

def canonical(value): return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'))

@dataclass(frozen=True)
class ModelProfile:
    model_path:str='/models/DeepSeek-V4-Flash-0731'
    model_name:str='DeepSeek-V4-Flash-0731'
    architecture:str='DeepseekV4ForCausalLM'
    context_length:int=1048576
    output_reserve:int=2048
    backend_boundary_reserve:int=1
    trigger_fraction:float=.90
    thinking:bool=False
    temperature:float=1.0
    top_p:float=1.0
    request_timeout_seconds:int=1800
    source_path:str='/opt/sglang-deepseek'
    fork_commit:str='ecd6f7f25c8fad4b2c1904a622270004b300532c'
    transformers_version:str='5.12.1'
    model_config_sha256:str='6c8f3d2d3b48707541b88f32f22ef3f0f8a6b57d8523281e2b8d3cdb0ae9a023'
    generation_config_sha256:str='5fccff80f55a4d455bbe516bdd552edf3e9623df95e99fbf2a3c3389fdf91af0'
    tokenizer_json_sha256:str='8f9f37ca37fdc4f5fd36d5cf4d3b0e8392edb4e894fd10cc0d70b4957c8633cf'
    tokenizer_config_sha256:str='6ac8c8dc065ed118161d02dd532749ae3f52c243deac27872134fae2f50d8547'
    official_encoder_sha256:str='abc0d26120250dda0ae077dc64aa28836026e61e970854aaeb792445e6a0dde6'

    def __post_init__(self):
        if (self.model_name!='DeepSeek-V4-Flash-0731' or self.architecture!='DeepseekV4ForCausalLM' or
            self.context_length!=1048576 or self.output_reserve!=2048 or self.backend_boundary_reserve!=1 or
            self.trigger_fraction!=.90 or self.thinking is not False or (self.temperature,self.top_p)!=(1.,1.)):
            raise ValueError('Frozen DeepSeek native model/capacity/sampling contract changed')
        if not Path(self.model_path).is_absolute() or not Path(self.source_path).is_absolute():
            raise ValueError('Explicit local model/fork paths required')
    @property
    def max_input_tokens(self):return self.context_length-self.output_reserve-self.backend_boundary_reserve
    @property
    def trigger_tokens(self):return math.ceil(self.trigger_fraction*self.max_input_tokens)
    @property
    def digest(self):return sha256(canonical(asdict(self)).encode()).hexdigest()
    def capacity(self):
        return dict(context_window=self.context_length,output_reserve=self.output_reserve,
            backend_boundary_reserve=self.backend_boundary_reserve,max_input_tokens=self.max_input_tokens,
            trigger_tokens=self.trigger_tokens,trigger_fraction=self.trigger_fraction,thinking=self.thinking,
            capacity_mode='native_checkpoint_context',model_name=self.model_name,model_profile_sha256=self.digest,
            output_and_input_share_context=True)
    def generation_profile(self):
        return dict(temperature=self.temperature,top_p=self.top_p,thinking=False,max_output=self.output_reserve,
            seed_policy='sha256(20260917,task_id,stage) mod 2**31; identical across arms/concurrency',
            source=self.model_path+'/generation_config.json',
            deployment_overrides=['DeepSeek chat_template_kwargs.thinking=false','output reserve2048','paired deterministic stage seed'],
            scope='foreground and caller-supplied LangMem LLM only; no Mem0/LightMem content-default overrides')
    def body(self,messages,max_tokens=None,task_id='',stage=''):
        maximum=self.output_reserve if max_tokens is None else max_tokens
        if type(maximum) is not int or not 0<maximum<=self.output_reserve:raise ValueError('Output exceeds frozen reserve')
        seed=int(sha256(canonical([20260917,task_id,stage]).encode()).hexdigest()[:16],16)%(2**31)
        return dict(model=self.model_name,messages=deepcopy(messages),temperature=self.temperature,top_p=self.top_p,
            max_tokens=maximum,seed=seed,stream=True,stream_options={'include_usage':True},chat_template_kwargs={'thinking':False})
    def verify_model_files(self):
        pairs={'config.json':self.model_config_sha256,'generation_config.json':self.generation_config_sha256,
            'tokenizer.json':self.tokenizer_json_sha256,'tokenizer_config.json':self.tokenizer_config_sha256,
            'encoding/encoding_dsv4.py':self.official_encoder_sha256}
        for name,expected in pairs.items():
            if sha256((Path(self.model_path)/name).read_bytes()).hexdigest()!=expected:raise ValueError('Model metadata changed: '+name)
        if (Path(self.source_path)/'TOOLSLACK_SOURCE_COMMIT').read_text().strip()!=self.fork_commit:raise ValueError('Fork commit changed')
        generation=json.loads((Path(self.model_path)/'generation_config.json').read_text())
        if (generation['temperature'],generation['top_p'])!=(self.temperature,self.top_p):raise ValueError('Checkpoint sampling changed')

def load_profile(path=None):return ModelProfile(**json.loads(Path(path).read_text())) if path else ModelProfile()
