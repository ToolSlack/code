"""Explicit model-family routing; preserve the real native encoding path."""
import json
from pathlib import Path
from server_profile import ServerProfile


def load_profile(path=None):
    if path is None:return ServerProfile()
    data=json.loads(Path(path).read_text())
    name=data.get('model_name')
    if name=='Qwen3-8B':return ServerProfile(**data)
    if name=='Qwen3.8-27B':
        from qwen_profile import ModelProfile
        return ModelProfile(**data)
    if name=='DeepSeek-V4-Flash-0731':
        from dsv4_profile import ModelProfile
        return ModelProfile(**data)
    raise ValueError('No audited native adapter for this served model')


def template_kwargs(value,profile):
    if value is None:value={}
    if not isinstance(value,dict):raise ValueError('chat_template_kwargs must be an object')
    if profile.model_name=='DeepSeek-V4-Flash-0731':
        if set(value)-{'thinking'}:raise ValueError('Unsupported DeepSeek template override')
        key='thinking';expected=profile.thinking
    elif profile.model_name in {'Qwen3-8B','Qwen3.8-27B'}:
        if 'thinking' in value:raise ValueError('DeepSeek template option is invalid for Qwen')
        key='enable_thinking';expected=profile.enable_thinking
    else:raise ValueError('Unsupported model family')
    if key in value and (type(value[key]) is not bool or value[key]!=expected):
        raise ValueError('Thinking mode differs from the frozen deployment')
    return {**value,key:expected}


def make_counter(profile):
    if profile.model_name=='Qwen3-8B':
        from transformers import AutoTokenizer
        from sglang_exact_count import SGLangExactCounter
        tokenizer=AutoTokenizer.from_pretrained(profile.model_path,local_files_only=True)
        return SGLangExactCounter(tokenizer,context_length=profile.context_length)
    if profile.model_name=='Qwen3.8-27B':
        from closed_counters import Qwen35ClosedCounter
        return Qwen35ClosedCounter(profile)
    if profile.model_name=='DeepSeek-V4-Flash-0731':
        from closed_counters import DeepSeekClosedCounter
        return DeepSeekClosedCounter(profile)
    raise ValueError('No audited native preprocessing implementation')
