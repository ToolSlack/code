"""Exact CPU-only pure-text preprocessing for the pinned Qwen conditional model.

The upstream model is multimodal even when a request contains text only.
SGLang decodes its rendered chat IDs to text, then TokenizerManager tokenizes
that text again. Reuse both actual official methods, rather than assuming the
old Qwen3-8B direct-input-ID route. No TokenizerManager constructor is run.
"""
import asyncio
from copy import deepcopy
import hashlib
import importlib.metadata
import inspect
from pathlib import Path
from types import SimpleNamespace
from qwen_profile import ModelProfile

EXPECTED_SOURCE = {
    'sglang.srt.entrypoints.openai.serving_chat': '082921f16a712e2a01518756f692f16d514067dec10c893c59831c12fa831a55',
    'sglang.srt.managers.tokenizer_manager': '5645ffcd9c61bd1d8c4c4062ebe41a6b48ae8e07891c56e6b2b7dcfb8d16d834',
}


def reject_nontext(body):
    for message in body.get('messages', []):
        content = message.get('content')
        if isinstance(content, list) and any(not isinstance(p, dict) or p.get('type') != 'text' for p in content):
            raise ValueError('Counter supports full pure-text requests only; image/audio/video are rejected')
        if any(message.get(k) is not None for k in ('audio', 'image', 'video')):
            raise ValueError('Non-text message data is not supported')


class SGLang0510Counter:
    def __init__(self, profile=None):
        import importlib
        self.profile = profile or ModelProfile()
        self.profile.verify_model_files()
        for name, expected in [('sglang', self.profile.sglang_version), ('transformers', self.profile.transformers_version)]:
            if importlib.metadata.version(name) != expected:
                raise ValueError('Unexpected preprocessing package: ' + name)
        self.source_hashes = {}
        for name, expected in EXPECTED_SOURCE.items():
            module = importlib.import_module(name)
            path = inspect.getsourcefile(module)
            actual = hashlib.sha256(Path(path).read_bytes()).hexdigest()
            if actual != expected:
                raise ValueError('Official preprocessing source changed: ' + name)
            self.source_hashes[path] = actual
        from sglang.srt.managers.tokenizer_manager import TokenizerManager, _get_processor_wrapper
        from sglang.srt.utils.hf_transformers_utils import get_tokenizer_from_processor
        from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
        args = SimpleNamespace(tokenizer_path=self.profile.model_path, tokenizer_mode='auto',
                               trust_remote_code=False, revision=None, disable_fast_image_processor=False,
                               context_length=self.profile.context_length, allow_auto_truncate=False)
        self.processor = _get_processor_wrapper(args)
        self.tokenizer = get_tokenizer_from_processor(self.processor)
        tm = TokenizerManager.__new__(TokenizerManager)
        tm.tokenizer = self.tokenizer
        tm.async_dynamic_batch_tokenizer = None
        tm.server_args = args
        tm.model_config = SimpleNamespace(is_multimodal=True)
        self.tm = tm
        serving = OpenAIServingChat.__new__(OpenAIServingChat)
        serving.tokenizer_manager = tm
        serving.template_manager = SimpleNamespace(chat_template_name=None, jinja_template_content_format='string')
        serving.tool_call_parser = self.profile.tool_call_parser
        serving.reasoning_parser = self.profile.reasoning_parser
        serving.is_gpt_oss = False
        serving.use_dpsk_v32_encoding = False
        self.serving = serving

    def input_ids(self, body):
        from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest
        reject_nontext(body)
        request = ChatCompletionRequest.model_validate(deepcopy(body))
        error = self.serving._validate_request(request)
        if error:
            raise ValueError(error)
        if request.chat_template_kwargs:
            effort = request.chat_template_kwargs.pop('reasoning_effort', None)
            if effort is not None:
                request.reasoning_effort = effort
        processed = self.serving._process_messages(request, is_multimodal=True)
        if processed.image_data or processed.video_data or processed.audio_data:
            raise ValueError('Unexpected multimodal content')
        ids, _ = asyncio.run(self.tm._tokenize_texts(processed.prompt, False))
        if not isinstance(ids, list) or any(type(i) is not int for i in ids):
            raise TypeError('Upstream preprocessing did not produce a flat token list')
        return ids

    def count(self, body):
        return len(self.input_ids(body))
