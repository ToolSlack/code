"""CPU native server preprocessing, cross-checked with model's official encoder.

No Jinja template, model construction, custom memory prompt, truncation, GPU,
network client, or external inference. Server protocol normalization remains
upstream. Reference preparation mirrors its text/assistant/tool transport rules.
"""
from copy import deepcopy
import hashlib
import importlib.metadata
import importlib.util
import inspect
from pathlib import Path
import sys
from types import SimpleNamespace
from dsv4_profile import ModelProfile

EXPECTED={
 'sglang.srt.entrypoints.openai.serving_chat':'2b2edf314710a0746748ce5955e62dbaf6dcdbff36c371abafac84161682dc89',
 'sglang.srt.entrypoints.openai.encoding_dsv4':'012e4dc254c4046f600674eaa799d59dffe5e4a46450da7b4629cf16706fc6c3',
 'sglang.srt.entrypoints.openai.protocol':'4ddf0dc7ec1e158ccead13afae413bd7ad546e0dc515df503a5bfce19b622f79'}

class DeepSeekCounter:
    def __init__(self,profile=None):
        self.profile=profile or ModelProfile();self.profile.verify_model_files()
        if importlib.metadata.version('transformers')!=self.profile.transformers_version:raise ValueError('Pinned transformers changed')
        sys.path.insert(0,str(Path(self.profile.source_path)/'python'))
        from transformers import AutoTokenizer
        from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
        self.source_hashes={}
        for name,expected in EXPECTED.items():
            mod=importlib.import_module(name);path=inspect.getsourcefile(mod)
            actual=hashlib.sha256(Path(path).read_bytes()).hexdigest()
            if actual!=expected:raise ValueError('Native server source changed: '+name)
            self.source_hashes[path]=actual
        self.tokenizer=AutoTokenizer.from_pretrained(self.profile.model_path,local_files_only=True,trust_remote_code=False)
        path=Path(self.profile.model_path)/'encoding/encoding_dsv4.py'
        spec=importlib.util.spec_from_file_location('deepseek_reference_encoding',path)
        self.official=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.official)
        self.source_hashes[str(path)]=self.profile.official_encoder_sha256
        server=OpenAIServingChat.__new__(OpenAIServingChat)
        server.tokenizer_manager=SimpleNamespace(tokenizer=self.tokenizer,
            server_args=SimpleNamespace(context_length=self.profile.context_length,allow_auto_truncate=False,reasoning_parser=None),
            model_config=SimpleNamespace(is_multimodal=False))
        server.template_manager=SimpleNamespace(chat_template_name=None,jinja_template_content_format='string',reasoning_config=None)
        server.default_chat_template_kwargs={};server.tool_call_parser=None;server.reasoning_parser=None
        server.is_gpt_oss=False;server.is_gemma4=False;server.chat_encoding_spec='dsv4'
        self.serving=server

    def input_ids(self,body):
        from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest
        from sglang.srt.parser.jinja_template_utils import process_content_for_template_format
        if body.get('model')!=self.profile.model_name or body.get('chat_template_kwargs')!={'thinking':False}:
            raise ValueError('Exact served model and DeepSeek thinking:false required')
        if body.get('input_ids') is not None or body.get('task') is not None or body.get('continue_final_message'):
            raise ValueError('This full-message counter does not support input-ID/task/prefix overrides')
        for msg in body.get('messages',[]):
            if not isinstance(msg.get('content'),(str,type(None))):raise ValueError('Text strings only; no silent multimodal removal')
        request=ChatCompletionRequest.model_validate(deepcopy(body))
        error=self.serving._validate_request(request)
        if error:raise ValueError(error)
        processed=self.serving._process_messages(deepcopy(request),is_multimodal=False)
        if processed.image_data or processed.video_data or processed.audio_data:raise ValueError('Unexpected media')
        ids=processed.prompt_ids
        messages=[m.model_dump() for m in request.messages]
        for msg in messages:
            # The model release encoder requires JSON-string tool arguments.
            # The real serving path parses these into dicts internally; its
            # adapted encoder accepts those dicts. Keep wire strings here so
            # both original encoders receive their documented input schema.
            if msg.get('content') is None:msg['content']=''
            msg.update(process_content_for_template_format(msg,'string',[],[],[],[],use_dpsk_v32_encoding=False))
        messages,prefix=self.serving._handle_last_assistant_message(messages,request)
        if prefix is not None:raise ValueError('Unexpected continuation prefix')
        if messages[0]['role']!='system':messages.insert(0,{'role':'system','content':''})
        if request.tools:messages[0]['tools']=[t.model_dump() for t in request.tools]
        effort=request.reasoning_effort if request.reasoning_effort in ('max','high') else None
        official_prompt=self.official.encode_messages(messages,thinking_mode='chat',reasoning_effort=effort)
        official_ids=self.tokenizer.encode(official_prompt)
        if ids!=official_ids:raise ValueError('Complete official reference IDs differ from actual fork preprocessing')
        if not isinstance(ids,list) or any(type(i) is not int for i in ids):raise TypeError('Native preprocessing must produce flat IDs')
        return ids
    def count(self,body):return len(self.input_ids(body))
