"""Exact complete and closed-prefix IDs, preserving each native encoding path.

This changes no memory algorithm, source text, tokenizer, or model weights.
Every closed prefix must be a token-for-token prefix of the complete request;
a future consumer must be checked again because native templates may rewrite it.
"""
from copy import copy,deepcopy


class ClosedTokenizer:
    def __init__(self,tokenizer):self.tokenizer=tokenizer
    def __getattr__(self,key):return getattr(self.tokenizer,key)
    def apply_chat_template(self,*args,**kwargs):
        kwargs=dict(kwargs,add_generation_prompt=False)
        return self.tokenizer.apply_chat_template(*args,**kwargs)


def require_prefix(full,closed):
    if not closed or full[:len(closed)]!=closed:
        raise ValueError('Native closed serialization is not an exact prefix of the complete consumer')
    return closed


class Qwen35ClosedCounter:
    def __init__(self,profile):
        from sglang0510_counter import SGLang0510Counter
        self.native=SGLang0510Counter(profile)
        self.tokenizer=self.native.tokenizer
        self.source_hashes=self.native.source_hashes

    def token_ids(self,body,*,add_generation_prompt=True):
        full=self.native.input_ids(body)
        if add_generation_prompt:return full
        if body.get('continue_final_message'):
            raise ValueError('An unfinished assistant message is not a closed prefix')
        # Preserve the conditional-model decode/re-tokenize route. Only the
        # public Jinja generation-header option changes, on an isolated wrapper.
        clone=copy(self.native);clone.serving=copy(self.native.serving)
        clone.serving.tokenizer_manager=copy(self.native.serving.tokenizer_manager)
        clone.serving.tokenizer_manager.tokenizer=ClosedTokenizer(self.tokenizer)
        return require_prefix(full,clone.input_ids(body))


class DeepSeekClosedCounter:
    def __init__(self,profile):
        from dsv4_counter import DeepSeekCounter
        self.native=DeepSeekCounter(profile)
        self.profile=profile;self.tokenizer=self.native.tokenizer
        self.source_hashes=self.native.source_hashes

    def _ids(self,body):
        from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest
        from sglang.srt.parser.jinja_template_utils import process_content_for_template_format
        if body.get('model')!=self.profile.model_name or body.get('chat_template_kwargs')!={'thinking':False}:
            raise ValueError('Exact DeepSeek served model and thinking:false required')
        if body.get('input_ids') is not None or body.get('task') is not None or body.get('continue_final_message'):
            raise ValueError('Complete messages required; task and continuation overrides unsupported')
        for message in body.get('messages',[]):
            content=message.get('content')
            if not isinstance(content,(str,type(None))):
                if not isinstance(content,list) or any(not isinstance(part,dict) or part.get('type')!='text' for part in content):
                    raise ValueError('Non-text content cannot be silently dropped')
        request=ChatCompletionRequest.model_validate(deepcopy(body))
        error=self.native.serving._validate_request(request)
        if error:raise ValueError(error)
        actual=self.native.serving._process_messages(deepcopy(request),is_multimodal=False)
        if actual.image_data or actual.video_data or actual.audio_data:
            raise ValueError('Unexpected media in text-only DeepSeek deployment')
        messages=[m.model_dump() for m in request.messages]
        for message in messages:
            if message.get('content') is None:message['content']=''
            message.update(process_content_for_template_format(message,'string',[],[],[],[],use_dpsk_v32_encoding=False))
        messages,prefix=self.native.serving._handle_last_assistant_message(messages,request)
        if prefix is not None:raise ValueError('Unexpected continuation prefix')
        if messages[0]['role']!='system':messages.insert(0,{'role':'system','content':''})
        if request.tools:messages[0]['tools']=[tool.model_dump() for tool in request.tools]
        effort=request.reasoning_effort if request.reasoning_effort in ('max','high') else None
        official=self.native.official
        prompt=official.encode_messages(messages,thinking_mode='chat',reasoning_effort=effort)
        full=self.tokenizer.encode(prompt)
        if full!=actual.prompt_ids:
            raise ValueError('Actual serving and official reference encoding disagree')
        return full,prompt,messages

    def token_ids(self,body,*,add_generation_prompt=True):
        full,prompt,messages=self._ids(body)
        if add_generation_prompt:return full
        official=self.native.official
        merged=official.merge_tool_messages(deepcopy(messages))
        last=merged[-1]
        if last.get('role') in ('user','developer'):
            if last.get('task') is not None:raise ValueError('Task transition is not supported')
            # Official render_message appends this transition to the final
            # user/developer message in chat mode. It contains no source text.
            suffix=official.ASSISTANT_SP_TOKEN+official.thinking_end_token
            if not prompt.endswith(suffix):raise ValueError('Official generation transition changed')
            closed=self.tokenizer.encode(prompt[:-len(suffix)])
        else:
            closed=full
        return require_prefix(full,closed)

    def future_position_prefix(self, closed_body, known_future_body):
        """Encode a completed final summary in its future historical position.

        This narrowly handles the native chat adapter's final-assistant-to-user
        rewrite. The last completed assistant must remain separated from the
        protected dispatch. No dispatch content or unknown tool result enters
        the returned object. Other layouts retain the conservative LCP path.
        """
        closed = closed_body.get('messages') or []
        if (not closed or closed[-1].get('role') != 'assistant'
                or closed[-1].get('tool_calls')):
            return None
        if (known_future_body.get('chat_template_kwargs') != {'thinking': False}
                or not known_future_body.get('tools')):
            return None
        if any(message.get('task') is not None or message.get('wo_eos')
               for message in known_future_body.get('messages', [])):
            return None
        full_ids, full_prompt, messages = self._ids(known_future_body)
        if (len(messages) < 2 or messages[-2].get('role') != 'assistant'
                or messages[-2].get('tool_calls')
                or messages[-1].get('role') != 'user'):
            raise ValueError('Native normalization changed the completed boundary')
        official = self.native.official
        merged = official.sort_tool_results_by_call_order(
            official.merge_tool_messages(deepcopy(messages)))
        if (len(merged) < 2 or merged[-2].get('role') != 'assistant'
                or merged[-1].get('role') != 'user'):
            raise ValueError('Tool-message merging crossed the completed boundary')
        # Render in the complete known future, never as a standalone final
        # assistant request. Chat mode plus the pinned encoder means unknown
        # later tool content cannot change these completed messages.
        prefix_prompt = official.bos_token + ''.join(
            official.render_message(i, merged, thinking_mode='chat',
                                    drop_thinking=False, reasoning_effort=None)
            for i in range(len(merged) - 1))
        if not full_prompt.startswith(prefix_prompt):
            raise ValueError('Completed future rendering differs from native prompt')
        prefix_ids = self.tokenizer.encode(prefix_prompt)
        # BPE may join a boundary token with a later suffix. Keep only the
        # exact common token prefix; consumer admission validates it again.
        boundary = 0
        for left, right in zip(prefix_ids, full_ids):
            if left != right:
                break
            boundary += 1
        if not boundary:
            raise ValueError('Completed future has no exact token prefix')
        return prefix_ids[:boundary]
