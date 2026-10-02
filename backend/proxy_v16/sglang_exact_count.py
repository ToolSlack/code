"""CPU-only exact chat token count for the experiment's pinned SGLang stack.

This deliberately reuses SGLang's request model and preprocessing. It does not
instantiate a TokenizerManager, load model weights, or perform generation.
Pinned stack: the bundled frozen SGLang source snapshot; see vendor/native_engine/SOURCE_MANIFEST.json.
The helper supports text-only Qwen3 chat with the model's Jinja template.
"""
from copy import copy, deepcopy
from types import SimpleNamespace


class SGLangExactCounter:
    def __init__(self, tokenizer, *, context_length=131072):
        from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat

        self.tokenizer = tokenizer
        # Only the fields used by _validate_request and _process_messages in
        # the pinned implementation. No GPU-backed serving constructor runs.
        serving = OpenAIServingChat.__new__(OpenAIServingChat)
        serving.tokenizer_manager = SimpleNamespace(
            tokenizer=tokenizer,
            server_args=SimpleNamespace(
                context_length=context_length, allow_auto_truncate=False
            ),
        )
        serving.template_manager = SimpleNamespace(
            chat_template_name=None, jinja_template_content_format="string"
        )
        serving.tool_call_parser = "qwen"
        serving.reasoning_parser = "qwen3"
        serving.is_gpt_oss = False
        serving.use_dpsk_v32_encoding = False
        self._serving = serving

    def token_ids(self, body, *, add_generation_prompt=True):
        """Tokenize an already normalized, complete OpenAI request body.

        The caller must forward the same body to generation (including thinking
        kwargs). SGLang normalization mutates its request, so use a deep copy.
        Image/audio/video inputs are rejected rather than undercounted.
        """
        from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest

        for message in body.get("messages", []):
            content = message.get("content")
            if isinstance(content, list):
                if any(not isinstance(part, dict) or part.get("type") != "text"
                       for part in content):
                    raise ValueError("Exact counter supports text-only messages")
        request = ChatCompletionRequest.model_validate(deepcopy(body))
        error = self._serving._validate_request(request)
        if error:
            raise ValueError(error)
        # This step occurs in _convert_to_internal_request before
        # _process_messages; preserve it to avoid duplicate kwargs/semantics.
        if request.chat_template_kwargs:
            effort = request.chat_template_kwargs.pop("reasoning_effort", None)
            if effort is not None:
                request.reasoning_effort = effort
        serving = self._serving
        if not add_generation_prompt:
            if request.continue_final_message:
                raise ValueError('An unfinished assistant message is not a closed prefix')
            serving = copy(serving)
            serving.tokenizer_manager = copy(serving.tokenizer_manager)
            serving.tokenizer_manager.tokenizer = ClosedPrefixTokenizer(self.tokenizer)
        processed = serving._process_messages(request, is_multimodal=False)
        ids = list(processed.prompt_ids)
        if not ids or any(type(x) is not int or x < 0 for x in ids):
            raise ValueError('Invalid exact prompt token IDs')
        return ids

    def count(self, body):
        return len(self.token_ids(body))


class ClosedPrefixTokenizer:
    """Use identical native preprocessing, omitting only the generation header.

    No fictitious tool output or user message is added. The resulting prefix is
    a candidate: a later consumer must still match every token before acquiring
    the KV handle, since some templates rewrite history when messages are added.
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __getattr__(self, name):
        return getattr(self.tokenizer, name)

    def apply_chat_template(self, *args, **kwargs):
        kwargs = dict(kwargs, add_generation_prompt=False)
        return self.tokenizer.apply_chat_template(*args, **kwargs)


ExactChatCounter = SGLangExactCounter
