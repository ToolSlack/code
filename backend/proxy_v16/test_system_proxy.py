import copy
import unittest

from model_proxy import normalize, priority_body, sha, validate_terminal
from sglang_exact_count import ClosedPrefixTokenizer


class SystemProxyTests(unittest.TestCase):
    def test_priority_does_not_change_content_defaults(self):
        native = dict(messages=[dict(role='user', content='native input')],
                      temperature=.1, top_p=.1, max_tokens=2000,
                      response_format={'type': 'json_object'})
        original = copy.deepcopy(native)
        for kind, required, priority in [('foreground', False, 100),
                                          ('memory', False, 0),
                                          ('memory', True, 100),
                                          ('kv_prefill', False, -10)]:
            wire = priority_body(native, kind, required)
            self.assertEqual(wire.pop('priority'), priority)
            self.assertEqual(wire, native)
        self.assertEqual(native, original)

    def test_invalid_or_conflicting_priority_rejected(self):
        for body, kind in [({'priority': True}, 'memory'),
                           ({'priority': 100}, 'memory'), ({}, 'unknown')]:
            with self.assertRaises(ValueError):
                priority_body(body, kind)

    def test_native_provider_settings_are_preserved(self):
        body = dict(messages=[dict(role='user', content='q')], temperature=.1,
                    top_p=.1, max_tokens=2000, response_format={'type':'json_object'})
        out = priority_body(normalize(body), 'memory')
        for field in ('messages', 'temperature', 'top_p', 'max_tokens', 'response_format'):
            self.assertEqual(out[field], body[field])

    def test_closed_prefix_changes_only_generation_header_flag(self):
        class Tokenizer:
            def apply_chat_template(self, *args, **kwargs):
                self.args, self.kwargs = args, kwargs
                return [1, 2, 3]
        underlying = Tokenizer()
        proxy = ClosedPrefixTokenizer(underlying)
        messages = [{'role': 'user', 'content': 'known text'}]
        tools = [{'type': 'function', 'function': {'name': 'lookup'}}]
        ids = proxy.apply_chat_template(messages, add_generation_prompt=True,
                                       tokenize=True, tools=tools, enable_thinking=False)
        self.assertEqual(ids, [1, 2, 3])
        self.assertIs(underlying.args[0], messages)
        self.assertIs(underlying.kwargs['tools'], tools)
        self.assertIs(underlying.kwargs['add_generation_prompt'], False)
        self.assertIs(underlying.kwargs['enable_thinking'], False)

    def test_token_identity_is_order_sensitive(self):
        self.assertNotEqual(sha([1, 2]), sha([2, 1]))
        self.assertNotEqual(sha([1, 2]), sha([1, 2, 3]))

    def test_truncated_or_mismatched_sse_is_not_terminal(self):
        good=dict(seen_done=True, finished_choices={0},
                  usage={'prompt_tokens':12,'completion_tokens':3,'total_tokens':15},
                  measured={'input_tokens':12},body={'max_tokens':8})
        validate_terminal(**good)
        for change in [dict(seen_done=False),dict(finished_choices=set()),
                       dict(usage={'prompt_tokens':11,'completion_tokens':3,'total_tokens':14}),
                       dict(usage={'prompt_tokens':12,'completion_tokens':9,'total_tokens':21})]:
            with self.assertRaises(ValueError):validate_terminal(**dict(good,**change))


if __name__ == '__main__':
    unittest.main()
