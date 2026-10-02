import copy
from types import SimpleNamespace
import unittest

from toolslack.langgraph_adapter import (LangGraphLangMemAdapter, LangMemSettings,
    NativeMemoryNoop, closed_prefix_stop, legal_scope_boundaries)
from toolslack.types import Scope, digest


def history():
    return [dict(role="system", content="protected instructions"),
            dict(role="user", content="old question"),
            dict(role="assistant", content="old answer"),
            dict(role="user", content="latest question"),
            dict(role="assistant", content="", tool_calls=[dict(id="t1", type="function",
                function=dict(name="search", arguments='{"q":"x"}'))])]


class AdapterTests(unittest.IsolatedAsyncioTestCase):
    async def adapter(self, **kwargs):
        self.inputs = []
        async def native(messages, **options):
            self.inputs.append((copy.deepcopy(messages), options))
            return SimpleNamespace(messages=[dict(role="user", content="official summary")],
                                   running_summary=object())
        async def model(*args):
            raise AssertionError("the injected native test never executes a model")
        async def exact(body):
            self.counted = copy.deepcopy(body)
            return 123
        return LangGraphLangMemAdapter("test-model", LangMemSettings(), lambda m: len(m),
            model, body_settings=dict(temperature=0), exact_body_counter=exact,
            native_summarizer=kwargs.get("native_summarizer", native),
            to_native=lambda m: m, from_native=lambda m: m, runnable_factory=lambda f: f)

    async def test_only_closed_scope_to_native_and_pending_suffix_preserved(self):
        messages = history()
        original = copy.deepcopy(messages)
        adapter = await self.adapter()
        scope = Scope(3, 17, 10, .1, .2, start=1, source_sha256=digest(messages[1:3]))
        result = await adapter.compact(dict(messages=messages, protected_from=4), scope)
        self.assertEqual(self.inputs[0][0], messages[1:3])
        self.assertEqual(messages, original)
        future = result.metadata["known_future_body"]["messages"]
        self.assertEqual(future[-2:], original[-2:])
        self.assertEqual(future[0], original[0])
        self.assertEqual(result.stable_body["messages"], future[:-1])
        self.assertEqual(result.metadata["stable_prefix_tokens"], 123)
        self.assertEqual(self.counted, result.stable_body)
        self.assertNotEqual(result.stable_body, result.metadata["known_future_body"])

    async def test_native_threshold_configuration_unchanged(self):
        adapter = await self.adapter()
        await adapter.compact(dict(messages=history(), protected_from=4), Scope(3, 17, 10,.1,.2,start=1))
        options = self.inputs[0][1]
        self.assertEqual((options["max_tokens"], options["max_tokens_before_summary"],
                          options["max_summary_tokens"]), (8192,4096,384))
        self.assertIsNone(options["running_summary"])

    async def test_native_system_summary_preserves_exact_prepared_consumer_wire(self):
        async def native(messages, **options):
            return SimpleNamespace(messages=[dict(role="system",content="official summary")],
                                   running_summary=object())
        adapter = await self.adapter(native_summarizer=native)
        messages = history()
        snapshot = adapter.body(messages)
        adapter.bind_snapshot(snapshot,task_id="task",protected_from=4)
        result = await adapter.compact(snapshot,Scope(3,17,10,.1,.2,start=1))
        future = result.metadata["known_future_body"]
        self.assertEqual([m["role"] for m in future["messages"][:2]],["system","system"])
        self.assertEqual(future["messages"][0],messages[0])
        # Materializing native replacement for the consumer must retain exactly
        # the same two system messages as KV preparation, without normalization.
        current = [*messages[:1],*result.replacement_messages,*messages[3:],
                   dict(role="tool",tool_call_id="t1",content="real result")]
        consumer = adapter.body(current)
        stop = result.metadata["closed_prefix_messages"]
        self.assertEqual(consumer["messages"][:stop],result.stable_body["messages"])
        self.assertEqual(consumer["messages"][:len(future["messages"])],future["messages"])
        self.assertEqual(self.counted,result.stable_body)

    async def test_noop_does_not_claim_prepared_memory(self):
        async def noop(*args, **kwargs):
            return SimpleNamespace(messages=args[0], running_summary=None)
        adapter = await self.adapter(native_summarizer=noop)
        with self.assertRaises(NativeMemoryNoop):
            await adapter.compact(dict(messages=history(), protected_from=4), Scope(3,17,10,.1,.2,start=1))

    async def test_unresolved_scope_or_source_mutation_rejected(self):
        adapter = await self.adapter()
        with self.assertRaises(ValueError):
            await adapter.compact(dict(messages=history(), protected_from=4), Scope(5,17,10,.1,.2,start=1))
        with self.assertRaises(ValueError):
            await adapter.compact(dict(messages=history(), protected_from=4), Scope(3,17,10,.1,.2,start=1,source_sha256="a"*64))
        self.assertFalse(self.inputs)

    def test_closed_boundary_tracks_all_parallel_results(self):
        messages = history()
        messages[-1]["tool_calls"].append(dict(id="t2", function=dict(name="search",arguments="{}")))
        messages.append(dict(role="tool", tool_call_id="t1", content="partial"))
        self.assertEqual(closed_prefix_stop(messages), 4)
        messages.append(dict(role="tool", tool_call_id="t2", content="complete"))
        self.assertEqual(closed_prefix_stop(messages), len(messages))

    def test_scope_candidates_exclude_instructions_and_open_dispatch(self):
        boundaries = legal_scope_boundaries(history(),4,3)
        self.assertTrue(boundaries)
        self.assertTrue(all(start==1 and stop<=4 for start,stop in boundaries))


if __name__ == "__main__":
    unittest.main()
