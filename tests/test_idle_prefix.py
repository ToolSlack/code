import asyncio
from copy import deepcopy
import unittest

from toolslack.idle_prefix import IdlePrefixManager
from toolslack.types import KVHandle, digest


class ExistingBackend:
    model_key = "test-model"
    prefetch_lead_s = .1

    def __init__(self):
        self.register_gate, self.place_gate = asyncio.Event(), asyncio.Event()
        self.register_gate.set();self.place_gate.set()
        self.register_started, self.place_started = asyncio.Event(), asyncio.Event()
        self.stop = asyncio.Event()
        self.register_calls = []
        self.place_calls = []
        self.releases = []
        self.handles = {}
        self.cache_hit = True
        self.drain_calls = 0

    async def register_existing(self, stable, future, deadline, identity):
        self.register_calls.append((deepcopy(stable), deepcopy(future), deadline, deepcopy(identity)))
        self.register_started.set()
        await self.register_gate.wait()
        if not self.cache_hit:
            return None
        handle = KVHandle(digest([1, 2, 3]), 3, "old", self.model_key, "epoch")
        self.handles[handle.handle_id] = handle
        return handle

    def device_ready(self, handle):
        return self.handles.get(handle.handle_id) is handle and handle.ready and handle.location == "hbm"

    async def place(self, handle, deadline):
        self.place_calls.append((handle, deadline));self.place_started.set()
        await self.place_gate.wait()
        return {"offload_skipped": self.stop.is_set(), "native_released_bytes": 0}

    def notify_tool_ready(self, handle):
        self.stop.set()

    async def release(self, handle):
        if handle.handle_id in self.releases:
            raise AssertionError("duplicate native release")
        self.releases.append(handle.handle_id)
        self.handles.pop(handle.handle_id, None)
        handle.ready = False;handle.location = "released"

    async def drain(self):
        self.drain_calls += 1
        for handle in list(self.handles.values()):
            await self.release(handle)


class IdlePrefixInvariantTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.backend = ExistingBackend()
        self.manager = IdlePrefixManager(self.backend, clock=lambda: 0., enable_tiering=False)
        self.stable = {"model": "test-model", "temperature": 0, "messages": [
            {"role": "system", "content": "instructions"}, {"role": "user", "content": "original history"}]}
        self.future = deepcopy(self.stable)
        self.future["messages"].append({"role": "assistant", "tool_calls": [{"id": "t", "function": {"name": "tool"}}]})
        self.current = deepcopy(self.future)
        self.current["messages"].append({"role": "tool", "tool_call_id": "t", "content": "result"})

    async def asyncTearDown(self):
        self.backend.register_gate.set();self.backend.place_gate.set()
        await self.manager.close()

    async def until(self, predicate):
        async def wait():
            while not predicate():
                await asyncio.sleep(.001)
        await asyncio.wait_for(wait(), 1)

    def dispatch(self, deadline=10):
        return self.manager.on_tool_dispatch("w", self.stable, self.future, deadline)

    async def test_adopts_existing_original_cache_without_prefill_or_generation(self):
        record = self.dispatch()
        await self.until(lambda: record.handle is not None)
        handle = self.manager.on_tool_result_ready("w", self.current, 0)
        self.assertIs(handle, record.handle)
        self.assertEqual(self.backend.place_calls, [])
        self.assertEqual(self.backend.register_calls[0][3]["body_sha256"], digest(self.stable))
        self.assertEqual(self.backend.register_calls[0][1], self.future)
        await self.manager.retire("w")
        self.assertEqual(self.backend.releases, ["old"])

    async def test_tool_ready_during_registration_does_not_wait_or_publish_late_handle(self):
        self.backend.register_gate.clear()
        self.dispatch()
        await self.backend.register_started.wait()
        self.assertIsNone(self.manager.on_tool_result_ready("w", self.current, 0))
        self.assertEqual(self.backend.releases, [])
        self.backend.register_gate.set()
        await self.manager.close()
        self.assertEqual(self.backend.releases, ["old"])

    async def test_compacted_context_supersedes_original_cache_even_when_ready(self):
        record = self.dispatch()
        await self.until(lambda: record.handle is not None)
        self.assertIsNone(self.manager.on_tool_result_ready("w", self.current, 1))
        await self.manager.retire("w")
        self.assertEqual(self.backend.releases, ["old"])

    async def test_changed_original_prefix_or_scaffold_cannot_reuse_original_handle(self):
        record = self.dispatch()
        await self.until(lambda: record.handle is not None)
        current = deepcopy(self.current);current["temperature"] = 1
        self.assertIsNone(self.manager.on_tool_result_ready("w", current, 0))
        await self.manager.retire("w")
        self.assertEqual(self.backend.releases, ["old"])

    async def test_cache_miss_does_not_fall_back_to_optional_prefill(self):
        self.backend.cache_hit = False
        record = self.dispatch()
        await record.registration_task
        self.assertIsNone(record.handle)
        self.assertEqual(len(self.backend.register_calls), 1)
        self.assertIsNone(self.manager.on_tool_result_ready("w", self.current, 0))

    async def test_placement_pending_cannot_be_consumed_and_close_waits_native_terminal(self):
        self.manager.enable_tiering = True
        self.backend.place_gate.clear()
        record = self.dispatch()
        await self.backend.place_started.wait()
        self.assertTrue(self.backend.device_ready(record.handle))
        self.assertIsNone(self.manager.on_tool_result_ready("w", self.current, 0))
        closing = asyncio.create_task(self.manager.close())
        await asyncio.sleep(0)
        self.assertFalse(closing.done())
        self.assertFalse(record.placement_task.cancelled())
        self.backend.place_gate.set()
        await closing
        self.assertEqual(self.backend.releases, ["old"])

    async def test_no_measured_transfer_lead_or_small_window_skips_tiering(self):
        self.manager.enable_tiering = True
        self.manager.transfer_lead_s = None
        record = self.dispatch()
        await record.registration_task
        self.assertEqual(self.backend.place_calls, [])
        self.assertIs(self.manager.on_tool_result_ready("w", self.current, 0), record.handle)

    async def test_expired_or_tiny_window_does_not_register(self):
        record = self.dispatch(.06)
        await record.registration_task
        self.assertEqual(self.backend.register_calls, [])

    async def test_shared_controller_close_does_not_drain_other_owned_handles(self):
        record = self.dispatch()
        await self.until(lambda: record.handle is not None)
        await self.manager.close(drain_backend=False)
        self.assertEqual(self.backend.drain_calls, 0)
        self.assertEqual(self.backend.releases, ["old"])
        await self.manager.close(drain_backend=False)
        self.assertEqual(self.backend.releases, ["old"])

    def test_unresolved_original_prefix_or_model_identity_is_rejected_before_rpc(self):
        with self.assertRaises(ValueError):
            self.manager.on_tool_dispatch("w", self.future, self.future, 10)
        with self.assertRaises(ValueError):
            self.manager.on_tool_dispatch("w", self.stable, self.future, 10, identity={"model_key": "wrong"})
        self.assertEqual(self.backend.register_calls, [])


if __name__ == "__main__":
    unittest.main()
