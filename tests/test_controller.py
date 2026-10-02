import asyncio
from copy import deepcopy
from dataclasses import replace
import unittest

from toolslack.controller import Job, ToolSlackController
from toolslack.idle_prefix import IdlePrefixManager
from toolslack.policy import LevelSelector, ProfileKey, ProfileStore, ToolSignature, WindowPredictor
from toolslack.types import KVHandle, NativeResult, Scope, digest


class Clock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        return self.value


class NativeBackend:
    """CPU ownership double; no fake measurements are exported as GPU results."""
    model_key = "test-model"

    def __init__(self):
        self.memory_gate, self.kv_gate, self.placement_gate = asyncio.Event(), asyncio.Event(), asyncio.Event()
        self.memory_gate.set();self.kv_gate.set();self.placement_gate.set()
        self.placement_started = asyncio.Event()
        self.placement_stop = asyncio.Event()
        self.prepare_started = asyncio.Event()
        self.compact_started = asyncio.Event()
        self.memory_calls = []
        self.kv_calls = []
        self.handles = {}
        self.release_calls = []
        self.drain_calls = 0
        self.metadata = {}
        self.mutate_result = None
        self.force_not_ready = False
        self.existing_calls = []

    async def register_existing(self, stable, future, deadline, identity):
        self.existing_calls.append((deepcopy(stable), deepcopy(future), deepcopy(identity)))
        handle = KVHandle(digest([9]*500), 500, "original", self.model_key, "epoch")
        self.handles[handle.handle_id] = handle
        return handle

    async def compact(self, snapshot, scope):
        self.memory_calls.append((deepcopy(snapshot), scope))
        self.compact_started.set()
        await self.memory_gate.wait()
        replacement = [{"role": "user", "content": "native summary"}]
        candidate = snapshot["messages"][:scope.start]+replacement+snapshot["messages"][scope.stop:]
        future = deepcopy(snapshot);future["messages"] = deepcopy(candidate)
        stable = deepcopy(future);stable["messages"] = deepcopy(candidate[:-1])
        result = NativeResult(replacement, stable, {
            "known_future_body": future, "stable_prefix_tokens": 300, **deepcopy(self.metadata)})
        if self.mutate_result:
            self.mutate_result(result)
        return result

    async def prepare_kv(self, body, tokens, deadline, identity):
        self.kv_calls.append((deepcopy(body), tokens, deadline, deepcopy(identity)))
        self.prepare_started.set()
        await self.kv_gate.wait()
        handle_id = "h"+str(len(self.kv_calls))
        handle = KVHandle(digest([1]*tokens), tokens, handle_id, self.model_key,
                          "epoch", metadata={"body_sha256": digest(body)})
        self.handles[handle_id] = handle
        return handle

    def device_ready(self, handle):
        return (not self.force_not_ready and self.handles.get(handle.handle_id) is handle
                and handle.ready and handle.location == "hbm")

    def consumer_headers(self, handle):
        if not self.device_ready(handle):
            raise ValueError("unconfirmed ready")
        return {"x-toolslack-kv-handle": handle.handle_id, "x-toolslack-kv-one-shot": "true"}

    def notify_tool_ready(self, handle):
        self.placement_stop.set()

    async def place(self, handle, deadline):
        self.placement_started.set()
        # Model an in-flight native operation whose Python flag has not yet
        # changed. The controller must gate residency independently.
        await self.placement_gate.wait()
        if self.placement_stop.is_set():
            return {"offload_skipped": True}
        return {"device_ready": True}

    async def release(self, handle):
        if handle.handle_id in self.release_calls:
            raise AssertionError("double native release")
        self.release_calls.append(handle.handle_id)
        handle.ready = False;handle.location = "released"
        self.handles.pop(handle.handle_id, None)
        return {"released": True}

    async def drain(self):
        self.drain_calls += 1
        for handle in list(self.handles.values()):
            await self.release(handle)
        return {"drained": True}


class ControllerInvariantTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.clock = Clock()
        self.backend = NativeBackend()
        self.key = ProfileKey("test-model", "native", "output=64", "workers=1", "cold", "cpu-invariants")
        self.profiles = ProfileStore(bucket_size=1)
        for tokens, cost in [(100, .05), (200, .1), (300, .15)]:
            self.profiles.observe_cost("memory", tokens, cost, self.key)
            self.profiles.observe_cost("kv", tokens, cost/2, self.key)
        self.selector = LevelSelector(self.profiles, self.key)
        self.predictor = WindowPredictor()
        self.signature = ToolSignature("test-model", "external-test", "fixed-input", "test-env", "cpu-invariants")
        self.snapshot = {"model": "test-model", "tools": [{"type": "function", "function": {"name": "external-test"}}],
                         "chat_template_kwargs": {"thinking": False}, "messages": [
            {"role": "system", "content": "protected instructions"},
            {"role": "user", "content": "old history"},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": "unselected newer history"},
            {"role": "assistant", "tool_calls": [{"id": "a", "type": "function",
                "function": {"name": "external-test", "arguments": "{}"}}]}]}
        self.scope = Scope(3, 100, 100, 1, 3, start=1,
                           source_sha256=digest(self.snapshot["messages"][1:3]))
        self.controllers = []

    async def asyncTearDown(self):
        self.backend.memory_gate.set();self.backend.kv_gate.set();self.backend.placement_gate.set()
        for controller in self.controllers:
            await controller.close()

    async def controller(self, **kwargs):
        controller = ToolSlackController(self.backend, self.selector, clock=self.clock,
                                         safety_margin_s=.01, **kwargs)
        self.controllers.append(controller)
        return await controller.start()

    def dispatch(self, controller, window_id="w", *, scopes=None, tools=("a",), duration=10):
        prediction = self.predictor.predict(self.signature, self.clock(), explicit_calibration_s=duration)
        return controller.on_tool_dispatch(window_id, self.snapshot, scopes or [self.scope], prediction,
                                           protected_from=4, required_tools=tools)

    def current(self):
        return deepcopy(self.snapshot["messages"])+[{"role": "tool", "tool_call_id": "a", "content": "actual result"}]

    async def until(self, predicate):
        async def wait():
            while not predicate():
                await asyncio.sleep(.001)
        await asyncio.wait_for(wait(), timeout=1)

    async def test_exact_closed_prefix_kv_identity_and_full_protected_suffix(self):
        controller = await self.controller()
        self.dispatch(controller)
        await self.until(lambda: controller.windows["w"].kv is not None)
        body, _, _, identity = self.backend.kv_calls[0]
        self.assertNotIn("tool_calls", body["messages"][-1])
        self.assertEqual(identity["model_key"], "test-model")
        self.assertEqual(identity["known_future_body"]["messages"][-1], self.snapshot["messages"][-1])
        state = controller.on_tool_result_ready("w", "a", self.current())
        self.assertEqual(state.level, 2)
        self.assertEqual(state.messages[-2:], self.current()[-2:])
        self.assertEqual(state.messages[0], self.snapshot["messages"][0])
        self.assertEqual(state.request_metadata["headers"]["x-toolslack-kv-one-shot"], "true")
        self.assertNotIn("toolslack_kv_handle_id", state.request_metadata)

    async def test_tool_return_never_waits_for_pending_native_memory(self):
        self.backend.memory_gate.clear()
        controller = await self.controller()
        self.dispatch(controller)
        await self.backend.compact_started.wait()
        state = controller.on_tool_result_ready("w", "a", self.current())
        self.assertEqual(state.level, 0)
        self.assertEqual(state.messages, self.current())
        closing = asyncio.create_task(controller.close())
        await asyncio.sleep(0)
        self.assertFalse(closing.done())
        self.backend.memory_gate.set()
        await closing
        self.assertEqual(self.backend.kv_calls, [])
        self.assertIsNone(controller.windows["w"].result)

    async def test_tool_return_during_kv_keeps_l1_and_late_handle_is_released_once(self):
        self.backend.kv_gate.clear()
        controller = await self.controller()
        self.dispatch(controller)
        await self.backend.prepare_started.wait()
        state = controller.on_tool_result_ready("w", "a", self.current())
        self.assertEqual(state.level, 1)
        self.backend.kv_gate.set()
        await controller.close()
        self.assertEqual(self.backend.release_calls, ["h1"])
        with self.assertRaises(RuntimeError):
            controller.on_tool_result_ready("w", "a", self.current())

    async def test_parallel_tool_dependencies_do_not_consume_an_early_result(self):
        controller = await self.controller(enable_kv=False)
        self.dispatch(controller, tools=("a", "b"))
        await self.until(lambda: controller.windows["w"].result is not None)
        self.assertIsNone(controller.on_tool_result_ready("w", "a", self.current()))
        self.assertFalse(controller.windows["w"].consumed)
        with self.assertRaises(ValueError):
            controller.on_tool_result_ready("w", "a", self.current())
        state = controller.on_tool_result_ready("w", "b", self.current()+[
            {"role": "tool", "tool_call_id": "b", "content": "second result"}])
        self.assertEqual(state.level, 1)
        self.assertEqual(state.messages[-1]["tool_call_id"], "b")

    async def test_unresolved_dispatch_in_stable_body_is_rejected_before_kv(self):
        self.backend.mutate_result = lambda result: result.stable_body.update(
            messages=deepcopy(result.metadata["known_future_body"]["messages"]))
        controller = await self.controller()
        self.dispatch(controller)
        await self.until(lambda: any(e["event"] == "stage_failed" for e in controller.events))
        state = controller.on_tool_result_ready("w", "a", self.current())
        self.assertEqual(state.level, 0)
        self.assertEqual(self.backend.kv_calls, [])

    async def test_changed_template_scaffold_is_rejected(self):
        self.backend.mutate_result = lambda result: result.stable_body.update(chat_template_kwargs={"thinking": True})
        controller = await self.controller()
        self.dispatch(controller)
        await self.until(lambda: any(e["event"] == "stage_failed" for e in controller.events))
        self.assertEqual(controller.on_tool_result_ready("w", "a", self.current()).level, 0)

    async def test_changed_replaced_prefix_forces_original_current_context(self):
        controller = await self.controller()
        self.dispatch(controller)
        await self.until(lambda: controller.windows["w"].kv is not None)
        current = self.current();current[1]["content"] = "edited old history"
        state = controller.on_tool_result_ready("w", "a", current)
        self.assertEqual(state.level, 0)
        self.assertEqual(state.messages, current)
        await controller.close()
        self.assertEqual(self.backend.release_calls, ["h1"])

    async def test_changed_unselected_prefix_preserves_l1_but_invalidates_l2(self):
        controller = await self.controller()
        self.dispatch(controller)
        await self.until(lambda: controller.windows["w"].kv is not None)
        current = self.current();current[3]["content"] = "edited newer history"
        state = controller.on_tool_result_ready("w", "a", current)
        self.assertEqual(state.level, 1)
        self.assertEqual(state.messages[2]["content"], "edited newer history")
        await controller.close()
        self.assertEqual(self.backend.release_calls, ["h1"])

    async def test_kv_refresh_keeps_actual_prefix_length_and_marginal_benefit(self):
        self.backend.kv_gate.clear()
        controller = await self.controller()
        self.dispatch(controller)
        await self.backend.prepare_started.wait()
        window = controller.windows["w"]
        plan = self.selector.rebudget_kv(window.plan, 300, self.clock(), .01,
                                         kv_lengths=(100,), queue_kv_s=0)
        job = Job("partial", window, "kv", plan, self.clock(), 99)
        refreshed = controller._refresh(job, self.clock())
        self.assertEqual(refreshed.scope.new_prefix_tokens, 300)
        self.assertAlmostEqual(refreshed.gain_s, 2/3)
        self.assertEqual(refreshed.memory_s, 0)
        self.assertEqual(refreshed.initial_budget, window.plan.initial_budget)

    async def test_future_kv_queue_counts_for_memory_deadline_but_not_utility_service(self):
        self.backend.memory_gate.clear()
        controller = await self.controller(max_running=1)
        self.dispatch(controller)
        await self.backend.compact_started.wait()
        memory_job = next(iter(controller.running.values()))[0]
        future_plan = replace(memory_job.plan, memory_s=0, kv_s=2)
        queued = Job("queued-kv", memory_job.window, "kv", future_plan, self.clock(), 99)
        controller.pending.append(queued)
        service = controller._service_cost(memory_job)
        path = controller._path_cost(memory_job)
        self.assertGreater(path, service)
        self.assertAlmostEqual(path-service, controller._queue_cost("kv", exclude_job=memory_job))
        self.assertEqual(controller._service_cost(memory_job), service)

    async def test_inflight_placement_is_not_consumed_despite_old_hbm_flag(self):
        self.backend.placement_gate.clear()
        controller = await self.controller(enable_tiering=True, transfer_lead_s=.1)
        self.dispatch(controller)
        await self.backend.placement_started.wait()
        handle = controller.windows["w"].kv
        self.assertTrue(self.backend.device_ready(handle))
        state = controller.on_tool_result_ready("w", "a", self.current())
        self.assertEqual(state.level, 1)
        self.assertIsNone(state.kv_handle)
        self.assertEqual(self.backend.release_calls, [])
        self.backend.placement_gate.set()
        await controller.close()
        self.assertEqual(self.backend.release_calls, ["h1"])

    async def test_close_is_idempotent_and_waits_native_placement_without_cancelling(self):
        self.backend.placement_gate.clear()
        controller = await self.controller(enable_tiering=True, transfer_lead_s=.1)
        self.dispatch(controller)
        await self.backend.placement_started.wait()
        first, second = asyncio.create_task(controller.close()), asyncio.create_task(controller.close())
        await asyncio.sleep(0);await asyncio.sleep(0)
        self.assertFalse(first.done());self.assertFalse(second.done())
        self.assertFalse(controller.windows["w"].placement_task.cancelled())
        self.backend.placement_gate.set()
        await asyncio.gather(first, second)
        self.assertEqual(self.backend.release_calls, ["h1"])
        self.assertEqual(self.backend.drain_calls, 1)

    async def test_adapter_readiness_overrides_python_handle_fields(self):
        controller = await self.controller()
        self.dispatch(controller)
        await self.until(lambda: controller.windows["w"].kv is not None)
        self.backend.force_not_ready = True
        self.assertEqual(controller.on_tool_result_ready("w", "a", self.current()).level, 1)

    async def test_online_profiles_require_explicit_matching_service_telemetry(self):
        self.backend.metadata.update(cost_profile_key=vars(self.key), stage_service_s=.2, stage_queue_s=.03)
        controller = await self.controller(enable_kv=False)
        self.dispatch(controller)
        await self.until(lambda: controller.windows["w"].result is not None)
        cost = self.profiles.predict_cost("memory", 100, self.key)
        self.assertEqual(cost.service_s, .2)
        self.assertEqual(cost.queue_s, .03)
        self.assertTrue(any(e["event"] == "cost_profile_updated" for e in controller.events))

    async def test_foreground_cap_protects_active_requests_and_wakes_on_exit(self):
        controller = await self.controller(enable_kv=False, foreground_limit=1)
        async with controller.foreground():
            self.dispatch(controller)
            await asyncio.sleep(.01)
            self.assertEqual(self.backend.memory_calls, [])
        await self.until(lambda: controller.windows["w"].result is not None)
        self.assertEqual(len(self.backend.memory_calls), 1)

    async def test_oversized_ready_stage_does_not_block_a_smaller_session(self):
        controller = await self.controller(enable_kv=False, max_running=1, max_prefill_tokens=100)
        oversized = replace(self.scope, input_tokens=200, gain_l1_s=5)
        self.dispatch(controller, "large", scopes=[oversized])
        self.dispatch(controller, "small")
        await self.until(lambda: controller.windows["small"].result is not None)
        self.assertIsNone(controller.windows["large"].result)
        self.assertEqual(len(self.backend.memory_calls), 1)
        self.assertEqual(self.backend.memory_calls[0][1].input_tokens, 100)

    async def test_idle_original_prefix_can_accompany_l0_without_memory_or_generation(self):
        manager = IdlePrefixManager(self.backend, enable_tiering=False, clock=self.clock)
        controller = await self.controller(enable_kv=False, idle_prefix_manager=manager)
        no_gain = replace(self.scope, gain_l1_s=0, gain_l2_s=0)
        self.assertIsNone(self.dispatch(controller, scopes=[no_gain]))
        await self.until(lambda: manager.windows["w"].handle is not None)
        state = controller.on_tool_result_ready("w", "a", self.current())
        self.assertEqual(state.level, 0)
        self.assertEqual(state.messages, self.current())
        self.assertEqual(state.kv_handle.handle_id, "original")
        self.assertEqual(state.request_metadata["headers"]["x-toolslack-kv-handle"], "original")
        self.assertEqual(self.backend.memory_calls, [])
        self.assertEqual(self.backend.kv_calls, [])
        await controller.close()
        self.assertEqual(self.backend.release_calls, ["original"])
        self.assertEqual(self.backend.drain_calls, 1)

    async def test_idle_original_prefix_is_superseded_by_compacted_l1(self):
        manager = IdlePrefixManager(self.backend, enable_tiering=False, clock=self.clock)
        controller = await self.controller(enable_kv=False, idle_prefix_manager=manager)
        self.dispatch(controller)
        await self.until(lambda: manager.windows["w"].handle is not None
                         and controller.windows["w"].result is not None)
        state = controller.on_tool_result_ready("w", "a", self.current())
        self.assertEqual(state.level, 1)
        self.assertIsNone(state.kv_handle)
        await controller.close()
        self.assertEqual(self.backend.release_calls, ["original"])
        self.assertEqual(self.backend.drain_calls, 1)

    async def test_idle_optional_dispatch_error_does_not_block_native_memory(self):
        manager = IdlePrefixManager(self.backend, enable_tiering=False, clock=self.clock)
        def decline(*args, **kwargs):
            raise ValueError("unsupported original-prefix request")
        manager.on_tool_dispatch = decline
        controller = await self.controller(enable_kv=False, idle_prefix_manager=manager)
        self.dispatch(controller)
        await self.until(lambda: controller.windows["w"].result is not None)
        self.assertEqual(controller.on_tool_result_ready("w", "a", self.current()).level, 1)
        self.assertTrue(any(e["event"] == "idle_prefix_optional_declined" for e in controller.events))


if __name__ == "__main__":
    unittest.main()
