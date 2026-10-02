import asyncio
from copy import deepcopy
import importlib.util
from pathlib import Path
import tempfile
import time
import unittest

from toolslack.backend import BackendProtocolError, BackendUnavailable, SGLangBackend
from toolslack.types import NativeResult, Scope, digest


PROFILE = 'a' * 64
BODY = {'model': 'Qwen3-8B', 'messages': [{'role': 'user', 'content': 'a real closed history'}]}
FUTURE = deepcopy(BODY)
FUTURE['messages'].append({'role': 'assistant', 'content': '', 'tool_calls': [
    {'id': 'tool-1', 'type': 'function', 'function': {'name': 'search', 'arguments': '{}'}}]})


class NativeServer:
    """Protocol adversary for CPU correctness, never a GPU performance result."""
    def __init__(self):
        self.calls = []
        self.cap = True
        self.mode = None
        self.started = asyncio.Event()
        self.finish = None
        self.pending_cancels = 0

    async def __call__(self, method, path, body):
        self.calls.append((method, path, deepcopy(body)))
        if path == '/toolslack/capabilities':
            if not self.cap:
                return 404, {}
            return 200, dict(bounded_prefix_prefill=True, deadline_terminal_drain=True,
                             existing_prefix_registration=True,
                             service_profile_sha256=PROFILE)
        if path == '/exact_tokens':
            ids = list(range(10))
            return 200, dict(input_ids=ids, input_ids_sha256=digest(ids),
                             payload_sha256=digest(body), model_name='Qwen3-8B',
                             kv_service_profile_sha256=PROFILE, model_profile_sha256='b' * 64)
        if path == '/kv/register-existing':
            self.started.set()
            if self.finish is not None:
                await self.finish.wait()
            if self.mode == 'existing_miss':
                return 200, dict(registration={'ok': False, 'error': {'code': 'cache_miss'}},
                                 optional_cache_miss=True, native_generate_submitted=False)
            ids = list(range(5))
            reg = dict(ok=True, action='register', request_id=body['request_id'] + ':register',
                       handle_id='epoch.handle', service_epoch='epoch', service_profile_sha256=PROFILE,
                       requested_prefix_tokens=len(ids), requested_prefix_sha256=digest(ids),
                       cached_prefix_tokens=len(ids), prefix_sha256=digest(ids), device_ready=True)
            if self.mode == 'existing_hash':
                reg['requested_prefix_sha256'] = 'f' * 64
            return 200, dict(registration=reg, existing_prefix_only=True, native_generate_submitted=False,
                             semantic_context_changed=False, optional_cache_miss=False,
                             requested_prefix_tokens=5, requested_prefix_sha256=digest(ids),
                             cached_prefix_tokens=5, prefix_sha256=digest(ids),
                             known_payload_sha256=digest(body['body']), service_profile_sha256=PROFILE,
                             future_consumer_match_required=True, stable_scope={'stable_prefix_tokens': 5})
        if path == '/kv/prefill':
            self.started.set()
            if self.finish is not None:
                await self.finish.wait()
            if self.mode == 'decline':
                return 400, dict(native_submitted=False, error='too little budget')
            if self.mode == 'unknown':
                return 503, dict(native_submitted=True, native_terminal_confirmed=False)
            ids = list(range(min(5, body['max_prefix_tokens'])))
            if self.mode == 'oversize':
                ids = list(range(body['max_prefix_tokens'] + 1))
            reg = dict(ok=True, request_id=body['request_id'] + ':register', action='register',
                       handle_id='epoch.handle', service_epoch='epoch', service_profile_sha256=PROFILE,
                       requested_prefix_tokens=len(ids), requested_prefix_sha256=digest(ids),
                       cached_prefix_tokens=len(ids), prefix_sha256=digest(ids), device_ready=True)
            if self.mode == 'hash':
                reg['prefix_sha256'] = 'f' * 64
            return 200, dict(registration=reg, selected_prefix_ids=ids,
                             requested_prefix_tokens=len(ids), requested_prefix_sha256=digest(ids),
                             prefix_sha256=digest(ids), service_profile_sha256=PROFILE,
                             prefill=dict(meta_info=dict(id=body['request_id'] + ':prefill',
                                 prompt_tokens=len(ids), completion_tokens=0)))
        if path == '/kv/control':
            out = dict(ok=True, request_id=body['request_id'], action=body['action'],
                       handle_id=body['handle_id'], service_epoch='epoch', service_profile_sha256=PROFILE,
                       prefix_sha256=digest(list(range(5))), consumer_refs=0, pending_operations=[])
            if body['action'] == 'cancel':
                if self.pending_cancels:
                    self.pending_cancels -= 1
                    out.update(ok=False, state='CANCEL_PENDING', error={'code': 'handle_busy'})
                else:
                    out['state'] = 'RELEASED'
            else:
                out.update(state='DEVICE_READY', device_ready=True, host_ready=True)
            if self.mode == 'control_epoch':
                out['service_epoch'] = 'other'
            return 200, out
        raise AssertionError(path)


class BackendTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.server = NativeServer()
        self.callback_inputs = []

        async def native(snapshot, scope):
            self.callback_inputs.append((deepcopy(snapshot), scope))
            snapshot['messages'][0]['content'] = 'mutated caller copy'
            return NativeResult([{'role': 'user', 'content': 'official output'}], deepcopy(BODY))

        self.backend = SGLangBackend('http://127.0.0.1:1234', 'Qwen3-8B', PROFILE, native,
                                    transport=self.server, poll_seconds=.001, drain_timeout_s=.05)

    async def prepare(self, **kwargs):
        identity = dict(model_key='Qwen3-8B', body_sha256=digest(BODY), known_future_body=FUTURE)
        identity.update(kwargs.pop('identity', {}))
        return await self.backend.prepare_kv(BODY, kwargs.pop('kv_tokens', 5),
                                            time.monotonic() + 1, identity)

    async def test_native_callback_preserves_original_and_scope(self):
        snapshot = deepcopy(BODY)
        scope = Scope(1, 5, 5, 1., 1., source_sha256=digest(snapshot['messages']))
        result = await self.backend.compact(snapshot, scope)
        self.assertEqual(snapshot, BODY)
        self.assertEqual(result.replacement_messages[0]['content'], 'official output')
        self.assertTrue(result.metadata['native_algorithm_external'])

    async def test_scope_hash_rejected_before_native(self):
        scope = Scope(1, 5, 5, 1., 1., source_sha256='b' * 64)
        with self.assertRaises(ValueError):
            await self.backend.compact(BODY, scope)
        self.assertEqual(self.callback_inputs, [])

    async def test_exact_budget_receipt_and_resident_handle(self):
        handle = await self.prepare()
        self.assertEqual(handle.prefix_sha256, digest(list(range(5))))
        self.assertTrue(self.backend.device_ready(handle))
        self.assertEqual(handle.metadata['requested_kv_tokens'], 5)
        request = next(c[2] for c in self.server.calls if c[1] == '/kv/prefill')
        self.assertEqual(request['max_prefix_tokens'], 5)
        self.assertIn('deadline_unix_ms', request)
        self.assertEqual(request['known_future_body'], FUTURE)
        await self.backend.release(handle)
        self.assertFalse(self.backend.device_ready(handle))

    async def test_old_proxy_declines_without_native_submission(self):
        self.server.cap = False
        with self.assertRaises(BackendUnavailable):
            await self.prepare()
        self.assertFalse(any(c[1] == '/kv/prefill' for c in self.server.calls))

    async def test_unknown_future_declines_before_native(self):
        with self.assertRaises(BackendUnavailable):
            await self.prepare(identity={'known_future_body': None})
        self.assertEqual(self.server.calls, [])

    async def test_unfinished_tool_in_stable_body_declined(self):
        with self.assertRaises(BackendUnavailable):
            await self.backend.prepare_kv(FUTURE, 5, time.monotonic() + 1,
                dict(model_key='Qwen3-8B', body_sha256=digest(FUTURE), known_future_body=FUTURE))
        self.assertEqual(self.server.calls, [])

    async def test_body_mutation_rejected(self):
        with self.assertRaises(ValueError):
            await self.prepare(identity={'body_sha256': 'b' * 64})
        self.assertEqual(self.server.calls, [])

    async def test_oversized_prefix_never_published(self):
        self.server.mode = 'oversize'
        with self.assertRaises(BackendProtocolError):
            await self.prepare()
        self.assertTrue(self.backend.requires_service_cleanup)
        self.assertFalse(next(iter(self.backend._handles.values())).ready)

    async def test_hash_mismatch_never_published(self):
        self.server.mode = 'hash'
        with self.assertRaises(BackendProtocolError):
            await self.prepare()
        self.assertTrue(self.backend.requires_service_cleanup)

    async def test_decline_is_valid_l1_fallback(self):
        self.server.mode = 'decline'
        with self.assertRaises(BackendUnavailable):
            await self.prepare()
        self.assertFalse(self.backend.requires_service_cleanup)
        self.assertEqual(await self.backend.drain(), {'drained': True, 'native_owners': 0, 'handles': 0})

    async def test_unknown_terminal_retains_guard_cleanup_requirement(self):
        self.server.mode = 'unknown'
        with self.assertRaises(BackendProtocolError):
            await self.prepare()
        with self.assertRaises(BackendProtocolError):
            await self.backend.drain()

    async def test_python_cancellation_does_not_claim_native_completion(self):
        self.server.finish = asyncio.Event()
        task = asyncio.create_task(self.prepare())
        await self.server.started.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(len(self.backend._native_owners), 1)
        self.server.finish.set()
        await self.backend.drain()
        self.assertTrue(any(c[2].get('action') == 'cancel' for c in self.server.calls if c[2]))
        self.assertEqual(self.backend._handles, {})

    async def test_release_retries_pending_native_owner(self):
        handle = await self.prepare()
        self.server.pending_cancels = 2
        await self.backend.release(handle)
        self.assertEqual(sum(c[2].get('action') == 'cancel' for c in self.server.calls if c[2]), 3)

    async def test_changed_control_epoch_does_not_release_handle(self):
        handle = await self.prepare()
        self.server.mode = 'control_epoch'
        with self.assertRaises(BackendProtocolError):
            await self.backend.release(handle)
        self.assertIn(handle.handle_id, self.backend._handles)

    async def test_no_calibrated_lead_retains_hbm(self):
        handle = await self.prepare()
        result = await self.backend.place(handle, time.monotonic() + 2)
        self.assertEqual(result['reason'], 'no_measured_transfer_lead')
        self.assertTrue(self.backend.device_ready(handle))

    async def existing(self, future=FUTURE):
        return await self.backend.register_existing(BODY, future, time.monotonic() + 1,
            dict(model_key='Qwen3-8B', body_sha256=digest(BODY)))

    async def test_existing_registration_has_no_native_generation(self):
        handle = await self.existing()
        self.assertEqual(handle.token_count, 5)
        self.assertTrue(self.backend.device_ready(handle))
        self.assertTrue(handle.metadata['existing_prefix_only'])
        self.assertFalse(any(c[1] == '/kv/prefill' for c in self.server.calls))
        self.assertEqual(self.backend.consumer_headers(handle)['x-toolslack-kv-one-shot'], 'true')
        await self.backend.release(handle)

    async def test_existing_cache_miss_is_safe_noop(self):
        self.server.mode = 'existing_miss'
        with self.assertRaises(BackendUnavailable):
            await self.existing()
        self.assertFalse(self.backend.requires_service_cleanup)
        self.assertEqual(self.backend._handles, {})

    async def test_existing_unknown_result_cannot_be_supplied_as_future(self):
        future = deepcopy(FUTURE)
        future['messages'].append({'role': 'tool', 'tool_call_id': 'tool-1', 'content': 'invented result'})
        with self.assertRaises(ValueError):
            await self.existing(future)
        self.assertEqual(self.server.calls, [])

    async def test_existing_native_hash_must_equal_exact_future_prefix(self):
        self.server.mode = 'existing_hash'
        with self.assertRaises(BackendProtocolError):
            await self.existing()
        self.assertTrue(self.backend.requires_service_cleanup)
        self.assertFalse(next(iter(self.backend._handles.values())).ready)

    async def test_existing_registration_python_cancellation_retains_ownership(self):
        self.server.finish = asyncio.Event()
        task = asyncio.create_task(self.existing())
        await self.server.started.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.server.finish.set()
        await self.backend.drain()
        self.assertEqual(self.backend._handles, {})


class PatchTests(unittest.TestCase):
    def test_new_proxy_enforces_explicit_limit_with_existing_budget(self):
        path = Path(__file__).resolve().parents[1] / 'backend/proxy_patch.py'
        spec = importlib.util.spec_from_file_location('toolslack_proxy_patch', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        source = Path(__file__).resolve().parent / 'fixtures/proxy_v15'
        if not source.is_dir():
            self.skipTest('Historical v15 source mirror unavailable')
        original = (source / 'model_proxy.py').read_bytes()
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / 'proxy'
            receipt = module.build(source, target)
            text = (target / 'model_proxy.py').read_text()
            self.assertIn('budget.select(bounded_tokens)', text)
            self.assertIn('ids = ids[:bounded_tokens]', text)
            self.assertIn('selected_prefix_ids=ids', text)
            self.assertIn("app.router.add_get('/toolslack/capabilities'", text)
            self.assertEqual((source / 'model_proxy.py').read_bytes(), original)
            self.assertFalse(receipt['GPU_validation_performed'])


if __name__ == '__main__':
    unittest.main()
