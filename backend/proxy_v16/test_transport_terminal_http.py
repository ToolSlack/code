"""Actual loopback HTTP coverage for foreground transport-terminal recovery."""
import asyncio
import json
import time
import unittest
from unittest.mock import patch

import aiohttp
from aiohttp import web

import model_proxy as proxy
from test_terminal_http import Harness


class LostResponseHarness(Harness):
    def __init__(self, mode):
        self.mode = mode

    async def start(self, **kwargs):
        self.failed_once = False
        self.failed_rid = None
        self.chat_rids = []
        self.active = {}
        self.abort_bodies = []
        self.recovery_order = []
        self.status_calls = 0
        self.after_abort_status_calls = 0
        original = proxy.serve

        async def configured(args):
            args.transport_terminal_wait = .25
            return await original(args)

        with patch.object(proxy, 'serve', configured):
            return await super().start(**kwargs)

    async def chat(self, request):
        body = await request.json()
        rid = body['rid']
        self.chat_calls += 1
        self.chat_rids.append(rid)
        self.chat_arrived.set()
        if not self.failed_once and rid.startswith('lost-'):
            self.failed_once = True
            self.failed_rid = rid
            self.active[rid] = self.mode not in {'terminal_proof','inactive_ambiguous'}
            request.transport.close()
            await asyncio.sleep(0)
            return web.Response(status=200)
        usage = dict(prompt_tokens=2, completion_tokens=1, total_tokens=3)
        return web.json_response(dict(id=rid, model='Qwen3-8B',
            choices=[dict(index=0, finish_reason='stop',
                          message={'role':'assistant','content':'CPU fixture'})],
            usage=usage))

    async def abort(self, request):
        body = await request.json()
        self.abort_bodies.append(body)
        self.recovery_order.append("abort")
        return web.json_response({'ok': True, 'rid': body.get('rid'),
                                  'terminal_proof': False})

    async def control(self, request):
        body = await request.json()
        self.controls.append(body)
        action = body['action']
        result = {key: body[key] for key in
                  ('request_id', 'action', 'service_profile_sha256')}
        result.update(ok=True, service_epoch='fixture-service-epoch')
        if action == 'request_status':
            self.status_calls += 1
            self.recovery_order.append("status")
            rid = body['consumer_request_id']
            if self.mode == 'unavailable':
                return web.json_response({'error':'fixture unavailable'}, status=503)
            if self.mode == 'slow_status':
                await asyncio.sleep(.75)
            active = self.active.get(rid, False)
            if self.mode == 'active_abort' and self.abort_bodies:
                self.after_abort_status_calls += 1
                if self.after_abort_status_calls >= 2:
                    active = False
                    self.active[rid] = False
            result.update(
                consumer_request_id='wrong-rid' if self.mode == 'wrong_identity' else rid,
                active=active,
                engine_epoch='fixture-engine-epoch',
                terminal_proof=self.mode == 'terminal_proof',
            )
            return web.json_response(result)
        if action == 'acquire':
            result['handle_id'] = body['handle_id']
            self.refs.add(body['consumer_request_id'])
            return web.json_response(result)
        if action == 'release_consumer':
            result['handle_id'] = body['handle_id']
            self.refs.discard(body['consumer_request_id'])
            return web.json_response(result)
        raise AssertionError('unexpected fixture control action: '+action)

    async def failed_request(self, rid):
        headers = {'x-toolslack-request-id': rid,
                   'x-toolslack-request-kind': 'foreground',
                   'x-toolslack-kv-handle': 'h'}
        try:
            async with self.client.post(self.url+'/v1/chat/completions',
                                        json=self.body(), headers=headers) as response:
                await response.read()
                return response.status
        except aiohttp.ClientError:
            return None


class TransportTerminalHTTP(unittest.IsolatedAsyncioTestCase):
    async def harness(self, mode):
        value = await LostResponseHarness(mode).start()
        self.addAsyncCleanup(value.close)
        return value

    async def assert_task_failed_but_service_continues(self, h, rid):
        self.assertNotEqual(await h.failed_request(rid), 200)
        self.assertEqual(h.chat_rids.count(rid), 1, 'model request must never be retried')
        self.assertEqual(h.refs, set(), 'confirmed terminal must release its owned KV ref')
        events = h.events()
        self.assertTrue(any(e['event']=='request_failed' and e['request_id']==rid for e in events))
        confirmed = [e for e in events
                     if e['event']=='native_transport_terminal_confirmed'
                     and e['request_id']==rid]
        self.assertEqual(len(confirmed), 1)
        self.assertTrue(confirmed[0]['request_still_failed'])
        self.assertFalse(confirmed[0]['model_request_retried'])
        self.assertFalse(any(e['event']=='kv_runtime_cleanup_required'
                             and e.get('request_id')==rid for e in events))
        self.assertEqual((await h.request(rid='unrelated-next',handle=False))[0], 200)
        self.assertEqual(h.chat_rids.count('unrelated-next'), 1)

    async def test_lost_response_with_exact_engine_terminal_proof_continues(self):
        h = await self.harness('terminal_proof')
        await self.assert_task_failed_but_service_continues(h, 'lost-proven')
        self.assertGreaterEqual(h.status_calls, 1)

    async def test_active_request_is_aborted_then_observed_terminal(self):
        h = await self.harness('active_abort')
        rid = 'lost-active-abort'
        await self.assert_task_failed_but_service_continues(h, rid)
        self.assertTrue(h.abort_bodies)
        self.assertTrue(all(body == {'rid':rid,'abort_all':False}
                            for body in h.abort_bodies))
        self.assertGreaterEqual(h.after_abort_status_calls, 2)
        self.assertEqual(h.recovery_order[0], 'status')
        self.assertIn('abort', h.recovery_order)
        self.assertLess(h.recovery_order.index('status'), h.recovery_order.index('abort'))

    async def assert_unconfirmed_poison(self, mode):
        h = await self.harness(mode)
        rid = 'lost-'+mode
        self.assertNotEqual(await h.failed_request(rid), 200)
        self.assertEqual(h.chat_rids, [rid])
        self.assertIn(rid, h.refs, 'unconfirmed native request must retain its KV ref')
        self.assertTrue(any(e['event']=='native_transport_terminal_unconfirmed'
                            and e['request_id']==rid for e in h.events()))
        self.assertTrue(any(e['event']=='kv_runtime_cleanup_required'
                            and e.get('request_id')==rid for e in h.events()))
        self.assertEqual((await h.request(rid='blocked-next',handle=False))[0], 503)
        self.assertEqual(h.chat_rids, [rid], 'poisoned run must not submit another generation')

    async def test_wrong_status_identity_remains_failed_and_poisoned(self):
        await self.assert_unconfirmed_poison('wrong_identity')

    async def test_unavailable_status_remains_failed_and_poisoned(self):
        await self.assert_unconfirmed_poison('unavailable')

    async def test_initial_inactive_without_terminal_proof_is_ambiguous(self):
        await self.assert_unconfirmed_poison('inactive_ambiguous')

    async def test_slow_status_is_bounded_by_advertised_recovery_budget(self):
        h = await self.harness('slow_status')
        rid = 'lost-slow-status'
        started = time.monotonic()
        self.assertNotEqual(await h.failed_request(rid), 200)
        elapsed = time.monotonic() - started
        self.assertLess(elapsed, .6, 'engine-control timeout must not exceed the .25s recovery budget')
        self.assertTrue(h.abort_bodies, 'the exact native abort is required even after status timeout')
        self.assertEqual(h.abort_bodies[0], {'rid':rid,'abort_all':False})
        self.assertTrue(any(e['event']=='native_transport_terminal_unconfirmed'
                            and e['request_id']==rid for e in h.events()))
        self.assertEqual((await h.request(rid='blocked-after-slow',handle=False))[0], 503)


if __name__ == '__main__':
    unittest.main()
