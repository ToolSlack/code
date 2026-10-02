"""Real loopback transport; artificial model fixtures, never GPU evidence."""
import asyncio
import json
import unittest
from unittest.mock import patch

from aiohttp import web

from cancellation_owner import CancellationOwner, shield_owned
from test_terminal_http import Harness
import model_proxy as proxy


class AbortHarness(Harness):
    def __init__(self, *, delay_headers=False, ignore_first=False, wrong_id=False):
        self.delay_headers = delay_headers
        self.ignore_first = ignore_first
        self.wrong_id = wrong_id
        self.aborts = []
        self.aborted = asyncio.Event()
        self.allow_terminal = asyncio.Event()
        self.first_written = asyncio.Event()
        self.native_terminal = asyncio.Event()
        self.native_disconnect = False

    async def abort(self, request):
        body = await request.json()
        self.aborts.append(body)
        if not self.ignore_first or len(self.aborts) > 1:
            self.aborted.set()
        return web.Response(status=200)

    async def chat(self, request):
        body = await request.json()
        self.chat_calls += 1
        self.chat_arrived.set()
        if body['rid'].startswith('control'):
            usage = dict(prompt_tokens=2, completion_tokens=1, total_tokens=3)
            return web.json_response(dict(id=body['rid'], model='Qwen3-8B',
                choices=[dict(index=0, finish_reason='stop', message={'content': 'ok'})], usage=usage))
        response = web.StreamResponse(headers={'Content-Type': 'text/event-stream'})
        try:
            if self.delay_headers:
                await self.aborted.wait()
            await response.prepare(request)
            first = dict(id=body['rid'], model='Qwen3-8B',
                         choices=[dict(index=0, delta={'content': 'first'})])
            await response.write(('data: '+json.dumps(first)+'\n\n').encode())
            self.first_written.set()
            await self.aborted.wait()
            await self.allow_terminal.wait()
            terminal = dict(id='unrelated' if self.wrong_id else body['rid'], model='Qwen3-8B',
                choices=[dict(index=0, delta={}, finish_reason='abort')],
                usage=dict(prompt_tokens=2, completion_tokens=1, total_tokens=3))
            await response.write(('data: '+json.dumps(terminal)+'\n\ndata: [DONE]\n\n').encode())
            await response.write_eof()
            self.native_terminal.set()
        except ConnectionError:
            self.native_disconnect = True
        return response

    async def close(self):
        self.aborted.set()
        self.allow_terminal.set()
        await super().close()


async def until(predicate, timeout=3):
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(.01)


class CancelHTTP(unittest.IsolatedAsyncioTestCase):
    async def harness(self, **kwargs):
        h = await AbortHarness(**kwargs).start()
        self.addAsyncCleanup(h.close)
        return h

    async def disconnect(self, h, rid='cancelled'):
        # Closing a raw TCP socket exercises the web server's real disconnect
        # path even when native prefill has not produced HTTP headers yet.
        reader, writer = await asyncio.open_connection('127.0.0.1', h.proxy_port)
        data = json.dumps(h.body(stream=True)).encode()
        headers = (f'POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\n'
                   f'Content-Type: application/json\r\nContent-Length: {len(data)}\r\n'
                   f'x-toolslack-request-id: {rid}\r\nx-toolslack-request-kind: memory\r\n'
                   'x-toolslack-kv-handle: h\r\n\r\n')
        writer.write(headers.encode() + data)
        await writer.drain()
        await asyncio.wait_for(h.chat_arrived.wait(), 2)
        if not h.delay_headers:
            await asyncio.wait_for(h.first_written.wait(), 2)
            await reader.read(1)
        writer.close()
        await writer.wait_closed()

    async def check_cancel(self, **kwargs):
        h = await self.harness(**kwargs)
        await self.disconnect(h)
        await asyncio.wait_for(h.aborted.wait(), 2)
        self.assertEqual(h.refs, {'cancelled'})
        self.assertEqual(h.release_calls, 0, 'abort HTTP 200 is not terminal')
        async with h.client.get(h.url+'/lifecycle/status') as response:
            state = await response.json()
            self.assertFalse(state['quiescent'])
            self.assertEqual(state['active_owners'], 1)
        self.assertFalse(h.native_disconnect, 'proxy prematurely closed upstream')
        self.assertTrue(all(x == {'rid': 'cancelled', 'abort_all': False} for x in h.aborts))
        h.allow_terminal.set()
        await until(lambda: not h.refs)
        self.assertTrue(h.native_terminal.is_set())
        self.assertFalse(h.native_disconnect)
        self.assertTrue(any(x['event'] == 'native_cancel_terminal' for x in h.events()))
        self.assertFalse(any(x['event'] == 'kv_runtime_cleanup_required' for x in h.events()))
        async with h.client.get(h.url+'/lifecycle/status') as response:
            self.assertTrue((await response.json())['quiescent'])
        self.assertEqual((await h.request(rid='control-after', handle=False))[0], 200)
        return h

    async def test_disconnect_during_decode_drains_terminal_before_release(self):
        await self.check_cancel()

    async def test_disconnect_before_upstream_headers_is_detected(self):
        await self.check_cancel(delay_headers=True)

    async def test_abort_before_registration_is_retried(self):
        h = await self.check_cancel(delay_headers=True, ignore_first=True)
        self.assertGreaterEqual(len(h.aborts), 2)

    async def test_wrong_terminal_identity_keeps_ref_and_poison(self):
        h = await self.harness(wrong_id=True)
        await self.disconnect(h)
        await asyncio.wait_for(h.aborted.wait(), 2)
        h.allow_terminal.set()
        await until(lambda: any(x['event'] == 'kv_runtime_cleanup_required' for x in h.events()))
        self.assertEqual(h.refs, {'cancelled'})
        self.assertEqual(h.release_calls, 0)
        self.assertEqual((await h.request(rid='control-after', handle=False))[0], 503)
        async with h.client.get(h.url+'/lifecycle/status') as response:
            self.assertFalse((await response.json())['quiescent'])

    async def test_duplicate_id_rejects_without_cancelling_owner(self):
        h = await self.harness(delay_headers=True)
        first = asyncio.create_task(h.request(rid='same', stream=True))
        await asyncio.wait_for(h.chat_arrived.wait(), 2)
        self.assertEqual((await h.request(rid='same', handle=False))[0], 409)
        self.assertEqual(h.aborts, [])
        h.aborted.set()
        h.allow_terminal.set()
        self.assertEqual((await first)[0], 200)
        await until(lambda: not h.refs)

    async def test_completed_native_id_cannot_be_reused(self):
        h = await self.check_cancel()
        self.assertEqual((await h.request(rid='cancelled', handle=False))[0], 409)

    async def test_abort_ack_without_terminal_fails_closed(self):
        h = await self.harness()
        def short_owner(*args, **kwargs):
            return CancellationOwner(*args, **kwargs, timeout=.3)
        with patch.object(proxy, 'CancellationOwner', side_effect=short_owner):
            await self.disconnect(h)
        await until(lambda: any(x['event'] == 'native_abort_unconfirmed' for x in h.events()))
        await until(lambda: any(x['event'] == 'kv_runtime_cleanup_required' for x in h.events()))
        self.assertEqual(h.refs, {'cancelled'})
        self.assertEqual(h.release_calls, 0)
        self.assertEqual((await h.request(rid='control-after', handle=False))[0], 503)

    async def test_handler_cancellation_does_not_cancel_worker(self):
        ended = asyncio.Event()
        proceed = asyncio.Event()
        events = []

        class Transport:
            def is_closing(self): return False

        class Request:
            transport = Transport()

        owner = CancellationOwner(Request(), None, 'http://unused',
                                  {'request_id': 'owned'}, lambda name, **kw: events.append(name))
        async def work():
            await proceed.wait()
            ended.set()
        worker = asyncio.create_task(work())
        handler = asyncio.create_task(shield_owned(worker, owner))
        await asyncio.sleep(0)
        handler.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await handler
        self.assertFalse(worker.done())
        self.assertIsNotNone(owner.cancelled_at)
        proceed.set()
        await worker
        self.assertTrue(ended.is_set())


if __name__ == '__main__':
    unittest.main()
