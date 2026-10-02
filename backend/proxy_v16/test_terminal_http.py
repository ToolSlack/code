"""Actual proxy HTTP; deliberately artificial CPU tokenizer/model/KV fixtures."""
import asyncio
import contextlib
import json
from pathlib import Path
import socket
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import aiohttp
from aiohttp import web
import model_proxy as proxy
from server_profile import ServerProfile


def port():
    with socket.socket() as s:
        s.bind(('127.0.0.1',0))
        return s.getsockname()[1]


class Counter:
    tokenizer=SimpleNamespace(chat_template='artificial_cpu_fixture')
    source_hashes={}
    def token_ids(self,body,add_generation_prompt=True):
        return [1,2] if add_generation_prompt else [1]


class Harness:
    async def start(self,*,status=200,stream='valid',generate_status=200,release='ok',cancel_codes=(),candidate_terminal_wait=5.0):
        self.tmp=tempfile.TemporaryDirectory()
        self.path=Path(self.tmp.name)
        (self.path/'tokenizer.json').write_text('{}')
        (self.path/'service.json').write_text('{"artificial_cpu_fixture":true}')
        self.profile=ServerProfile(model_path=str(self.path),output_reserve=8,request_timeout_seconds=5)
        self.status=status;self.stream=stream;self.generate_status=generate_status;self.release=release
        self.cancel_codes=list(cancel_codes);self.cancel_calls=0
        self.chat_calls=0;self.generate_calls=0;self.controls=[];self.refs=set();self.release_calls=0
        self.chat_arrived=asyncio.Event();self.allow_chat=asyncio.Event();self.allow_chat.set()
        self.acquire_arrived=asyncio.Event();self.allow_acquire=asyncio.Event();self.allow_acquire.set()
        self.up_port=port();self.proxy_port=port()
        app=web.Application();app.router.add_post('/v1/chat/completions',self.chat)
        if hasattr(self, 'abort'):
            app.router.add_post('/abort_request', self.abort)
        app.router.add_post('/generate',self.generate);app.router.add_post('/toolslack/kv/control',self.control)
        self.runner=web.AppRunner(app,access_log=None);await self.runner.setup()
        await web.TCPSite(self.runner,'127.0.0.1',self.up_port).start()
        self.patches=[patch.object(proxy,'load_profile',return_value=self.profile),patch.object(proxy,'make_counter',return_value=Counter())]
        for p in self.patches:p.start()
        self.task=asyncio.create_task(proxy.serve(SimpleNamespace(profile=self.path/'unused.json',
            output=self.path/'out',port=self.proxy_port,upstream=f'http://127.0.0.1:{self.up_port}',
            kv_service_profile=self.path/'service.json',kv_max_prefills=1,
            candidate_terminal_wait=candidate_terminal_wait)))
        self.client=aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=8))
        self.url=f'http://127.0.0.1:{self.proxy_port}'
        for _ in range(100):
            if self.task.done():await self.task
            try:
                async with self.client.get(self.url+'/health') as r:
                    if r.status==200:return self
            except aiohttp.ClientConnectorError:pass
            await asyncio.sleep(.01)
        raise RuntimeError('CPU proxy failed to start')
    async def close(self):
        self.allow_chat.set();self.allow_acquire.set()
        await self.client.close();self.task.cancel()
        with contextlib.suppress(asyncio.CancelledError):await self.task
        await self.runner.cleanup()
        for p in reversed(self.patches):p.stop()
        self.tmp.cleanup()
    def body(self,stream=False):
        return dict(model='Qwen3-8B',messages=[{'role':'user','content':'artificial fixture only'}],max_tokens=8,stream=stream)
    async def request(self,*,rid='r',handle=True,stream=False,body=None,one_shot=False):
        headers={'x-toolslack-request-id':rid,'x-toolslack-request-kind':'foreground'}
        if handle:headers['x-toolslack-kv-handle']='h'
        if one_shot:headers['x-toolslack-kv-one-shot']='true'
        async with self.client.post(self.url+'/v1/chat/completions',json=body if body is not None else self.body(stream),headers=headers) as r:
            text='';failed=False
            try:
                async for chunk in r.content.iter_any():text+=chunk.decode()
            except aiohttp.ClientPayloadError:failed=True
            return r.status,text,failed
    async def prefill(self,rid='build'):
        body=self.body();future=dict(body,messages=body['messages']+[
            {'role':'assistant','content':None,'tool_calls':[{'id':'call-1','type':'function',
             'function':{'name':'read','arguments':'{}'}}]}])
        async with self.client.post(self.url+'/kv/prefill',json={'request_id':rid,'body':body,
            'known_future_body':future}) as r:
            return r.status,await r.json()
    async def control(self,request):
        body=await request.json();self.controls.append(body)
        result={k:body[k] for k in ('request_id','action','service_profile_sha256')}
        result.update(ok=True,handle_id=body.get('handle_id','h'),service_epoch='fixture-epoch')
        action=body['action']
        if action=='acquire':
            self.refs.add(body['consumer_request_id']);self.acquire_arrived.set();await self.allow_acquire.wait()
        if action=='release_consumer':
            self.release_calls+=1
            if self.release=='busy_once' and self.release_calls==1:result.update(ok=False,error={'code':'busy'})
            elif self.release=='consumer_active_once' and self.release_calls==1:
                result.update(ok=False,error={'code':'consumer_active','message':'Native request still active'})
            elif self.release=='error':result.update(ok=False,error={'code':'bad_identity'})
            else:self.refs.discard(body['consumer_request_id'])
        if action=='register':
            result.update(cached_prefix_tokens=1,requested_prefix_tokens=1,requested_prefix_sha256=proxy.sha([1]),prefix_sha256=proxy.sha([1]))
        if action=='cancel':
            self.cancel_calls+=1
            if self.cancel_codes:
                result.update(ok=False,error={'code':self.cancel_codes.pop(0)},
                              state='BUSY',consumer_refs=len(self.refs),pending_operations=['fixture'])
            else:
                result.update(state='RELEASED',consumer_refs=0,pending_operations=[])
        return web.json_response(result)
    async def generate(self,request):
        self.generate_calls+=1;body=await request.json()
        if self.generate_status!=200:return web.json_response({'error':'artificial failure'},status=self.generate_status)
        return web.json_response({'meta_info':{'id':body['rid'],'prompt_tokens':1,'completion_tokens':0}})
    async def chat(self,request):
        self.chat_calls+=1;body=await request.json();self.chat_arrived.set();await self.allow_chat.wait()
        if self.status!=200:return web.json_response({'error':'artificial failure'},status=self.status)
        usage=dict(prompt_tokens=2,completion_tokens=1,total_tokens=3)
        if not body.get('stream'):
            return web.json_response(dict(id=body['rid'],model='Qwen3-8B',choices=[dict(index=0,finish_reason='stop',message={'role':'assistant','content':'fixture'})],usage=usage))
        item=dict(id=body['rid'],model='Qwen3-8B',choices=[dict(index=0,finish_reason='stop',delta={'content':'fixture'})],usage=usage)
        payload='data: '+json.dumps(item)+'\n\n'
        if self.stream=='valid':payload+='data: [DONE]\n\n'
        elif self.stream=='truncated':payload+='data: {"partial"'
        elif self.stream=='bad_usage':
            item['usage']['prompt_tokens']=3;payload='data: '+json.dumps(item)+'\n\ndata: [DONE]\n\n'
        return web.Response(text=payload,content_type='text/event-stream')
    def events(self):
        return [json.loads(s) for s in (self.path/'out/proxy_events.jsonl').read_text().splitlines()]


class TerminalHTTP(unittest.IsolatedAsyncioTestCase):
    async def harness(self,**kwargs):
        h=await Harness().start(**kwargs);self.addAsyncCleanup(h.close);return h
    async def test_all_native_non200_are_unknown_and_poison(self):
        for status in (400,401,403,404,405,408,413,415,422,429,500,502,503):
            with self.subTest(status=status):
                h=await self.harness(status=status)
                got,body,_=await h.request()
                self.assertEqual(got,502);self.assertEqual(json.loads(body)['error']['type'],'unconfirmed_upstream')
                self.assertEqual(h.refs,{'r'});self.assertEqual(h.release_calls,0)
                self.assertEqual((await h.request(rid='second'))[0],503)
                self.assertEqual((await h.prefill())[0],503)
                self.assertEqual(h.chat_calls,1);self.assertEqual(h.generate_calls,0)
                self.assertTrue(any(e.get('upstream_terminal_observed') is False for e in h.events()))
                # Close now so patches for the next subcase cannot stack.
                await h.close();self._cleanups.pop()
    async def test_local_pre_admission400_does_not_poison(self):
        h=await self.harness()
        self.assertEqual((await h.request(body={'model':'wrong'}))[0],400)
        self.assertEqual(h.controls,[]);self.assertEqual(h.chat_calls,0)
        self.assertEqual((await h.request())[0],200);self.assertEqual(h.refs,set())
    async def test_valid_nonstream_releases(self):
        h=await self.harness();self.assertEqual((await h.request())[0],200)
        self.assertEqual(h.refs,set());self.assertEqual(h.release_calls,1)
    async def test_one_shot_handle_is_released_after_terminal_consumer(self):
        h=await self.harness();self.assertEqual((await h.request(one_shot=True))[0],200)
        actions=[c['action'] for c in h.controls]
        self.assertEqual(actions,['acquire','release_consumer','cancel'])
        self.assertTrue(any(e['event']=='kv_one_shot_released' for e in h.events()))
        # A framework-side drain after proxy-owned cleanup is idempotent and
        # must not issue another native cancel against a released handle.
        async with h.client.post(h.url+'/kv/control',json={
            'request_id':'client-drain','action':'cancel','handle_id':'h'}) as response:
            receipt=await response.json();self.assertEqual(response.status,200)
        self.assertTrue(receipt['idempotent']);self.assertEqual(receipt['state'],'RELEASED')
        self.assertEqual([c['action'] for c in h.controls],actions)

        # Retiring this logical handle does not alter ownership of a distinct
        # handle which may reference the same natural radix-cache pages.
        self.assertEqual((await h.request(rid='ordinary-shared',one_shot=False))[0],200)
        later=[c['action'] for c in h.controls]
        self.assertEqual(later[-2:],['acquire','release_consumer'])
        self.assertEqual(later.count('cancel'),1)

    async def test_one_shot_idempotent_drain_validates_identity_and_profile(self):
        h=await self.harness();self.assertEqual((await h.request(one_shot=True))[0],200)
        native_controls=list(h.controls)
        async with h.client.post(h.url+'/kv/control',json={
            'action':'cancel','handle_id':'h'}) as response:
            self.assertEqual(response.status,400)
            self.assertIn('unique KV operation request ID',str(await response.json()))
        async with h.client.post(h.url+'/kv/control',json={
            'request_id':'wrong-profile','action':'cancel','handle_id':'h',
            'service_profile_sha256':'not-the-bound-profile'}) as response:
            self.assertEqual(response.status,400)
            self.assertIn('profile identity differs',str(await response.json()))
        # Neither malformed drain may trigger another native operation or
        # consume the cached terminal proof.
        self.assertEqual(h.controls,native_controls)
        async with h.client.post(h.url+'/kv/control',json={
            'request_id':'valid-drain','action':'cancel','handle_id':'h'}) as response:
            receipt=await response.json();self.assertEqual(response.status,200)
        self.assertTrue(receipt['idempotent'])
        self.assertEqual(receipt['service_profile_sha256'],proxy.hashlib.sha256(
            (h.path/'service.json').read_bytes()).hexdigest())
        self.assertEqual(h.controls,native_controls)

    async def test_client_drain_cannot_race_proxy_owned_one_shot_cleanup(self):
        h=await self.harness();h.allow_chat.clear()
        consuming=asyncio.create_task(h.request(one_shot=True))
        await asyncio.wait_for(h.chat_arrived.wait(),2)
        async with h.client.post(h.url+'/kv/control',json={
            'request_id':'early-client-drain','action':'cancel','handle_id':'h'}) as response:
            pending=await response.json();self.assertEqual(response.status,200)
        self.assertFalse(pending['ok']);self.assertEqual(pending['state'],'DRAINING')
        self.assertEqual(pending['error']['code'],'handle_busy')
        self.assertFalse(pending['native_cancel_issued'])
        self.assertEqual([c['action'] for c in h.controls],['acquire'])
        h.allow_chat.set();self.assertEqual((await consuming)[0],200)
        self.assertEqual([c['action'] for c in h.controls],
                         ['acquire','release_consumer','cancel'])
        async with h.client.post(h.url+'/kv/control',json={
            'request_id':'late-client-drain','action':'cancel','handle_id':'h'}) as response:
            done=await response.json();self.assertEqual(response.status,200)
        self.assertTrue(done['ok']);self.assertEqual(done['state'],'RELEASED')
        self.assertTrue(done['idempotent'])
        self.assertEqual([c['action'] for c in h.controls],
                         ['acquire','release_consumer','cancel'])

    async def test_sparse_manager_busy_codes_are_bounded_retryable(self):
        for code in ('busy','handle_busy','transfer_busy'):
            with self.subTest(code=code):
                h=await self.harness(cancel_codes=[code])
                self.assertEqual((await h.request(one_shot=True))[0],200)
                cancels=[c for c in h.controls if c['action']=='cancel']
                self.assertEqual(len(cancels),2)
                self.assertEqual(len({c['request_id'] for c in cancels}),2)
                self.assertEqual(h.cancel_calls,2)
    async def test_valid_sse_releases_only_after_terminal(self):
        h=await self.harness();status,text,failed=await h.request(stream=True)
        self.assertEqual(status,200);self.assertIn('data: [DONE]',text);self.assertFalse(failed)
        # HTTP response EOF may be visible before the handler's final cleanup.
        for _ in range(100):
            if not h.refs:break
            await asyncio.sleep(.01)
        self.assertEqual(h.refs,set());self.assertEqual(h.release_calls,1)
    async def test_missing_done_retains_and_poison(self):
        h=await self.harness(stream='missing_done');_,text,_=await h.request(stream=True)
        self.assertNotIn('[DONE]',text);self.assertEqual(h.refs,{'r'});self.assertEqual(h.release_calls,0)
        self.assertEqual((await h.request(rid='after'))[0],503)
    async def test_truncated_sse_retains_and_poison(self):
        h=await self.harness(stream='truncated');_,text,_=await h.request(stream=True)
        self.assertNotIn('[DONE]',text);self.assertEqual(h.refs,{'r'})
        self.assertEqual((await h.request(rid='after'))[0],503)
    async def test_bad_usage_even_with_done_is_unknown(self):
        h=await self.harness(stream='bad_usage');_,text,_=await h.request(stream=True)
        self.assertNotIn('[DONE]',text);self.assertEqual(h.refs,{'r'})
        self.assertEqual((await h.request(rid='after'))[0],503)
    async def test_native_prefill_non200_is_unknown(self):
        for status in (400,408,500,503):
            with self.subTest(status=status):
                h=await self.harness(generate_status=status)
                self.assertEqual((await h.prefill())[0],502);self.assertEqual(h.controls,[])
                self.assertEqual((await h.prefill('again'))[0],503)
                self.assertEqual((await h.request())[0],503);self.assertEqual(h.generate_calls,1)
                await h.close();self._cleanups.pop()
    async def test_valid_prefill_registers(self):
        h=await self.harness();self.assertEqual((await h.prefill())[0],200)
        self.assertEqual([c['action'] for c in h.controls],['register'])
    async def test_busy_release_retries_with_new_control_identity(self):
        h=await self.harness(release='busy_once');self.assertEqual((await h.request())[0],200)
        self.assertEqual(h.refs,set());self.assertEqual(h.release_calls,2)
        ids=[c['request_id'] for c in h.controls if c['action']=='release_consumer']
        self.assertEqual(len(set(ids)),2);self.assertEqual(h.chat_calls,1)
    async def test_proven_terminal_consumer_active_release_retries_with_new_identity(self):
        h=await self.harness(release='consumer_active_once')
        self.assertEqual((await h.request(one_shot=True))[0],200)
        self.assertEqual(h.refs,set());self.assertEqual(h.release_calls,2)
        releases=[c for c in h.controls if c['action']=='release_consumer']
        self.assertEqual(len(releases),2)
        self.assertEqual(len({c['request_id'] for c in releases}),2)
        self.assertEqual([c['action'] for c in h.controls][-1],'cancel')
    async def test_nonbusy_release_failure_poison_keeps_ref(self):
        h=await self.harness(release='error');self.assertEqual((await h.request())[0],500)
        self.assertEqual(h.refs,{'r'});self.assertEqual(h.release_calls,1)
        self.assertEqual((await h.request(rid='after'))[0],503)
    async def test_poison_blocks_model_waiting_on_acquire(self):
        h=await self.harness(status=500);h.allow_acquire.clear()
        waiting=asyncio.create_task(h.request(rid='waiting'))
        await asyncio.wait_for(h.acquire_arrived.wait(),2)
        self.assertEqual((await h.request(rid='failure',handle=False))[0],502)
        h.allow_acquire.set();self.assertEqual((await waiting)[0],500)
        self.assertEqual(h.chat_calls,1);self.assertEqual(h.refs,{'waiting'})
    async def test_poison_allows_only_control_cleanup(self):
        h=await self.harness(status=500);await h.request()
        async with h.client.post(h.url+'/kv/control',json={'request_id':'no','action':'prefetch','handle_id':'h'}) as r:self.assertEqual(r.status,503)
        async with h.client.post(h.url+'/kv/control',json={'request_id':'yes','action':'status','handle_id':'h'}) as r:self.assertEqual(r.status,200)
        self.assertNotIn('prefetch',[c['action'] for c in h.controls])

if __name__=='__main__':unittest.main()
