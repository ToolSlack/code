"""Real HTTP/owner/queue integration with explicit artificial CPU model replies."""
import asyncio
import json
import time
import unittest
from unittest.mock import patch
from aiohttp import web
from test_terminal_http import Harness
from test_cancellation_http import AbortHarness
import model_proxy as proxy


async def until(condition):
    async with asyncio.timeout(3):
        while not condition(): await asyncio.sleep(.005)


class QueueHarness(Harness):
    async def start(self, **kwargs):
        self.release_blocker=asyncio.Event()
        self.arrivals=[]
        original=proxy.serve
        async def enabled(args):
            args.optional_memory_workers=1
            return await original(args)
        with patch.object(proxy, 'serve', enabled):
            return await super().start(**kwargs)

    async def chat(self, request):
        body=await request.json();rid=body['rid']
        self.arrivals.append((rid,body['priority']))
        self.chat_calls+=1;self.chat_arrived.set()
        if rid=='blocker':await self.release_blocker.wait()
        if body.get('stream'):
            await asyncio.sleep(.08 if rid.startswith('pre-') else .02)
            item=dict(id=rid,model='Qwen3-8B',
                choices=[dict(index=0,finish_reason='stop',delta={'content':'CPU fixture'})],
                usage=dict(prompt_tokens=2,completion_tokens=1,total_tokens=3))
            return web.Response(text='data: '+json.dumps(item)+'\n\ndata: [DONE]\n\n',
                                content_type='text/event-stream')
        return web.json_response(dict(id=rid, model='Qwen3-8B',
            choices=[dict(index=0,finish_reason='stop',message={'role':'assistant','content':'CPU fixture'})],
            usage=dict(prompt_tokens=2,completion_tokens=1,total_tokens=3)))

    def metadata(self, rid, *, budget=10.,cost=1.,session=None):
        return {'x-toolslack-request-id':rid,'x-toolslack-request-kind':'memory',
                'x-toolslack-session-id':session or 'session-'+rid,
                'x-toolslack-arm':'toolslack_on',
                'x-toolslack-cohort-id':'cpu-http:toolslack_on',
                'x-toolslack-maintenance':json.dumps(dict(clock_domain='same-host-wall',
                    deadline_unix_s=time.time()+budget, snapshot_id='prefix-'+rid,
                    remaining_pipeline_cost_s=cost, estimate_source='CPU fixture cost',
                    ttft_gain_query=dict(run_id='cpu-http',
                        model_profile_hash=proxy.sha(self.profile.__dict__),
                        memory_algorithm='CPU.fixture',queue_bucket='c4:q0',
                        cache_state='cold_prefill',pre_input_length_bucket='0-8k')))}

    async def memory(self, rid, *, required=False, **kwargs):
        headers=self.metadata(rid,**kwargs)
        if required:
            headers.pop('x-toolslack-maintenance');headers['x-toolslack-memory-required']='true'
        async with self.client.post(self.url+'/v1/chat/completions',json=self.body(),headers=headers) as r:
            return r.status,await r.json()

    async def lifecycle(self,path,body,status=201):
        async with self.client.post(self.url+path,json=body) as response:
            payload=await response.json()
            if response.status!=status:
                raise AssertionError((response.status,payload))
            return payload

    async def foreground_stream(self,rid,session):
        headers={'x-toolslack-request-id':rid,'x-toolslack-request-kind':'foreground',
                 'x-toolslack-session-id':session,'x-toolslack-arm':'toolslack_on',
                 'x-toolslack-cohort-id':'cpu-http:toolslack_on'}
        async with self.client.post(self.url+'/v1/chat/completions',json=self.body(True),headers=headers) as r:
            payload=await r.text()
            if r.status!=200:
                raise AssertionError((r.status,payload))
            return payload

    async def seed_positive_history(self,session='history-session'):
        profile_hash=proxy.sha(self.profile.__dict__)
        common=dict(session_id=session,run_id='cpu-http',arm='toolslack_on',
                    model_profile_hash=profile_hash,memory_algorithm='CPU.fixture',
                    queue_bucket='c4:q0',cache_state='cold_prefill')
        await self.lifecycle('/toolslack/v1/ttft/pre-requests',dict(common,
            request_id='pre-benefit',context_epoch='epoch-0',snapshot_id='snapshot-0',
            input_length_bucket='0-8k',commit_id=None))
        await self.foreground_stream('pre-benefit',session)
        await self.lifecycle('/toolslack/v1/ttft/commits',dict(common,
            commit_id='commit-benefit',pre_request_id='pre-benefit',
            pre_context_epoch='epoch-0',post_context_epoch='epoch-1',
            post_snapshot_id='snapshot-1',pre_input_length_bucket='0-8k',
            post_input_length_bucket='0-8k',original_scope_tokens=2,
            materialized_scope_tokens=1))
        await self.lifecycle('/toolslack/v1/ttft/post-requests',dict(common,
            request_id='post-benefit',context_epoch='epoch-1',snapshot_id='snapshot-1',
            input_length_bucket='0-8k',commit_id='commit-benefit'))
        await self.foreground_stream('post-benefit',session)

    async def close(self):
        self.release_blocker.set()
        await super().close()


class SharedAdmissionHTTP(unittest.IsolatedAsyncioTestCase):
    async def harness(self):
        h=await QueueHarness().start();self.addAsyncCleanup(h.close);return h

    async def test_unknown_history_is_neutral_and_preserves_foreground_and_required_bypass(self):
        h=await self.harness()
        block=asyncio.create_task(h.memory('blocker'))
        await h.chat_arrived.wait()
        dense=asyncio.create_task(h.memory('dense'))
        sparse=asyncio.create_task(h.memory('compressible'))
        await until(lambda:sum(e['event']=='maintenance_queued' for e in h.events())==3)
        self.assertEqual((await h.request(rid='foreground',handle=False))[0],200)
        self.assertEqual((await h.memory('required',required=True))[0],200)
        self.assertEqual(h.arrivals,[('blocker',0),('foreground',100),('required',100)])
        h.release_blocker.set()
        self.assertEqual([r[0] for r in await asyncio.gather(block,dense,sparse)],[200,200,200])
        self.assertEqual([x[0] for x in h.arrivals][-2:],['dense','compressible'])
        estimates=[e for e in h.events() if e['event']=='ttft_history_estimate']
        self.assertEqual(len(estimates),3)
        self.assertTrue(all(e['history_value'] is None and e['scheduler_value']==0 for e in estimates))
        async with h.client.get(h.url+'/lifecycle/status') as r:
            s=await r.json();self.assertTrue(s['quiescent']);self.assertEqual(s['optional_memory_admission']['running'],[])

    async def test_completed_history_is_read_by_admission_and_orders_higher_gain_first(self):
        h=await self.harness();await h.seed_positive_history()
        block=asyncio.create_task(h.memory('blocker'))
        await until(lambda:any(rid=='blocker' for rid,_ in h.arrivals))
        unknown=asyncio.create_task(h.memory('unknown'))
        benefit=asyncio.create_task(h.memory('benefit',session='history-session'))
        await until(lambda:sum(e['event']=='maintenance_queued' for e in h.events())==3)
        h.release_blocker.set()
        self.assertEqual([r[0] for r in await asyncio.gather(block,unknown,benefit)],[200,200,200])
        memory_arrivals=[rid for rid,_ in h.arrivals if rid in {'blocker','unknown','benefit'}]
        self.assertEqual(memory_arrivals,['blocker','benefit','unknown'])
        estimates=[e for e in h.events() if e['event']=='ttft_history_estimate']
        by_session={e['session_id']:e for e in estimates}
        self.assertGreater(by_session['history-session']['history_value'],.5)
        self.assertEqual(by_session['history-session']['scheduler_value'],
                         by_session['history-session']['history_value'])
        self.assertIsNone(by_session['session-unknown']['history_value'])
        self.assertEqual(by_session['session-unknown']['scheduler_value'],0.)

    async def test_history_completed_after_arrival_does_not_leak_into_queued_request(self):
        h=await self.harness()
        block=asyncio.create_task(h.memory('blocker'))
        await until(lambda:any(rid=='blocker' for rid,_ in h.arrivals))
        early=asyncio.create_task(h.memory('early',session='history-session'))
        await until(lambda:sum(e['event']=='maintenance_queued' for e in h.events())==2)
        await h.seed_positive_history()
        late=asyncio.create_task(h.memory('late',session='history-session'))
        await until(lambda:sum(e['event']=='maintenance_queued' for e in h.events())==3)
        h.release_blocker.set()
        self.assertEqual([r[0] for r in await asyncio.gather(block,early,late)],[200,200,200])
        memory_arrivals=[rid for rid,_ in h.arrivals if rid in {'blocker','early','late'}]
        self.assertEqual(memory_arrivals,['blocker','late','early'])
        estimates=[e for e in h.events()
                   if e['event']=='ttft_history_estimate' and e['session_id']=='history-session']
        self.assertEqual(len(estimates),2)
        self.assertIsNone(estimates[0]['history_value'])
        self.assertEqual(estimates[0]['scheduler_value'],0.)
        self.assertGreater(estimates[1]['history_value'],.5)

    async def test_expired_optional_never_reaches_native_engine(self):
        h=await self.harness();block=asyncio.create_task(h.memory('blocker'));await h.chat_arrived.wait()
        status,body=await h.memory('expires',budget=.4,cost=.1)
        self.assertEqual(status,400);self.assertFalse(body['native_submitted'])
        self.assertEqual(h.arrivals,[('blocker',0)])
        h.release_blocker.set();await block

    async def test_missing_metadata_is_not_silently_fifo(self):
        h=await self.harness()
        async with h.client.post(h.url+'/v1/chat/completions',json=h.body(),
                headers={'x-toolslack-request-kind':'memory'}) as r:
            self.assertEqual(r.status,400)
        self.assertEqual(h.arrivals,[])

    async def test_caller_gain_and_incomplete_history_query_fail_before_native(self):
        h=await self.harness()
        caller=h.metadata('caller-gain')
        caller_spec=json.loads(caller['x-toolslack-maintenance'])
        caller_spec['expected_relative_ttft_gain']=.99
        caller['x-toolslack-maintenance']=json.dumps(caller_spec)
        async with h.client.post(h.url+'/v1/chat/completions',json=h.body(),headers=caller) as r:
            payload=await r.json();self.assertEqual(r.status,400)
            self.assertIn('caller-supplied',str(payload))
        incomplete=h.metadata('incomplete-query')
        incomplete_spec=json.loads(incomplete['x-toolslack-maintenance'])
        del incomplete_spec['ttft_gain_query']['memory_algorithm']
        incomplete['x-toolslack-maintenance']=json.dumps(incomplete_spec)
        async with h.client.post(h.url+'/v1/chat/completions',json=h.body(),headers=incomplete) as r:
            payload=await r.json();self.assertEqual(r.status,400)
            self.assertIn('query identities',str(payload))
        self.assertEqual(h.arrivals,[])

    async def test_disconnected_queued_request_never_reaches_native_engine(self):
        h=await self.harness();block=asyncio.create_task(h.memory('blocker'));await h.chat_arrived.wait()
        reader,writer=await asyncio.open_connection('127.0.0.1',h.proxy_port)
        data=json.dumps(h.body()).encode();headers=h.metadata('disconnected')
        wire=(f'POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {len(data)}\r\n'+''.join(f'{k}: {v}\r\n' for k,v in headers.items())+'\r\n').encode()+data
        writer.write(wire);await writer.drain()
        await until(lambda:any(e['event']=='maintenance_queued' and e['request_id']=='disconnected' for e in h.events()))
        writer.close();await writer.wait_closed()
        await until(lambda:any(e['event']=='maintenance_not_dispatched' and e['request_id']=='disconnected' for e in h.events()))
        h.release_blocker.set();await block
        self.assertEqual(h.arrivals,[('blocker',0)])



class QueuedAbortHarness(AbortHarness):
    async def start(self, **kwargs):
        original=proxy.serve
        async def enabled(args):
            args.optional_memory_workers=1
            return await original(args)
        with patch.object(proxy,'serve',enabled):
            return await super().start(**kwargs)

    metadata=QueueHarness.metadata
    memory=QueueHarness.memory


class NativeTerminalAdmissionHTTP(unittest.IsolatedAsyncioTestCase):
    async def scenario(self, wrong_id=False):
        h=await QueuedAbortHarness(delay_headers=True,wrong_id=wrong_id).start()
        self.addAsyncCleanup(h.close)
        reader,writer=await asyncio.open_connection('127.0.0.1',h.proxy_port)
        data=json.dumps(h.body(stream=True)).encode();headers=h.metadata('cancel-first')
        wire=(f'POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {len(data)}\r\n'+''.join(f'{k}: {v}\r\n' for k,v in headers.items())+'\r\n').encode()+data
        writer.write(wire);await writer.drain();await h.chat_arrived.wait()
        writer.close();await writer.wait_closed();await h.aborted.wait()
        second=asyncio.create_task(h.memory('control-after'))
        await until(lambda:sum(e['event']=='maintenance_queued' for e in h.events())==2)
        self.assertEqual(h.chat_calls,1, 'abort ACK must not release native reservation')
        async with h.client.get(h.url+'/lifecycle/status') as r:
            state=await r.json();self.assertEqual(state['optional_memory_admission']['running'],['cancel-first'])
        h.allow_terminal.set()
        status,body=await second
        if wrong_id:
            self.assertEqual(status,503)
            self.assertEqual(h.chat_calls,1)
            async with h.client.get(h.url+'/lifecycle/status') as r:
                state=await r.json();self.assertTrue(state['guard_cleanup_required'])
                self.assertEqual(state['optional_memory_admission']['running'],['cancel-first'])
        else:
            self.assertEqual(status,200);self.assertEqual(h.chat_calls,2)
            names=[(e['event'],e.get('request_id')) for e in h.events()]
            self.assertLess(names.index(('maintenance_terminal','cancel-first')),names.index(('maintenance_dispatched','control-after')))

    async def test_abort_ack_holds_slot_until_exact_native_terminal(self):
        await self.scenario()

    async def test_wrong_native_identity_keeps_slot_and_rejects_waiter(self):
        await self.scenario(wrong_id=True)


if __name__=='__main__':unittest.main()
