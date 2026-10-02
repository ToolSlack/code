"""CPU-only budget protocol tests, including the actual HTTP proxy path."""
import asyncio
import copy
import hashlib
import json
import time
import unittest
from unittest.mock import patch
from aiohttp import web
import model_proxy as proxy
from prefix_budget import PrefixBudget, PrefixCost, BudgetDeclined
from test_terminal_http import Harness


def fixture(service='fixture'):
    return dict(schema='toolslack.native-prefix-prefill-cost.v1',
                service_profile_sha256=service, extrapolation_allowed=False,
                knots=[dict(tokens=n, monotone_upper_ms=ms, sample_count=1,
                            samples_ms=[ms]) for n, ms in ((2, 20), (4, 1000), (8, 2000))])


class BudgetUnit(unittest.TestCase):
    def test_partial_exact_and_never_extrapolate(self):
        p=PrefixCost(fixture(), 'fixture')
        b=PrefixBudget(dict(deadline_unix_ms=1001500,safety_ms=50), p, 1000, 10)
        self.assertEqual(b.select(8,10)['selected_prefix_tokens'],4)
        self.assertEqual(b.select(3,10)['selected_prefix_tokens'],2)
        self.assertEqual(b.select(100000,10)['selected_prefix_tokens'],4)
        with self.assertRaises(BudgetDeclined):b.select(8,11.48)

    def test_binding_and_invalid_profiles(self):
        with self.assertRaises(ValueError):PrefixCost(fixture(), 'wrong')
        for mutate in (lambda x:x['knots'].reverse(),
                       lambda x:x['knots'][1].update(monotone_upper_ms=1),
                       lambda x:x.update(knots=[]),
                       lambda x:x.update(extrapolation_allowed=True)):
            x=fixture();mutate(x)
            with self.assertRaises(ValueError):PrefixCost(x,'fixture')
        for value in (float('nan'),float('inf'),True,-1):
            with self.assertRaises(ValueError):PrefixBudget(dict(deadline_unix_ms=value),PrefixCost(fixture(),'fixture'))
        with self.assertRaises(BudgetDeclined):PrefixBudget(dict(deadline_unix_ms=1000),None)


class ExactCounter:
    tokenizer=type('Tokenizer',(),{'chat_template':'artificial_cpu_fixture'})()
    source_hashes={}
    def token_ids(self,body,add_generation_prompt=True):
        return list(range(9 if add_generation_prompt else 8))


class BudgetHarness(Harness):
    async def start(self, **kwargs):
        self.generate_arrived=asyncio.Event()
        self.allow_generate=asyncio.Event();self.allow_generate.set()
        self.native_payloads=[]
        serve=proxy.serve
        make_counter=proxy.make_counter
        async def configured(args):
            path=args.output.parent/'cost.json'
            service=hashlib.sha256(args.kv_service_profile.read_bytes()).hexdigest()
            path.write_text(json.dumps(fixture(service)))
            args.kv_cost_profile=path
            with patch.object(proxy,'make_counter',return_value=ExactCounter()):
                return await serve(args)
        with patch.object(proxy,'serve',side_effect=configured):
            return await super().start(**kwargs)

    async def close(self):
        self.allow_generate.set()
        await super().close()

    async def generate(self,request):
        self.generate_calls+=1
        body=await request.json();self.native_payloads.append(body)
        self.generate_arrived.set();await self.allow_generate.wait()
        return web.json_response({'meta_info':{'id':body['rid'],
            'prompt_tokens':len(body['input_ids']),'completion_tokens':0}})

    async def control(self,request):
        body=await request.json();self.controls.append(body)
        ids=body.get('input_ids',[])
        return web.json_response(dict(ok=True,action=body['action'],request_id=body['request_id'],
            service_profile_sha256=body['service_profile_sha256'],handle_id='fixture',
            cached_prefix_tokens=len(ids),requested_prefix_tokens=len(ids),
            requested_prefix_sha256=proxy.sha(ids),prefix_sha256=proxy.sha(ids)))

    def specification(self, rid, remaining_ms):
        body=self.body();future=dict(body,messages=body['messages']+[
            dict(role='assistant',content=None,tool_calls=[dict(id='call',type='function',
                function=dict(name='read',arguments='{}'))])])
        return dict(request_id=rid,body=body,known_future_body=future,
                    deadline_unix_ms=time.time()*1000+remaining_ms,safety_ms=10)

    async def send(self,spec):
        async with self.client.post(self.url+'/kv/prefill',json=spec) as response:
            return response.status,await response.json()


class BudgetHTTP(unittest.IsolatedAsyncioTestCase):
    async def harness(self):
        h=await BudgetHarness().start();self.addAsyncCleanup(h.close);return h

    async def test_partial_preserves_future_and_registration_identity(self):
        h=await self.harness();spec=h.specification('partial',700);original=copy.deepcopy(spec)
        status,body=await h.send(spec)
        self.assertEqual(status,200);self.assertEqual(body['requested_prefix_tokens'],2)
        self.assertEqual(body['cached_prefix_tokens'],2)
        self.assertEqual(body['prefix_sha256'],proxy.sha([0,1]))
        self.assertEqual(spec,original)
        self.assertEqual(body['budget_selection']['full_stable_prefix_tokens'],8)
        self.assertTrue(body['budget_selection']['whole_future_request_preserved'])
        self.assertEqual(h.native_payloads[0]['input_ids'],[0,1])

    async def test_full_when_budget_fits_and_expired_does_not_poison(self):
        h=await self.harness()
        status,body=await h.send(h.specification('expired',-1))
        self.assertEqual(status,400);self.assertFalse(body['native_submitted'])
        self.assertEqual(h.generate_calls,0)
        status,body=await h.send(h.specification('full',4000))
        self.assertEqual(status,200);self.assertEqual(body['cached_prefix_tokens'],8)
        self.assertFalse(body['completed_after_deadline'])

    async def test_queue_wait_consumes_budget_and_no_second_native_work(self):
        h=await self.harness();h.allow_generate.clear()
        first=asyncio.create_task(h.send(h.specification('first',4000)))
        await asyncio.wait_for(h.generate_arrived.wait(),2)
        second=asyncio.create_task(h.send(h.specification('queued',200)))
        await asyncio.sleep(.3);h.allow_generate.set()
        self.assertEqual((await first)[0],200)
        status,body=await second
        self.assertEqual(status,400);self.assertTrue(body['optional_declined'])
        self.assertEqual(h.generate_calls,1)
        self.assertEqual((await h.send(h.specification('after',4000)))[0],200)

    async def test_serialization_cost_consumes_budget(self):
        h=await self.harness()
        original=ExactCounter.token_ids
        def slow(self,*args,**kwargs):
            time.sleep(.1)
            return original(self,*args,**kwargs)
        with patch.object(ExactCounter,'token_ids',slow):
            status,body=await h.send(h.specification('count-expired',200))
        self.assertEqual(status,400);self.assertFalse(body['native_submitted'])
        self.assertEqual(h.generate_calls,0)


if __name__=='__main__':unittest.main()
