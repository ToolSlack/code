"""Actual HTTP existing-cache protocol, all model/control fixtures CPU-only."""
import unittest
from aiohttp import web
from test_prefix_budget import BudgetHarness


class ExistingHarness(BudgetHarness):
    miss=False
    async def control(self, request):
        if not self.miss:return await super().control(request)
        body=await request.json();self.controls.append(body)
        return web.json_response(dict(action=body['action'],request_id=body['request_id'],
            service_profile_sha256=body['service_profile_sha256'],ok=False,
            error={'code':'prefix_not_cached'},cached_prefix_tokens=0))


class ExistingHTTP(unittest.IsolatedAsyncioTestCase):
    async def harness(self):
        h=await ExistingHarness().start();self.addAsyncCleanup(h.close);return h

    async def send(self,h,spec):
        async with h.client.post(h.url+'/kv/register-existing',json=spec) as response:
            return response.status,await response.json()

    async def test_existing_prefix_registration_never_launches_prefill(self):
        h=await self.harness();spec=h.specification('existing',1000)
        code,result=await self.send(h,spec)
        self.assertEqual(code,200);self.assertTrue(result['existing_prefix_only'])
        self.assertFalse(result['semantic_context_changed'])
        self.assertFalse(result['native_generate_submitted'])
        self.assertFalse(result['optional_cache_miss'])
        self.assertEqual(result['cached_prefix_tokens'],8)
        self.assertEqual(h.generate_calls,0);self.assertEqual(h.chat_calls,0)
        self.assertEqual([c['action'] for c in h.controls],['register'])

    async def test_cache_miss_remains_optional_and_does_not_generate(self):
        h=await self.harness();h.miss=True
        code,result=await self.send(h,h.specification('missing',1000))
        self.assertEqual(code,200);self.assertTrue(result['optional_cache_miss'])
        self.assertEqual(h.generate_calls,0)
        h.miss=False
        self.assertEqual((await self.send(h,h.specification('later',1000)))[0],200)

    async def test_future_prefix_rewrite_rejected_before_native_control(self):
        h=await self.harness();spec=h.specification('mutated',1000)
        spec['known_future_body']['messages'][0]={'role':'user','content':'different context'}
        code,result=await self.send(h,spec)
        self.assertEqual(code,400);self.assertFalse(result['native_submitted'])
        self.assertEqual(h.controls,[]);self.assertEqual(h.generate_calls,0)


if __name__=='__main__':unittest.main()
