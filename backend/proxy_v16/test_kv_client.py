import asyncio
import time
import unittest

from kv_client import PrefixLifecycle


class Reply:
    def __init__(self, data): self.data, self.status = data, 200
    async def __aenter__(self): return self
    async def __aexit__(self, *args): return False
    async def json(self): return self.data


class ProtocolDouble:
    """CPU protocol test only: these receipts are never GPU evidence."""
    def __init__(self):
        self.calls=[]
        self.state='DEVICE_READY'
        self.host=False
        self.device=True
        self.offloaded=asyncio.Event()
    def post(self, url, json):
        action=json.get('action','build'); self.calls.append(action)
        if action=='build':
            return Reply({'registration': self.receipt(), 'service_profile_sha256':'cpu-profile',
                          'prefix_sha256':'cpu-prefix', 'requested_prefix_sha256':'cpu-prefix-full',
                          'cached_prefix_tokens':3,'requested_prefix_tokens':4})
        if action=='backup': self.host=True
        if action=='offload':
            self.device=False; self.state='OFFLOADED'; self.offloaded.set()
        if action=='prefetch': self.device=True; self.state='DEVICE_READY'
        if action=='cancel': self.state='RELEASED'; self.device=False; self.host=False
        return Reply(dict(self.receipt(),request_id=json['request_id'],action=action))
    def receipt(self):
        return dict(ok=True,handle_id='cpu-protocol-double',host_ready=self.host,
                    device_ready=self.device,state=self.state,pending_operations=[],consumer_refs=0,
                    service_epoch='cpu-epoch',service_profile_sha256='cpu-profile',prefix_sha256='cpu-prefix',
                    requested_prefix_sha256='cpu-prefix-full',cached_prefix_tokens=3,requested_prefix_tokens=4)


class ClientTests(unittest.IsolatedAsyncioTestCase):
    def lifecycle(self, session):
        return PrefixLifecycle(session,'http://unused',request_prefix='test',
            emit=lambda *a,**k:None,preload_lead_seconds=.01,poll_seconds=.001,timeout_seconds=1)

    async def test_foreground_readiness_wakes_background_timer(self):
        fake=ProtocolDouble(); lc=self.lifecycle(fake)
        task=lc.start({'messages':[]},predicted_todo_end=time.monotonic()+100)
        await asyncio.wait_for(fake.offloaded.wait(),.5)
        self.assertFalse(task.done())
        lc.notify_consumer_ready()
        result=await asyncio.wait_for(task,.5)
        self.assertTrue(result['ok'])
        self.assertEqual(fake.calls[:3],['build','backup','status'])
        self.assertLess(fake.calls.index('offload'),fake.calls.index('prefetch'))
        await lc.retire_and_drain()
        self.assertEqual(fake.calls[-1],'cancel')

    async def test_short_remaining_window_does_not_force_transfer(self):
        fake=ProtocolDouble(); lc=self.lifecycle(fake)
        result=await lc.start({},predicted_todo_end=time.monotonic())
        self.assertTrue(result['offload_skipped'])
        self.assertEqual(fake.calls,['build'])
        await lc.retire_and_drain()

    async def test_retirement_does_not_run_unneeded_prefetch(self):
        fake=ProtocolDouble(); lc=self.lifecycle(fake)
        lc.start({},predicted_todo_end=time.monotonic()+100)
        await asyncio.wait_for(fake.offloaded.wait(),.5)
        await asyncio.wait_for(lc.retire_and_drain(),.5)
        self.assertNotIn('prefetch',fake.calls)
        self.assertEqual(fake.calls[-1],'cancel')

    async def test_registration_failure_is_not_a_ready_handle(self):
        class Failed(ProtocolDouble):
            def post(self,url,json):
                return Reply({'registration':{'ok':False,'error':'cache_miss'}})
        lc=self.lifecycle(Failed())
        result=await lc.start({},predicted_todo_end=time.monotonic()+10)
        self.assertFalse(result['ok'])
        self.assertIsNone(lc.handle_id)
        self.assertTrue((await lc.retire_and_drain())['no_registered_handle'])

    async def test_lost_build_reply_cannot_claim_successful_cleanup(self):
        class Lost(ProtocolDouble):
            def post(self,url,json): raise ConnectionError('No completion receipt')
        lc=self.lifecycle(Lost())
        result=await lc.start({},predicted_todo_end=time.monotonic()+10)
        self.assertFalse(result['ok'])
        with self.assertRaisesRegex(RuntimeError,'guard must close'):
            await lc.retire_and_drain()

    async def test_empty_pending_with_active_consumer_is_not_released(self):
        class Active(ProtocolDouble):
            def post(self,url,json):
                reply=super().post(url,json)
                if json.get('action')=='cancel':
                    reply.data.update(state='CANCEL_PENDING',consumer_refs=1)
                return reply
        lc=self.lifecycle(Active());lc.timeout=.01
        await lc.start({},predicted_todo_end=time.monotonic())
        with self.assertRaisesRegex(TimeoutError,'cleanup was not acknowledged'):
            await lc.retire_and_drain()


if __name__=='__main__': unittest.main()
