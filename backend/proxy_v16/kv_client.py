"""TODO-timed client for the real engine KV protocol.

No tensors or text are represented as fake GPU residency here. Every state and
released-token count comes from the service. A timer delays only the background
prefetch; an actual foreground-ready event can wake it early. Failures retain
the original/new text in the application, and never claim a ready KV prefix.
"""
import asyncio
import time


class KVProtocolError(RuntimeError):
    pass


class PrefixLifecycle:
    def __init__(self, client, endpoint, *, request_prefix, emit, preload_lead_seconds,
                 minimum_offloaded_seconds=0.0, poll_seconds=.02, timeout_seconds=120):
        if preload_lead_seconds <= 0 or minimum_offloaded_seconds < 0:
            raise ValueError('Transfer lead must be a positive calibrated estimate')
        self.client, self.endpoint = client, endpoint.rstrip('/')
        self.request_prefix, self.emit = request_prefix, emit
        self.preload_lead = preload_lead_seconds
        self.minimum_offloaded = minimum_offloaded_seconds
        self.poll_seconds, self.timeout = poll_seconds, timeout_seconds
        self.handle_id = None
        self.receipt = None
        self.consumer_ready = asyncio.Event()
        self.retired = asyncio.Event()
        self.stage_task = None
        self.operation_counter = 0
        self.service_profile_sha256 = None
        self.service_epoch = None
        self.build_completion_unknown = False

    def _rid(self, action):
        self.operation_counter += 1
        return f'{self.request_prefix}:{action}:{self.operation_counter}'

    async def _post(self, path, body):
        started = time.monotonic()
        try:
            async with self.client.post(self.endpoint+path, json=body) as response:
                result = await response.json()
                if response.status != 200:
                    raise KVProtocolError(f'KV HTTP {response.status}: {result}')
                return result
        finally:
            self.emit('kv_client_operation', path=path, request_id=body['request_id'],
                      handle_id=self.handle_id, seconds=time.monotonic()-started)

    async def control(self, action, **extra):
        if self.handle_id is None:
            raise KVProtocolError('No engine handle has been registered')
        rid = self._rid(action)
        receipt = await self._post('/kv/control', dict(request_id=rid,
            action=action, handle_id=self.handle_id,
            service_profile_sha256=self.service_profile_sha256, **extra))
        expected = dict(request_id=rid, action=action, handle_id=self.handle_id,
                        service_profile_sha256=self.service_profile_sha256,
                        service_epoch=self.service_epoch)
        if any(receipt.get(k) != v for k, v in expected.items()):
            raise KVProtocolError('KV receipt operation/handle/service identity differs')
        self.receipt = receipt
        self.emit('kv_engine_receipt', handle_id=self.handle_id, action=action, receipt=receipt)
        return receipt

    async def _wait_state(self, predicate):
        deadline = time.monotonic()+self.timeout
        while True:
            receipt = await self.control('status')
            if not receipt.get('ok'):
                raise KVProtocolError(str(receipt.get('error')))
            if predicate(receipt):
                return receipt
            if time.monotonic() >= deadline:
                raise TimeoutError('KV native completion events did not become ready')
            await asyncio.sleep(self.poll_seconds)

    def start(self, known_body, *, predicted_todo_end, lease_seconds=600):
        if self.stage_task is not None:
            raise ValueError('A lifecycle can be started only once')
        self.stage_task = asyncio.create_task(self._stage(known_body, predicted_todo_end, lease_seconds))
        return self.stage_task

    def notify_consumer_ready(self):
        self.consumer_ready.set()

    async def _stage(self, body, predicted_end, lease_seconds):
        started = time.monotonic()
        try:
            self.build_completion_unknown = True
            built = await self._post('/kv/prefill', dict(body=body,
                request_id=self._rid('build'), lease_seconds=lease_seconds))
            registered = built['registration']
            self.emit('kv_build_result', result=built)
            if not registered.get('ok'):
                if not registered.get('handle_id'):
                    # The proxy received a definitive native build terminal and
                    # register rejection. A missing/invalid HTTP receipt takes
                    # the exception path and cannot make this assertion.
                    self.build_completion_unknown = False
                return dict(ok=False, reason='register_failed', receipt=registered)
            if (not isinstance(registered.get('handle_id'), str) or not registered['handle_id'] or
                not isinstance(registered.get('service_epoch'), str) or not registered['service_epoch'] or
                not isinstance(built.get('service_profile_sha256'), str) or
                registered.get('service_profile_sha256') != built['service_profile_sha256'] or
                registered.get('prefix_sha256') != built.get('prefix_sha256') or
                registered.get('requested_prefix_sha256') != built.get('requested_prefix_sha256') or
                registered.get('requested_prefix_tokens') != built.get('requested_prefix_tokens') or
                registered.get('cached_prefix_tokens') != built.get('cached_prefix_tokens') or
                type(registered.get('cached_prefix_tokens')) is not int or
                type(registered.get('requested_prefix_tokens')) is not int or
                not 0 < registered['cached_prefix_tokens'] <= registered['requested_prefix_tokens']):
                raise KVProtocolError('Build/register identity chain differs')
            self.handle_id = registered['handle_id']
            self.service_profile_sha256 = registered['service_profile_sha256']
            self.service_epoch = registered['service_epoch']
            self.build_completion_unknown = False
            self.receipt = registered
            if self.retired.is_set():
                return dict(ok=False, reason='retired_during_build')
            remaining = predicted_end-time.monotonic()
            # Do not offload just to claim it happened when there is no useful
            # interval left. Future request arrival is not delayed for a timer.
            if self.consumer_ready.is_set() or remaining <= self.preload_lead+self.minimum_offloaded:
                self.emit('kv_keep_resident', handle_id=self.handle_id, remaining_seconds=remaining)
                return dict(ok=True, handle_id=self.handle_id, offload_skipped=True, receipt=registered)
            backed = await self.control('backup')
            if not backed.get('ok'):
                return dict(ok=False, reason='backup_rejected', receipt=backed)
            await self._wait_state(lambda r:r.get('host_ready') is True)
            if self.retired.is_set():
                return dict(ok=False, reason='retired_during_backup')
            remaining = predicted_end-time.monotonic()
            if self.consumer_ready.is_set() or remaining <= self.preload_lead+self.minimum_offloaded:
                self.emit('kv_keep_resident', handle_id=self.handle_id, remaining_seconds=remaining)
                return dict(ok=True, handle_id=self.handle_id, offload_skipped=True, receipt=self.receipt)
            offloaded = await self.control('offload')
            if not offloaded.get('ok'):
                return dict(ok=False, reason='offload_rejected', receipt=offloaded)
            delay = max(0.0, predicted_end-self.preload_lead-time.monotonic())
            foreground = asyncio.create_task(self.consumer_ready.wait())
            retire = asyncio.create_task(self.retired.wait())
            try:
                await asyncio.wait([foreground, retire], timeout=delay, return_when=asyncio.FIRST_COMPLETED)
            finally:
                for waiter in (foreground, retire):
                    if not waiter.done(): waiter.cancel()
                await asyncio.gather(foreground, retire, return_exceptions=True)
            if self.retired.is_set():
                return dict(ok=False, reason='retired_while_offloaded')
            loaded = await self.control('prefetch', deadline_monotonic_ns=int(predicted_end*1e9))
            if not loaded.get('ok'):
                return dict(ok=False, reason='prefetch_rejected', receipt=loaded)
            ready = await self._wait_state(lambda r:r.get('device_ready') is True)
            self.emit('kv_device_ready', handle_id=self.handle_id,
                      before_predicted_end=time.monotonic() <= predicted_end,
                      before_actual_consumer_ready=not self.consumer_ready.is_set(), receipt=ready)
            return dict(ok=True, handle_id=self.handle_id, receipt=ready)
        except Exception as exc:
            self.emit('kv_lifecycle_failed', handle_id=self.handle_id,
                      error=f'{type(exc).__name__}: {exc}')
            return dict(ok=False, reason='lifecycle_failed', error=str(exc))
        finally:
            self.emit('kv_lifecycle_finished', handle_id=self.handle_id,
                      seconds=time.monotonic()-started)

    async def retire_and_drain(self):
        """Quiesce actual submitted work before reporting experiment completion."""
        self.retired.set()
        if self.stage_task is not None:
            await asyncio.shield(self.stage_task)
        if self.handle_id is None:
            if self.build_completion_unknown:
                raise KVProtocolError('Build/register completion is unknown; guard must close service')
            return dict(ok=True, no_registered_handle=True)
        deadline = time.monotonic()+self.timeout
        while True:
            receipt = await self.control('cancel')
            if (receipt.get('ok') and receipt.get('state') == 'RELEASED'
                and receipt.get('consumer_refs') == 0 and receipt.get('pending_operations') == []):
                return receipt
            if time.monotonic() >= deadline:
                raise TimeoutError('KV handle cleanup was not acknowledged; guard must close service')
            await asyncio.sleep(self.poll_seconds)
