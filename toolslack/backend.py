"""Native memory and scheduler-owned exact-prefix SGLang transport.

Memory content is supplied by an unchanged official caller. All GPU work,
residency and readiness below come from native service receipts. The v15
proxy needs the explicit-prefix-cap patch before bounded L2 is advertised.
"""
from __future__ import annotations

import asyncio
from copy import deepcopy
import math
import time
from typing import Any, Awaitable, Callable
import uuid

from .types import KVHandle, NativeResult, Scope, digest


class BackendUnavailable(RuntimeError):
    """A definitive optional decline; the caller may keep ready L1 text."""


class BackendProtocolError(RuntimeError):
    """Native ownership or identity is uncertain; cleanup must be confirmed."""


class SGLangBackend:
    def __init__(self, proxy_url: str, model_key: str,
                 service_profile_sha256: str,
                 native_compactor: Callable[[dict, Scope], Awaitable[NativeResult]],
                 *, transport=None, clock=time.monotonic, wall_clock=time.time,
                 poll_seconds=.02, drain_timeout_s=1800., safety_s=.02,
                 prefetch_lead_s: float | None = None, minimum_offloaded_s=.1):
        if not proxy_url.startswith(('http://', 'https://')):
            raise ValueError('A real proxy HTTP endpoint is required')
        if not model_key or len(service_profile_sha256) != 64:
            raise ValueError('Model and frozen engine service identity are required')
        for value in (poll_seconds, drain_timeout_s, safety_s, minimum_offloaded_s):
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError('Finite nonnegative backend timing required')
        if poll_seconds <= 0 or drain_timeout_s <= 0:
            raise ValueError('Positive poll and drain timeout required')
        if prefetch_lead_s is not None and (not math.isfinite(prefetch_lead_s) or prefetch_lead_s <= 0):
            raise ValueError('Prefetch lead must be a positive measured estimate')
        self.proxy_url, self.model_key = proxy_url.rstrip('/'), model_key
        self.service_profile_sha256 = service_profile_sha256
        self.native_compactor, self.transport = native_compactor, transport
        self.clock, self.wall_clock = clock, wall_clock
        self.poll_seconds, self.drain_timeout_s, self.safety_s = poll_seconds, drain_timeout_s, safety_s
        self.prefetch_lead_s, self.minimum_offloaded_s = prefetch_lead_s, minimum_offloaded_s
        self._session = None
        self._capabilities = None
        self._handles: dict[str, KVHandle] = {}
        self._retiring: dict[str, asyncio.Event] = {}
        self._tool_ready: dict[str, asyncio.Event] = {}
        self._native_owners: set[asyncio.Task] = set()
        self._placements: set[asyncio.Task] = set()
        self.requires_service_cleanup = False

    async def _http(self, method, path, body=None):
        if self.transport is not None:
            result = await self.transport(method, path, deepcopy(body))
            if not isinstance(result, tuple) or len(result) != 2:
                raise BackendProtocolError('Transport must return HTTP status and JSON mapping')
            return result
        import aiohttp  # Only the real HTTP implementation needs this dependency.
        if self._session is None:
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.drain_timeout_s))
        async with self._session.request(method, self.proxy_url + path, json=body) as response:
            return response.status, await response.json()

    async def capabilities(self):
        if self._capabilities is None:
            status, value = await self._http('GET', '/toolslack/capabilities')
            if status == 404:
                self._capabilities = {}
            elif status != 200 or not isinstance(value, dict):
                raise BackendUnavailable('Native proxy capability discovery failed')
            elif value.get('service_profile_sha256') != self.service_profile_sha256:
                raise BackendProtocolError('Proxy capability service profile differs')
            else:
                self._capabilities = value
        return dict(self._capabilities)

    async def compact(self, snapshot: dict, scope: Scope) -> NativeResult:
        if not isinstance(snapshot, dict) or not isinstance(snapshot.get('messages'), list):
            raise ValueError('An immutable message snapshot is required')
        if scope.stop > len(snapshot['messages']):
            raise ValueError('Scope extends beyond the source snapshot')
        selected = snapshot['messages'][scope.start:scope.stop]
        if scope.source_sha256 and digest(selected) != scope.source_sha256:
            raise ValueError('Selected original scope identity differs')
        result = await self.native_compactor(deepcopy(snapshot), scope)
        if not isinstance(result, NativeResult):
            raise BackendProtocolError('Native callback must return NativeResult')
        if (not isinstance(result.replacement_messages, list)
                or not all(isinstance(m, dict) for m in result.replacement_messages)
                or not isinstance(result.stable_body, dict)):
            raise BackendProtocolError('Native memory returned malformed candidate')
        result = deepcopy(result)
        result.metadata.setdefault('native_algorithm_external', True)
        return result

    @staticmethod
    def _closed(messages):
        """Optional KV must never include a tool dispatch lacking its result."""
        outstanding = set()
        for message in messages:
            if not isinstance(message, dict):
                raise ValueError('Messages must be native mappings')
            if message.get('role') == 'assistant':
                for call in message.get('tool_calls') or []:
                    call_id = call.get('id') if isinstance(call, dict) else None
                    if not isinstance(call_id, str) or not call_id or call_id in outstanding:
                        raise ValueError('Malformed tool-call identity')
                    outstanding.add(call_id)
            elif message.get('role') == 'tool':
                call_id = message.get('tool_call_id')
                if call_id not in outstanding:
                    raise ValueError('Tool result has no earlier dispatch')
                outstanding.remove(call_id)
        if outstanding:
            raise BackendUnavailable('The supplied KV body includes an unfinished tool call')

    async def prepare_kv(self, stable_body: dict, kv_tokens: int, deadline: float,
                         identity: dict) -> KVHandle:
        if self.requires_service_cleanup:
            raise BackendProtocolError('An earlier native operation needs service cleanup')
        if type(kv_tokens) is not int or kv_tokens <= 0:
            raise ValueError('A positive exact prefix token budget is required')
        if not math.isfinite(deadline) or deadline <= self.clock() + self.safety_s:
            raise BackendUnavailable('Tool budget expired before KV preparation')
        if not isinstance(identity, dict) or identity.get('model_key') != self.model_key:
            raise ValueError('Candidate model identity differs')
        if identity.get('body_sha256') != digest(stable_body):
            raise ValueError('Candidate stable-body identity differs')
        messages = stable_body.get('messages')
        if not isinstance(messages, list) or not messages:
            raise ValueError('Closed native body is required')
        self._closed(messages)
        future = identity.get('known_future_body')
        if not isinstance(future, dict):
            raise BackendUnavailable('An exact known tool-dispatch serialization is required')
        if future.get('messages', [])[:len(messages)] != messages:
            raise ValueError('Known future does not preserve the stable messages')
        scaffold = lambda b: {k: v for k, v in b.items() if k != 'messages'}
        if digest(scaffold(stable_body)) != digest(scaffold(future)):
            raise ValueError('Known future changed native template settings')
        cap = await self.capabilities()
        if cap.get('bounded_prefix_prefill') is not True or cap.get('deadline_terminal_drain') is not True:
            raise BackendUnavailable('Proxy has no confirmed bounded-prefix / terminal-drain capability')
        request_id = 'toolslack:' + uuid.uuid4().hex + ':kv'
        spec = dict(body=deepcopy(stable_body), known_future_body=deepcopy(future),
                    request_id=request_id, max_prefix_tokens=kv_tokens,
                    deadline_unix_ms=(self.wall_clock() + deadline - self.clock()) * 1000,
                    safety_ms=self.safety_s * 1000, lease_seconds=min(1800, max(1, self.drain_timeout_s)))
        owner = asyncio.create_task(self._prepare_owned(spec, identity))
        self._native_owners.add(owner)
        # Keep RPC ownership if a tool-ready event cancels this Python waiter.
        # Drain later includes the native server response and any created handle.
        try:
            return await asyncio.shield(owner)
        finally:
            if owner.done():
                self._native_owners.discard(owner)

    async def _prepare_owned(self, spec, identity):
        try:
            status, result = await self._http('POST', '/kv/prefill', spec)
        except BaseException as error:
            self.requires_service_cleanup = True
            raise BackendProtocolError('KV RPC lost; native termination unknown') from error
        if not isinstance(result, dict):
            self.requires_service_cleanup = True
            raise BackendProtocolError('Malformed native prefill response')
        if status != 200:
            if result.get('native_submitted') is False or result.get('native_terminal_confirmed') is True:
                raise BackendUnavailable(str(result.get('error', 'Optional KV declined')))
            self.requires_service_cleanup = True
            raise BackendProtocolError('Optional prefill terminal state is unconfirmed')
        registration = result.get('registration')
        if not isinstance(registration, dict):
            self.requires_service_cleanup = True
            raise BackendProtocolError('Missing native registration receipt')
        if registration.get('ok') is not True:
            if registration.get('handle_id'):
                self.requires_service_cleanup = True
                raise BackendProtocolError('Rejected registration may retain a native handle')
            raise BackendUnavailable(str(registration.get('error', 'Native prefix was evicted')))
        # Adopt ownership before validating all identities so a rejected receipt
        # still has a handle to retire; never publish that handle to inference.
        handle_id, epoch = registration.get('handle_id'), registration.get('service_epoch')
        if not isinstance(handle_id, str) or not handle_id or not isinstance(epoch, str) or not epoch:
            self.requires_service_cleanup = True
            raise BackendProtocolError('Missing native handle or engine epoch')
        provisional = KVHandle('', 0, handle_id, self.model_key, epoch, ready=False,
                               metadata={'profile_sha256': self.service_profile_sha256})
        self._handles[handle_id] = provisional
        self._retiring[handle_id] = asyncio.Event()
        self._tool_ready[handle_id] = asyncio.Event()
        try:
            ids = result.get('selected_prefix_ids')
            covered = registration.get('cached_prefix_tokens')
            if (not isinstance(ids, list) or not ids or not all(type(v) is int and v >= 0 for v in ids)
                    or len(ids) > spec['max_prefix_tokens'] or type(covered) is not int
                    or not 0 < covered <= len(ids)):
                raise BackendProtocolError('Native prefix violates exact token budget')
            expected = {'action': 'register', 'request_id': spec['request_id'] + ':register',
                        'service_profile_sha256': self.service_profile_sha256,
                        'requested_prefix_tokens': len(ids), 'requested_prefix_sha256': digest(ids),
                        'prefix_sha256': digest(ids[:covered]), 'device_ready': True}
            if any(registration.get(k) != v for k, v in expected.items()):
                raise BackendProtocolError('Native prefix operation / token identity differs')
            if (result.get('requested_prefix_tokens') != len(ids)
                    or result.get('requested_prefix_sha256') != digest(ids)
                    or result.get('prefix_sha256') != digest(ids[:covered])
                    or result.get('service_profile_sha256') != self.service_profile_sha256):
                raise BackendProtocolError('Proxy and engine prefix identity differs')
            meta = (result.get('prefill') or {}).get('meta_info') or {}
            if (meta.get('id') != spec['request_id'] + ':prefill'
                    or meta.get('prompt_tokens') != len(ids) or meta.get('completion_tokens') != 0):
                raise BackendProtocolError('Native zero-generation terminal receipt differs')
            provisional.prefix_sha256, provisional.token_count = digest(ids[:covered]), covered
            provisional.ready = True
            provisional.metadata.update(registration=deepcopy(registration),
                requested_prefix_ids=ids, cached_prefix_ids=ids[:covered],
                body_sha256=identity['body_sha256'], native_prefill_terminal_confirmed=True,
                requested_kv_tokens=spec['max_prefix_tokens'], deadline_unix_ms=spec['deadline_unix_ms'])
            return provisional
        except BackendProtocolError:
            self.requires_service_cleanup = True
            # Resource ownership is preserved for explicit drain/guard cleanup.
            raise

    async def register_existing(self, stable_body: dict, known_future_body: dict,
                                deadline: float, identity: dict) -> KVHandle:
        """Pin a native resident prefix without generation or content changes.

        The authoritative registration must be an exact leading token sequence
        of the known next tool-dispatch serialization. A cache miss is a no-op;
        it is never repaired with speculative model work.
        """
        if self.requires_service_cleanup:
            raise BackendProtocolError('An earlier native operation needs service cleanup')
        if not math.isfinite(deadline) or deadline <= self.clock() + self.safety_s:
            raise BackendUnavailable('Tool budget expired before existing-prefix registration')
        if (not isinstance(identity, dict) or identity.get('model_key') != self.model_key
                or identity.get('body_sha256') != digest(stable_body)):
            raise ValueError('Existing-prefix candidate identity differs')
        messages = stable_body.get('messages')
        if not isinstance(messages, list) or not messages:
            raise ValueError('Closed native body is required')
        self._closed(messages)
        if not isinstance(known_future_body, dict):
            raise BackendUnavailable('Exact known tool-dispatch body is required')
        future_messages = known_future_body.get('messages')
        if not isinstance(future_messages, list) or future_messages[:len(messages)] != messages:
            raise ValueError('Known future does not preserve the stable messages')
        suffix = future_messages[len(messages):]
        if (len(suffix) != 1 or suffix[0].get('role') != 'assistant'
                or not suffix[0].get('tool_calls')):
            raise ValueError('Known future must append exactly one assistant tool dispatch')
        scaffold = lambda b: {k: v for k, v in b.items() if k != 'messages'}
        if digest(scaffold(stable_body)) != digest(scaffold(known_future_body)):
            raise ValueError('Known future changed native template settings')
        cap = await self.capabilities()
        if cap.get('existing_prefix_registration') is not True:
            raise BackendUnavailable('Proxy has no confirmed native existing-prefix capability')
        rid = 'toolslack:' + uuid.uuid4().hex + ':existing'
        spec = dict(body=deepcopy(stable_body), known_future_body=deepcopy(known_future_body),
                    request_id=rid, deadline_unix_ms=(self.wall_clock() + deadline - self.clock()) * 1000,
                    safety_ms=self.safety_s * 1000,
                    lease_seconds=min(1800, max(1, self.drain_timeout_s)))
        owner = asyncio.create_task(self._register_existing_owned(spec, identity, deadline))
        self._native_owners.add(owner)
        try:
            return await asyncio.shield(owner)
        finally:
            if owner.done():
                self._native_owners.discard(owner)

    async def _register_existing_owned(self, spec, identity, deadline):
        counts = []
        for body in (spec['body'], spec['known_future_body']):
            status, exact = await self._http('POST', '/exact_tokens', body)
            if status != 200 or not isinstance(exact, dict):
                raise BackendUnavailable('Native exact serialization is unavailable')
            ids = exact.get('input_ids')
            if (not isinstance(ids, list) or not ids or not all(type(v) is int and v >= 0 for v in ids)
                    or exact.get('input_ids_sha256') != digest(ids)
                    or exact.get('model_name') != self.model_key
                    or exact.get('kv_service_profile_sha256') != self.service_profile_sha256
                    or not isinstance(exact.get('payload_sha256'), str)
                    or len(exact['payload_sha256']) != 64
                    or not isinstance(exact.get('model_profile_sha256'), str)
                    or len(exact['model_profile_sha256']) != 64):
                raise BackendProtocolError('Native exact serialization identity differs')
            counts.append(exact)
        if counts[0].get('model_profile_sha256') != counts[1].get('model_profile_sha256'):
            raise BackendProtocolError('Native model profile changed between serializations')
        if self.clock() + self.safety_s >= deadline:
            raise BackendUnavailable('Serialization exhausted the existing-prefix window')
        try:
            status, result = await self._http('POST', '/kv/register-existing', spec)
        except BaseException as error:
            self.requires_service_cleanup = True
            raise BackendProtocolError('Existing registration RPC lost; native ownership unknown') from error
        if status != 200 or not isinstance(result, dict):
            if isinstance(result, dict) and result.get('native_submitted') is False:
                raise BackendUnavailable(str(result.get('error', 'Existing prefix declined')))
            self.requires_service_cleanup = True
            raise BackendProtocolError('Existing native registration is unconfirmed')
        registration = result.get('registration')
        if not isinstance(registration, dict):
            self.requires_service_cleanup = True
            raise BackendProtocolError('Missing existing-prefix registration receipt')
        if registration.get('ok') is not True:
            if registration.get('handle_id'):
                self.requires_service_cleanup = True
                raise BackendProtocolError('Failed existing registration may retain native ownership')
            raise BackendUnavailable('Exact existing prefix is no longer resident')
        handle_id, epoch = registration.get('handle_id'), registration.get('service_epoch')
        if not isinstance(handle_id, str) or not handle_id or not isinstance(epoch, str) or not epoch:
            self.requires_service_cleanup = True
            raise BackendProtocolError('Missing existing handle or native service epoch')
        handle = KVHandle('', 0, handle_id, self.model_key, epoch, ready=False,
                          metadata={'profile_sha256': self.service_profile_sha256})
        self._handles[handle_id] = handle
        self._retiring[handle_id] = asyncio.Event()
        self._tool_ready[handle_id] = asyncio.Event()
        try:
            requested, covered = registration.get('requested_prefix_tokens'), registration.get('cached_prefix_tokens')
            future_ids = counts[1]['input_ids']
            if (type(requested) is not int or type(covered) is not int
                    or not 0 < covered <= requested <= len(future_ids)):
                raise BackendProtocolError('Existing prefix token count is invalid')
            ids = future_ids[:requested]
            expected = dict(action='register', request_id=spec['request_id'] + ':register',
                            service_profile_sha256=self.service_profile_sha256,
                            requested_prefix_sha256=digest(ids), prefix_sha256=digest(ids[:covered]),
                            device_ready=True)
            if any(registration.get(k) != v for k, v in expected.items()):
                raise BackendProtocolError('Existing native prefix is not an exact known-future prefix')
            expected_result = dict(existing_prefix_only=True, native_generate_submitted=False,
                                   semantic_context_changed=False, optional_cache_miss=False,
                                   requested_prefix_tokens=requested, requested_prefix_sha256=digest(ids),
                                   cached_prefix_tokens=covered, prefix_sha256=digest(ids[:covered]),
                                   known_payload_sha256=counts[0]['payload_sha256'],
                                   service_profile_sha256=self.service_profile_sha256,
                                   future_consumer_match_required=True)
            if any(result.get(k) != v for k, v in expected_result.items()):
                raise BackendProtocolError('Proxy existing registration provenance differs')
            handle.prefix_sha256, handle.token_count, handle.ready = digest(ids[:covered]), covered, True
            handle.metadata.update(registration=deepcopy(registration), requested_prefix_ids=ids,
                                   cached_prefix_ids=ids[:covered], body_sha256=identity['body_sha256'],
                                   existing_prefix_only=True, native_generate_submitted=False,
                                   stable_scope=deepcopy(result.get('stable_scope')),
                                   deadline_unix_ms=spec['deadline_unix_ms'])
            return handle
        except BackendProtocolError:
            self.requires_service_cleanup = True
            raise

    async def _control(self, handle: KVHandle, action: str, **extra):
        request_id = 'toolslack:' + uuid.uuid4().hex + ':' + action
        body = dict(request_id=request_id, action=action, handle_id=handle.handle_id,
                    service_profile_sha256=self.service_profile_sha256, **extra)
        owner = asyncio.create_task(self._http('POST', '/kv/control', body))
        self._native_owners.add(owner)
        try:
            status, receipt = await asyncio.shield(owner)
        except asyncio.CancelledError:
            raise
        except BaseException as error:
            self.requires_service_cleanup = True
            raise BackendProtocolError('KV control transport lost; native operation may remain') from error
        finally:
            if owner.done():
                self._native_owners.discard(owner)
        expected = dict(request_id=request_id, action=action, handle_id=handle.handle_id,
                        service_profile_sha256=self.service_profile_sha256, service_epoch=handle.epoch)
        if status != 200 or not isinstance(receipt, dict) or any(receipt.get(k) != v for k, v in expected.items()):
            self.requires_service_cleanup = True
            raise BackendProtocolError('KV control operation / engine identity differs')
        if handle.prefix_sha256 and receipt.get('prefix_sha256') != handle.prefix_sha256:
            self.requires_service_cleanup = True
            raise BackendProtocolError('KV control token identity differs')
        handle.metadata['last_receipt'] = deepcopy(receipt)
        return receipt

    async def _wait(self, handle, predicate):
        until = self.clock() + self.drain_timeout_s
        while True:
            receipt = await self._control(handle, 'status')
            if receipt.get('ok') is not True:
                raise BackendUnavailable(str(receipt.get('error')))
            if predicate(receipt):
                return receipt
            if self.clock() >= until:
                self.requires_service_cleanup = True
                raise BackendProtocolError('Native KV operation did not drain')
            await asyncio.sleep(self.poll_seconds)

    def device_ready(self, handle: KVHandle) -> bool:
        return (self._handles.get(handle.handle_id) is handle and handle.ready is True
                and handle.location == 'hbm' and not self._retiring[handle.handle_id].is_set())

    def consumer_headers(self, handle: KVHandle) -> dict[str, str]:
        if not self.device_ready(handle):
            raise BackendUnavailable('Prefix is not locally confirmed ready')
        return {'x-toolslack-kv-handle': handle.handle_id,
                'x-toolslack-kv-one-shot': 'true'}

    def notify_tool_ready(self, handle: KVHandle):
        """End the background placement timer without cancelling native DMA."""
        event = self._tool_ready.get(handle.handle_id)
        if event is not None:
            event.set()

    stop_placement = notify_tool_ready

    async def place(self, handle: KVHandle, deadline: float):
        """Optional real DRAM round trip; timer changes only background work."""
        if self.prefetch_lead_s is None:
            return {'offload_skipped': True, 'reason': 'no_measured_transfer_lead'}
        if handle.handle_id not in self._handles or not self.device_ready(handle):
            raise BackendUnavailable('No owned ready native prefix')
        if deadline - self.clock() <= self.prefetch_lead_s + self.minimum_offloaded_s:
            return {'offload_skipped': True, 'reason': 'insufficient_reuse_window'}
        task = asyncio.current_task()
        self._placements.add(task)
        retired = self._retiring[handle.handle_id]
        tool_ready = self._tool_ready[handle.handle_id]
        try:
            receipt = await self._control(handle, 'backup')
            if receipt.get('ok') is not True:
                raise BackendUnavailable(str(receipt.get('error')))
            await self._wait(handle, lambda r: r.get('host_ready') is True)
            if retired.is_set() or tool_ready.is_set() or deadline - self.clock() <= self.prefetch_lead_s + self.minimum_offloaded_s:
                return {'offload_skipped': True, 'reason': 'tool_window_ended_during_backup'}
            handle.ready = False  # Admission stops before native residency can change.
            receipt = await self._control(handle, 'offload')
            if receipt.get('ok') is not True:
                handle.ready = True  # Definitive native rejection did not offload.
                raise BackendUnavailable(str(receipt.get('error')))
            handle.location, handle.ready = 'dram', False
            # Engine may retain shared nodes: its events, not this adapter,
            # provide the released byte/token count.
            handle.metadata['offload_receipt'] = deepcopy(receipt)
            delay = max(0., deadline - self.prefetch_lead_s - self.clock())
            waiters = [asyncio.create_task(retired.wait()), asyncio.create_task(tool_ready.wait())]
            try:
                await asyncio.wait(waiters, timeout=delay, return_when=asyncio.FIRST_COMPLETED)
            finally:
                for waiter in waiters:
                    waiter.cancel()
                await asyncio.gather(*waiters, return_exceptions=True)
            if retired.is_set() or tool_ready.is_set():
                return {'retired_while_offloaded': retired.is_set(),
                        'actual_tool_ready_before_prefetch': tool_ready.is_set()}
            receipt = await self._control(handle, 'prefetch', deadline_monotonic_ns=int(deadline * 1e9))
            if receipt.get('ok') is not True:
                raise BackendUnavailable(str(receipt.get('error')))
            await self._wait(handle, lambda r: r.get('device_ready') is True)
            handle.location, handle.ready = 'hbm', True
            return {'device_ready': True, 'before_deadline': self.clock() <= deadline}
        finally:
            self._placements.discard(task)

    async def release(self, handle: KVHandle):
        if handle.handle_id not in self._handles:
            return {'released': True, 'already_retired': True}
        self._retiring[handle.handle_id].set()
        handle.ready = False
        until = self.clock() + self.drain_timeout_s
        while True:
            receipt = await self._control(handle, 'cancel')
            if (receipt.get('ok') is True and receipt.get('state') == 'RELEASED'
                    and receipt.get('consumer_refs') == 0 and receipt.get('pending_operations') == []):
                self._handles.pop(handle.handle_id, None)
                self._retiring.pop(handle.handle_id, None)
                self._tool_ready.pop(handle.handle_id, None)
                handle.location = 'released'
                return receipt
            code = (receipt.get('error') or {}).get('code')
            if (code not in (None, 'handle_busy', 'consumer_busy', 'transfer_busy') or self.clock() >= until):
                self.requires_service_cleanup = True
                raise BackendProtocolError('Native handle release could not be confirmed')
            await asyncio.sleep(self.poll_seconds)

    async def drain(self):
        errors = []
        owners = list(self._native_owners)
        if owners:
            results = await asyncio.gather(*(asyncio.shield(t) for t in owners), return_exceptions=True)
            errors.extend(r for r in results if isinstance(r, BaseException) and not isinstance(r, BackendUnavailable))
            self._native_owners.difference_update(owners)
        for handle in list(self._handles.values()):
            self._retiring[handle.handle_id].set()
        if self._placements:
            results = await asyncio.gather(*(asyncio.shield(t) for t in list(self._placements)), return_exceptions=True)
            errors.extend(r for r in results if isinstance(r, BaseException) and not isinstance(r, BackendUnavailable))
        for handle in list(self._handles.values()):
            try:
                await self.release(handle)
            except Exception as error:
                errors.append(error)
        if errors or self.requires_service_cleanup:
            raise BackendProtocolError('Native ownership drainage needs guard cleanup') from (errors[0] if errors else None)
        return {'drained': True, 'native_owners': 0, 'handles': 0}

    async def close(self):
        await self.drain()
        if self._session is not None:
            await self._session.close()
            self._session = None
