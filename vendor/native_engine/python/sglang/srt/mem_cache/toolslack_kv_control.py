"""Scheduler-owned exact-prefix leases. No model or memory algorithm changes.

The transfer adapter reserves GPU slots without publishing node.value. Its native
HiCache copies run asynchronously; publication/free is exclusively scheduler-owned.
"""
from collections import OrderedDict
import copy
import hashlib
import json
import time
import uuid


def token_hash(ids):
    return hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()


class Rejected(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def require(ok, code, message):
    if not ok:
        raise Rejected(code, message)


class PrefixManager:
    """All methods (including tick/on_split) run in the scheduler owner thread."""

    def __init__(self, cache, transfers, service_profile_sha256, active_rids,
                 clock=time.monotonic_ns, submit_grace_seconds=120, max_handles=256,
                 foreground_reservation=None):
        self.cache, self.transfers = cache, transfers
        self.profile = service_profile_sha256
        self.active_rids, self.clock = active_rids, clock
        self.foreground_reservation = foreground_reservation
        self.submit_grace_ns = int(submit_grace_seconds * 1e9)
        self.max_handles = max_handles
        self.epoch = uuid.uuid4().hex
        self.handles = {}
        self.requests = OrderedDict()
        self.write_events = {}
        self.reserved_device_tokens = 0

    def emit(self, h, event, **fields):
        h['events'].append(dict(event=event, event_id=f"{h['id']}:{len(h['events'])}",
                                monotonic_ns=self.clock(), **fields))

    def path(self, endpoint):
        result = []
        while endpoint is not self.cache.root_node:
            require(endpoint is not None, 'stale_tree', 'Prefix detached from current radix root')
            result.append(endpoint)
            endpoint = endpoint.parent
        return list(reversed(result))

    def on_split(self, child, parent):
        # Native split copies lock_ref, so endpoint chain GPU refs already follow
        # the split. Native host_ref_counter is not copied: add only our own refs.
        for h in self.handles.values():
            if not h['released'] and child in h['host_pins']:
                parent.protect_host()
                h['host_pins'].add(parent)
        if child.id in self.write_events:
            self.write_events[parent.id] = self.write_events[child.id]

    def _refresh_native_events(self):
        for start, finish, ids in self.cache.cache_controller.ack_write_queue:
            for node_id in ids:
                self.write_events[node_id] = (start, finish)

    def _native_load_pending(self, nodes):
        wanted = set(nodes)
        for endpoint in self.cache.ongoing_load_back.values():
            if wanted.intersection(self.path(endpoint)):
                return True
        return False

    def _pending_write(self, nodes):
        return any(n.id in self.write_events and not self.write_events[n.id][1].query()
                   for n in nodes)

    def _host_ready(self, h):
        nodes = self.path(h['endpoint'])
        return all(n.backuped for n in nodes) and not self._pending_write(nodes)

    def _device_ready(self, h):
        nodes = self.path(h['endpoint'])
        return (h['load'] is None and all(not n.evicted for n in nodes)
                and not self._native_load_pending(nodes)
                and (h['gpu_hold'] or bool(h['consumers'])))

    def _gpu_hold(self, h):
        if not h['gpu_hold']:
            require(all(not n.evicted for n in self.path(h['endpoint'])),
                    'not_resident', 'Cannot pin an incomplete GPU chain')
            self.cache.inc_lock_ref(h['endpoint'])
            h['gpu_hold'] = True

    def _gpu_drop(self, h):
        if h['gpu_hold']:
            self.cache.dec_lock_ref(h['endpoint'])
            h['gpu_hold'] = False

    def _headroom_reservation(self, action, h=None):
        try:
            value = self.foreground_reservation() if self.foreground_reservation else None
        except Exception:
            value = None
        if (not isinstance(value, dict) or value.get('schema') != 'native-admitted-request-headroom.v1'
                or value.get('supported') is not True or type(value.get('reserve_tokens')) is not int
                or value['reserve_tokens'] < 0):
            self._headroom_reject(action, h, dict(reservation=value,
                reason='admitted_reservation_unknown'))
        return value

    def _headroom_accounting(self, action, h=None):
        try:
            value = self.transfers.accounting()
        except Exception:
            value = None
        required = ('allocator_total_tokens', 'allocator_free_tokens',
                    'radix_protected_tokens', 'radix_evictable_tokens', 'other_allocated_tokens')
        if (not isinstance(value, dict) or any(type(value.get(k)) is not int or value[k] < 0 for k in required)
                or sum(value[k] for k in required[1:]) != value['allocator_total_tokens']):
            self._headroom_reject(action, h, dict(accounting=value,
                reason='native_allocator_accounting_unknown'))
        return value

    def _headroom_reject(self, action, h, fields):
        evidence = dict(event='optional_headroom_rejected', operation=action,
            monotonic_ns=self.clock(), **fields)
        if h is not None and h.get('id') in self.handles:
            self.emit(h, 'optional_headroom_rejected', operation=action, **fields)
        exc = Rejected('foreground_headroom',
            'Optional prefix hold would consume admitted native request capacity')
        exc.headroom = evidence
        raise exc

    def _optional_gpu_hold(self, h, action):
        if h['gpu_hold']:
            return None
        reservation = self._headroom_reservation(action, h)
        before = self._headroom_accounting(action, h)
        self._gpu_hold(h)
        try:
            after = self._headroom_accounting(action, h)
            available = after['allocator_free_tokens'] + after['radix_evictable_tokens']
            evidence = dict(reservation=reservation, allocator_before=before, allocator_after=after,
                newly_protected_tokens=after['radix_protected_tokens']-before['radix_protected_tokens'],
                allocatable_after_hold_tokens=available, materialization_tokens=0)
            if available < reservation['reserve_tokens']:
                self._headroom_reject(action, h, evidence)
            return evidence
        except BaseException:
            self._gpu_drop(h)
            raise

    def _register(self, body):
        ids = body.get('input_ids')
        require(isinstance(ids, list) and ids and all(type(x) is int and x >= 0 for x in ids),
                'invalid_tokens', 'input_ids must be a nonempty list of nonnegative integers')
        require(body.get('position_base') == 0, 'position_mismatch', 'Only absolute prefix position zero is supported')
        require(body.get('prefix_sha256') == token_hash(ids), 'token_hash_mismatch', 'Requested token hash differs')
        ttl = body.get('lease_seconds', 600)
        require(type(ttl) in (int, float) and 0 < ttl <= 7200, 'invalid_lease', 'lease_seconds must be in (0,7200]')
        require(sum(not h['released'] for h in self.handles.values()) < self.max_handles,
                'lease_capacity', 'Too many active prefix handles')
        nodes = self.transfers.find_prefix(ids)
        actual = [x for n in nodes for x in n.key.token_ids]
        minimum = max(1, len(ids)-self.cache.page_size)
        require(minimum <= len(actual) <= len(ids) and actual == ids[:len(actual)],
                'cache_miss', 'Complete page-aligned prefix no longer present after build')
        require(len(actual) >= self.cache.load_back_threshold,
                'prefix_too_short', 'Cached prefix shorter than native HiCache transfer threshold')
        require(all(not n.evicted for n in nodes) and not self._native_load_pending(nodes),
                'not_resident', 'Register requires completed resident prefix build')
        handle = self.epoch + '.' + uuid.uuid4().hex
        h = dict(id=handle, requested_hash=token_hash(ids), requested_n=len(ids),
                 ids=actual, hash=token_hash(actual), n=len(actual), endpoint=nodes[-1],
                 host_pins=set(nodes), gpu_hold=False, consumers={}, load=None,
                 backup_requested=False, cancelled=False, released=False,
                 expires_ns=self.clock()+int(ttl*1e9), events=[])
        admission = self._optional_gpu_hold(h, 'register')
        h['headroom_admission'] = admission
        # Native protect_host adds exactly one host ref. Snapshot each counter
        # first, including a callee that increments then raises. Publish only
        # after every own host ref has been established successfully.
        host_before = {}
        try:
            for node in nodes:
                host_before[node] = node.host_ref_counter
                node.protect_host()
            self.handles[handle] = h
            self.emit(h, 'prefix_registered', requested_prefix_tokens=len(ids),
                      cached_prefix_tokens=h['n'], bytes=h['n']*self.transfers.bytes_per_token,
                      requested_prefix_sha256=h['requested_hash'], prefix_sha256=h['hash'],
                      position_base=0, headroom_admission=admission)
            return h
        except BaseException as error:
            self.handles.pop(handle, None)
            rollback_errors = []
            for node, before in reversed(list(host_before.items())):
                try:
                    delta = node.host_ref_counter - before
                    if delta not in (0, 1):
                        raise RuntimeError('Native own host-ref delta differs')
                    if delta:
                        node.release_host()
                except BaseException as rollback_error:
                    rollback_errors.append(type(rollback_error).__name__ + ': ' + str(rollback_error))
            try:
                self._gpu_drop(h)
            except BaseException as rollback_error:
                rollback_errors.append(type(rollback_error).__name__ + ': ' + str(rollback_error))
            if rollback_errors:
                self.registration_poisoned = dict(reason='registration_rollback_failed',
                                                  errors=rollback_errors)
                fatal = RuntimeError('Native prefix ownership poisoned; independent guard cleanup required')
                fatal.requires_guard_cleanup = True
                raise fatal from error
            h['host_pins'].clear()
            raise

    def _backup(self, h):
        require(h['load'] is None, 'transfer_busy', 'Cannot back up during controlled load')
        nodes = self.path(h['endpoint'])
        missing = [n for n in nodes if not n.backuped]
        require(all(not n.evicted for n in missing), 'not_resident', 'Missing source GPU KV')
        need = sum(len(n.key) for n in missing)
        require(self.cache.token_to_kv_pool_host.available_size() >= need,
                'host_capacity', 'Insufficient host slots; no implicit unrelated eviction')
        self._optional_gpu_hold(h, 'backup')
        for node in missing:
            if self.cache.write_backup(node) != len(node.key):
                raise RuntimeError('Native D2H allocation changed after capacity check')
        self._refresh_native_events()
        h['backup_requested'] = True
        self.emit(h, 'd2h_submitted', tokens=need, bytes=need*self.transfers.bytes_per_token,
                  direction='GPU_TO_CPU', existing_host_tokens=h['n']-need)
        if self._host_ready(h):
            self.emit(h, 'd2h_ready', newly_copied_tokens=need)
            h['backup_requested'] = False

    def _reclaim_idle_trailing_leaf(self, h, limit):
        # Opt-in, bounded native-cache eviction; never removes semantic text.
        # A short unused generation suffix can otherwise pin every ancestor.
        endpoint = h['endpoint']
        if not limit or endpoint.lock_ref or len(endpoint.children) != 1:
            return 0
        leaf = next(iter(endpoint.children.values()))
        if (leaf.evicted or leaf.children or leaf.lock_ref or leaf.host_ref_counter
                or len(leaf.key) > limit or self._pending_write([leaf])
                or self._native_load_pending([leaf])):
            return 0
        if leaf.backuped:
            return self.cache._evict_backuped(leaf)
        return self.cache._evict_regular(leaf)

    def _offload(self, h, body):
        require(self._host_ready(h), 'host_not_ready', 'D2H must complete before GPU eviction')
        require(h['load'] is None, 'transfer_busy', 'Prefetch still in progress')
        limit = body.get('reclaim_idle_suffix_tokens', 0)
        require(type(limit) is int and 0 <= limit <= 8192,
                'invalid_suffix_limit', 'Suffix reclamation must be bounded by 8192 tokens')
        self._gpu_drop(h)
        before = self.transfers.free_device()
        extra = self._reclaim_idle_trailing_leaf(h, limit)
        released, retained = extra, []
        for node in reversed(self.path(h['endpoint'])):
            if node.evicted:
                continue
            if node.lock_ref or any(not child.evicted for child in node.children.values()):
                retained.append(dict(node_id=node.id, tokens=len(node.key),
                                     reason='active_or_shared_descendant'))
                continue
            released += self.cache._evict_backuped(node)
        after = self.transfers.free_device()
        if after-before != released:
            raise RuntimeError('Scheduler-owned allocator release delta mismatch')
        self.emit(h, 'gpu_slots_released', released_tokens=released,
                  released_bytes=released*self.transfers.bytes_per_token,
                  allocator_before=before, allocator_after=after, retained=retained,
                  idle_suffix_released_tokens=extra, prefix_released_tokens=released-extra)

    def _prefetch(self, h, body):
        require(not h['cancelled'], 'cancelled', 'Cancelled handle cannot prefetch')
        deadline = body.get('deadline_monotonic_ns')
        require(deadline is None or (type(deadline) is int and deadline > self.clock()),
                'deadline_expired', 'Prefetch admission deadline already passed')
        if h['load'] is not None:
            return
        nodes = self.path(h['endpoint'])
        if all(not n.evicted for n in nodes):
            self._optional_gpu_hold(h, 'prefetch_resident')
            self.emit(h, 'prefetch_already_resident', newly_loaded_tokens=0)
            return
        require(self._host_ready(h), 'host_not_ready', 'Prefetch source is not a completed CPU backup')
        missing = [n for n in nodes if n.evicted]
        need = sum(len(n.key) for n in missing)
        require(missing == nodes[len(nodes)-len(missing):], 'transfer_busy',
                'Optional load requires a contiguous missing suffix')
        require(self.transfers.free_device() >= need, 'gpu_capacity',
                'Insufficient GPU slots; optional prefetch must queue outside engine')
        require(not any(x['load'] is not None for x in self.handles.values()),
                'transfer_busy', 'One controlled H2D at a time; foreground continues')
        # The adapter captures canonical per-token host positions. Splits can
        # occur while DMA runs; publication resolves the current path afterwards.
        ancestor = missing[0].parent
        reservation = self._headroom_reservation('prefetch', h)
        before = self._headroom_accounting('prefetch', h)
        self.cache.inc_lock_ref(ancestor)
        try:
            after = self._headroom_accounting('prefetch', h)
            # Protecting resident ancestors and reserving all missing
            # slots must leave room for previously admitted native work.
            available = after['allocator_free_tokens'] + after['radix_evictable_tokens'] - need
            evidence = dict(reservation=reservation, allocator_before=before, allocator_after=after,
                materialization_tokens=need, allocatable_after_materialization_tokens=available,
                newly_protected_resident_tokens=after['radix_protected_tokens']-before['radix_protected_tokens'])
            if available < reservation['reserve_tokens']:
                self._headroom_reject('prefetch', h, evidence)
            transfer = self.transfers.start_load(missing)
        except BaseException:
            self.cache.dec_lock_ref(ancestor)
            raise
        if transfer is None:
            self.cache.dec_lock_ref(ancestor)
        require(transfer is not None, 'gpu_capacity', 'Native allocator could not reserve slots')
        offsets, offset = [], 0
        for node in nodes:
            if node.evicted:
                offsets.extend(range(offset, offset+len(node.key)))
            offset += len(node.key)
        transfer.update(missing_token_offsets=offsets, tokens=need,
                         deadline_monotonic_ns=deadline, ancestor_pin=ancestor)
        h['load'] = transfer
        self.reserved_device_tokens += need
        self.cache.protected_size_ += need
        self.emit(h, 'h2d_submitted', tokens=need, bytes=need*self.transfers.bytes_per_token,
                  direction='CPU_TO_GPU', values_published=False,
                  deadline_monotonic_ns=deadline)

    @staticmethod
    def _node_at_offset(nodes, offset):
        for node in nodes:
            if offset < len(node.key):
                return node
            offset -= len(node.key)
        raise RuntimeError('Token offset outside prefix')

    def _complete_load(self, h):
        load = h['load']
        if load is None or not self.transfers.ready(load):
            return
        # Event completion establishes all-layer visibility. No global device
        # synchronize; an in-flight copy is never freed or exposed to consumers.
        published, redundant = self.transfers.publish_or_discard(
            load, self.path(h['endpoint']), cancelled=h['cancelled'])
        self.reserved_device_tokens -= load['tokens']
        self.cache.protected_size_ -= load['tokens']
        self.cache.evictable_size_ += published
        h['load'] = None
        if not h['cancelled']:
            self._gpu_hold(h)
        self.cache.dec_lock_ref(load['ancestor_pin'])
        self.emit(h, 'h2d_ready', loaded_tokens=load['tokens'], published_tokens=published,
                  redundant_or_cancelled_tokens=redundant,
                  bytes=load['tokens']*self.transfers.bytes_per_token,
                  dma_elapsed_ms=self.transfers.elapsed_ms(load),
                  deadline_monotonic_ns=load['deadline_monotonic_ns'],
                  missed_deadline=load['deadline_monotonic_ns'] is not None
                  and self.clock() > load['deadline_monotonic_ns'])

    def _cleanup(self, h, active):
        if not h['cancelled']:
            return
        for rid, item in list(h['consumers'].items()):
            if rid not in active and self.clock()-item['acquired_ns'] >= self.submit_grace_ns:
                self.cache.dec_lock_ref(h['endpoint'])
                del h['consumers'][rid]
                self.emit(h, 'inactive_consumer_reaped', consumer_request_id=rid,
                          submit_grace_ns=self.submit_grace_ns)
        if h['load'] is not None or self._pending_write(self.path(h['endpoint'])) or h['consumers']:
            return
        self._gpu_drop(h)
        for node in h['host_pins']:
            node.release_host()
        h['host_pins'].clear()
        h['released'] = True
        self.emit(h, 'handle_released')
        h['ids'] = []
        h['endpoint'] = None

    def tick(self):
        self._refresh_native_events()
        self.cache.writing_check()
        self.cache.loading_check()
        active = set(self.active_rids())
        for h in list(self.handles.values()):
            if h['released']:
                continue
            if self.clock() >= h['expires_ns'] and not h['cancelled']:
                h['cancelled'] = True
                self.emit(h, 'lease_expired')
            self._complete_load(h)
            if h['backup_requested'] and self._host_ready(h):
                h['backup_requested'] = False
                self.emit(h, 'd2h_ready')
            self._cleanup(h, active)
        self.write_events = {k: v for k, v in self.write_events.items() if not v[1].query()}

    def snapshot(self, h):
        if h['released']:
            nodes, host_ready, device_ready = [], False, False
        else:
            nodes = self.path(h['endpoint'])
            host_ready, device_ready = self._host_ready(h), self._device_ready(h)
        pending = []
        if h['load'] is not None:
            pending.append('h2d')
        if nodes and self._pending_write(nodes):
            pending.append('d2h')
        accounting = self.transfers.accounting() if hasattr(self.transfers, 'accounting') else {}
        state = ('RELEASED' if h['released'] else 'CANCEL_PENDING' if h['cancelled']
                 else 'H2D_PENDING' if h['load'] is not None else 'D2H_PENDING' if 'd2h' in pending
                 else 'DEVICE_READY' if device_ready else 'HOST_READY' if host_ready else 'RESIDENT')
        return dict(handle_id=h['id'], service_epoch=self.epoch, state=state,
                    service_profile_sha256=self.profile, prefix_sha256=h['hash'],
                    requested_prefix_sha256=h['requested_hash'],
                    requested_prefix_tokens=h['requested_n'], cached_prefix_tokens=h['n'],
                    uncached_tail_tokens=h['requested_n']-h['n'],
                    device_tokens=sum(len(n.key) for n in nodes if not n.evicted),
                    host_tokens=sum(len(n.key) for n in nodes if n.backuped),
                    device_ready=device_ready, host_ready=host_ready,
                    pending_operations=pending, consumer_refs=len(h['consumers']),
                    controlled_reserved_device_tokens=self.reserved_device_tokens,
                    events=copy.deepcopy(h['events']), **accounting)

    def control(self, body):
        if (getattr(self, 'registration_poisoned', None) or
                getattr(self.transfers, 'ownership_poisoned', None)):
            fatal = RuntimeError('Native prefix ownership poisoned; independent guard cleanup required')
            fatal.requires_guard_cleanup = True
            raise fatal
        request_id, action = body.get('request_id'), body.get('action')
        if not isinstance(request_id, str) or not 0 < len(request_id) <= 256:
            return dict(ok=False, action=action,
                        error=dict(code='invalid_request_id', message='request_id must be a bounded string'))
        digest = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        if request_id in self.requests:
            previous_digest, response = self.requests[request_id]
            if previous_digest == digest:
                return copy.deepcopy(response)
            return dict(ok=False, request_id=request_id, action=action,
                        error=dict(code='idempotency_conflict', message='request_id reused with different payload'))
        h, extra = None, {}
        try:
            require(isinstance(request_id, str) and 0 < len(request_id) <= 256,
                    'invalid_request_id', 'request_id is required and bounded')
            require(body.get('service_profile_sha256') == self.profile,
                    'profile_mismatch', 'Frozen service profile identity differs')
            self.tick()
            if action == 'request_status':
                rid = body.get('consumer_request_id')
                require(isinstance(rid, str) and rid, 'invalid_consumer', 'consumer_request_id required')
                extra.update(active=rid in set(self.active_rids()), consumer_request_id=rid,
                             observed_monotonic_ns=self.clock(), engine_epoch=self.epoch,
                             terminal_proof=False)
            elif action == 'register':
                h = self._register(body)
            else:
                handle = body.get('handle_id')
                require(isinstance(handle, str) and handle.startswith(self.epoch+'.'),
                        'stale_handle', 'Handle belongs to another service epoch')
                h = self.handles.get(handle)
                require(h is not None, 'unknown_handle', 'Unknown prefix handle')
                require(not h['released'] or action in ('status', 'cancel', 'release_consumer'),
                        'released', 'Handle was already released')
                require(not h['cancelled'] or action in ('status', 'cancel', 'release_consumer'),
                        'cancelled', 'Handle is draining cancellation')
                if action == 'fingerprint':
                    import os
                    require(os.environ.get('TOOLSLACK_KV_ENABLE_DIAGNOSTICS') == '1',
                            'diagnostics_disabled', 'Readback only allowed in explicit correctness canary')
                    source = body.get('source')
                    require(source in ('host', 'device'), 'invalid_source', 'source must be host or device')
                    require(self._host_ready(h) if source == 'host' else self._device_ready(h),
                            'not_ready', 'Fingerprint source must be complete and protected')
                    start_ns = self.clock()
                    extra['payload_sha256'] = self.transfers.fingerprint(self.path(h['endpoint']), source)
                    extra['fingerprint_source'] = source
                    self.emit(h, 'diagnostic_readback', source=source,
                              duration_ns=self.clock()-start_ns, correctness_only=True,
                              payload_sha256=extra['payload_sha256'])
                elif action == 'backup':
                    self._backup(h)
                elif action == 'offload':
                    self._offload(h, body)
                elif action == 'prefetch':
                    self._prefetch(h, body)
                elif action == 'acquire':
                    rid, ids = body.get('consumer_request_id'), body.get('input_ids')
                    require(isinstance(rid, str) and rid, 'invalid_consumer', 'consumer_request_id required')
                    require(isinstance(ids, list) and all(type(x) is int and x >= 0 for x in ids)
                            and ids[:h['n']] == h['ids'],
                            'prefix_mismatch', 'Actual consumer does not begin with exact cached prefix')
                    require(self._device_ready(h), 'device_not_ready', 'All-layer GPU-ready lease required')
                    if rid not in h['consumers']:
                        self.cache.inc_lock_ref(h['endpoint'])
                        h['consumers'][rid] = dict(acquired_ns=self.clock(), input_sha256=token_hash(ids))
                        self.emit(h, 'consumer_acquired', consumer_request_id=rid,
                                  actual_input_sha256=token_hash(ids), actual_input_tokens=len(ids))
                    else:
                        require(h['consumers'][rid]['input_sha256'] == token_hash(ids),
                                'consumer_conflict', 'Consumer rid reused with different complete input')
                    extra['consumer_lease_id'] = h['id']+'/'+rid
                elif action == 'release_consumer':
                    rid = body.get('consumer_request_id')
                    require(isinstance(rid, str) and rid, 'invalid_consumer', 'consumer_request_id required')
                    require(rid not in set(self.active_rids()), 'busy', 'Consumer still active in scheduler')
                    if rid in h['consumers']:
                        self.cache.dec_lock_ref(h['endpoint'])
                        del h['consumers'][rid]
                        self.emit(h, 'consumer_released', consumer_request_id=rid)
                elif action == 'cancel':
                    if not h['cancelled']:
                        h['cancelled'] = True
                        self.emit(h, 'cancel_requested')
                elif action != 'status':
                    raise Rejected('unknown_action', 'Unsupported KV action')
            self.tick()
            identity = self.snapshot(h) if h is not None else dict(
                service_epoch=self.epoch, service_profile_sha256=self.profile)
            response = dict(ok=True, request_id=request_id, action=action, **identity, **extra)
        except Rejected as exc:
            response = dict(ok=False, request_id=request_id, action=action,
                            service_epoch=self.epoch, service_profile_sha256=self.profile,
                            error=dict(code=exc.code, message=str(exc)))
            if hasattr(exc, 'headroom'):
                response['headroom_rejection'] = copy.deepcopy(exc.headroom)
            if h is not None:
                response.update(self.snapshot(h))
        self.requests[request_id] = (digest, copy.deepcopy(response))
        while len(self.requests) > 4096:
            self.requests.popitem(last=False)
        return response
