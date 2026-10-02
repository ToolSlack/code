"""Native MHA HiCache copy primitives, owned exclusively by scheduler thread.

No model forward, summary, token deletion, or external GPU-memory manipulation.
The private H2D stream avoids exposing native load-back's pending node values and
does not consume the engine's three foreground LayerDoneCounter event slots.
"""


class NativeTransfers:
    def __init__(self, cache):
        from sglang.srt.mem_cache.hiradix_cache import HiRadixCache
        from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
        if not isinstance(cache, HiRadixCache) or not isinstance(cache.kv_cache, MHATokenToKVPool):
            raise ValueError('system_kv_v1 supports native HiRadix MHA only')
        if cache.tp_world_size != 1 or cache.enable_storage:
            raise ValueError('system_kv_v1 initial backend requires TP1, native L3 disabled')
        ctl = cache.cache_controller
        if ctl.io_backend != 'direct' or cache.token_to_kv_pool_host.layout != 'layer_first':
            raise ValueError('system_kv_v1 requires native direct layer_first transfers')
        if ctl.write_policy != 'write_back':
            raise ValueError('system_kv_v1 requires native write_back policy')
        self.cache = cache
        self.stream = None
        pool = cache.kv_cache
        per_token = pool.k_buffer[0][0]
        self.bytes_per_token = 2 * pool.layer_num * per_token.numel() * per_token.element_size()

    def find_prefix(self, ids):
        from sglang.srt.mem_cache.radix_cache import RadixKey
        c = self.cache
        n = len(ids)//c.page_size*c.page_size
        if not n:
            return []
        _, endpoint = c._match_prefix_helper(c.root_node, RadixKey(ids[:n], extra_key=None))
        nodes = []
        while endpoint is not c.root_node:
            nodes.append(endpoint)
            endpoint = endpoint.parent
        return list(reversed(nodes))

    def free_device(self):
        return int(self.cache.cache_controller.mem_pool_device_allocator.available_size())

    def start_load(self, nodes):
        import torch
        from types import SimpleNamespace
        if getattr(self, 'ownership_poisoned', None):
            fatal = RuntimeError('Native transfer ownership poisoned; independent guard cleanup required')
            fatal.requires_guard_cleanup = True
            raise fatal
        c, ctl = self.cache, self.cache.cache_controller
        device_indices = None
        streams = []
        try:
            host_indices = torch.cat([node.host_value for node in nodes])
            device_indices = ctl.mem_pool_device_allocator.alloc(len(host_indices))
            if device_indices is None:
                return None
            # Exception-only synchronization below covers the source/default
            # stream as well as our dedicated copy stream. No normal-path
            # synchronize is introduced.
            streams.append(torch.cuda.current_stream(device=c.device))
            if self.stream is None:
                self.stream = torch.cuda.Stream(device=c.device)
            streams.append(self.stream)
            copy_host, copy_device = ctl.move_indices(SimpleNamespace(
                host_indices=host_indices, device_indices=device_indices))
            source_ready = torch.cuda.Event()
            started = torch.cuda.Event(enable_timing=True)
            finished = torch.cuda.Event(enable_timing=True)
            source_ready.record()
            with torch.cuda.stream(self.stream):
                source_ready.wait(self.stream)
                started.record()
                for layer in range(ctl.layer_num):
                    ctl.mem_pool_host.load_to_device_per_layer(
                        ctl.mem_pool_device, copy_host, copy_device, layer, ctl.io_backend)
                finished.record()
                if copy_host.is_cuda:
                    copy_host.record_stream(self.stream)
                if copy_device.is_cuda:
                    copy_device.record_stream(self.stream)
            return dict(device_indices=device_indices, host_indices=host_indices,
                        copy_host_indices=copy_host, copy_device_indices=copy_device,
                        start_event=started, finish_event=finished, source_event=source_ready)
        except BaseException as error:
            if device_indices is None:
                raise
            sync_errors = []
            if not streams:
                try:
                    streams.append(torch.cuda.current_stream(device=c.device))
                except BaseException as sync_error:
                    sync_errors.append(type(sync_error).__name__ + ': ' + str(sync_error))
            seen = set()
            for stream in streams:
                if id(stream) in seen:
                    continue
                seen.add(id(stream))
                try:
                    stream.synchronize()
                except BaseException as sync_error:
                    sync_errors.append(type(sync_error).__name__ + ': ' + str(sync_error))
            if not sync_errors:
                try:
                    ctl.mem_pool_device_allocator.free(device_indices)
                except BaseException as free_error:
                    sync_errors.append('free: ' + type(free_error).__name__ + ': ' + str(free_error))
            if sync_errors:
                # Retain slots rather than freeing while accepted DMA may use
                # them. This error is deliberately NOT Rejected: the native
                # control handler cannot turn poisoned ownership into success.
                self.ownership_poisoned = dict(reason='transfer_rollback_unconfirmed',
                    reserved_tokens=len(device_indices), errors=sync_errors,
                    requires_guard_cleanup=True)
                fatal = RuntimeError('Native transfer ownership poisoned; independent guard cleanup required')
                fatal.requires_guard_cleanup = True
                raise fatal from error
            raise

    def ready(self, transfer):
        return bool(transfer['finish_event'].query())

    def elapsed_ms(self, transfer):
        return float(transfer['start_event'].elapsed_time(transfer['finish_event']))

    def publish_or_discard(self, transfer, nodes, cancelled):
        c = self.cache
        if not self.ready(transfer):
            raise RuntimeError('Attempted publication/free before native DMA completion')
        # Absolute token positions bind the reserved values across radix splits.
        positions = {offset: i for i, offset in enumerate(transfer['missing_token_offsets'])}
        offset, published, released = 0, 0, 0
        for node in nodes:
            length = len(node.key)
            overlap = offset in positions
            if overlap:
                start = positions[offset]
                if positions.get(offset+length-1) != start+length-1:
                    raise RuntimeError('Radix segment crosses incompatible transfer allocation')
                values = transfer['device_indices'][start:start+length]
                if not cancelled and node.evicted:
                    node.value = values
                    published += length
                else:
                    c.cache_controller.mem_pool_device_allocator.free(values)
                    released += length
            offset += length
        if published+released != transfer['tokens']:
            raise RuntimeError('Reserved KV slot accounting mismatch')
        return published, released

    def accounting(self):
        c = self.cache
        total = int(c.cache_controller.mem_pool_device_allocator.size)
        free = self.free_device()
        protected, evictable = int(c.protected_size_), int(c.evictable_size_)
        return dict(allocator_total_tokens=total, allocator_free_tokens=free,
                    radix_protected_tokens=protected, radix_evictable_tokens=evictable,
                    other_allocated_tokens=total-free-protected-evictable)

    def fingerprint(self, nodes, source):
        # Canary correctness-only explicit readback; never a timed KV transfer.
        import hashlib
        import os
        import torch
        if os.environ.get('TOOLSLACK_KV_ENABLE_DIAGNOSTICS') != '1':
            raise ValueError('KV diagnostic readback is disabled')
        c, digest = self.cache, hashlib.sha256()
        if source == 'device':
            indices = torch.cat([n.value for n in nodes])
            layers = c.kv_cache.get_cpu_copy(indices)
            tensors = (torch.cat([chunk[kv] for chunk in chunks], dim=0)
                       for chunks in layers for kv in (0, 1))
        else:
            indices = torch.cat([n.host_value for n in nodes]).cpu().long()
            host = c.token_to_kv_pool_host
            tensors = (host.kv_buffer[kv, layer, indices]
                       for layer in range(host.layer_num) for kv in (0, 1))
        for tensor in tensors:
            digest.update(str((tuple(tensor.shape), str(tensor.dtype))).encode())
            digest.update(tensor.contiguous().view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()
