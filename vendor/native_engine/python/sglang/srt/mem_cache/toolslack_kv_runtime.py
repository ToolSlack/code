"""Small typed-message integration; all resource mutations stay in scheduler."""
from collections import deque
import hashlib
import os
from pathlib import Path

from sglang.srt.mem_cache.toolslack_kv_control import PrefixManager
from sglang.srt.mem_cache.toolslack_kv_transfers import NativeTransfers
from sglang.srt.mem_cache.toolslack_kv_headroom import foreground_reservation


# Control runs at the beginning of a scheduler iteration. In overlap mode
# result_queue holds launched GPU work whose output processor has not yet run.
# That queue is authoritative even when the shared Req is marked finished by an
# earlier batch. cur/last/running can retain finished Req after results drain.
QUEUED_FIELDS = ('waiting_queue', 'chunked_req', 'grammar_queue', 'grammar_backend_queue')
BATCH_FIELDS = ('running_batch', 'cur_batch', 'last_batch')


def active_rids(scheduler):
    found = set()

    def visit(value, include_finished, seen):
        if value is None or id(value) in seen:
            return
        seen.add(id(value))
        rid = getattr(value, 'rid', None)
        if isinstance(rid, str):
            finished = getattr(value, 'finished', None)
            if include_finished or not callable(finished) or not finished():
                found.add(rid)
            return
        if isinstance(value, dict):
            for item in value.values(): visit(item, include_finished, seen)
        elif isinstance(value, (list, tuple, deque)):
            for item in value: visit(item, include_finished, seen)
        else:
            for attr in ('reqs', 'queue'):
                child = getattr(value, attr, None)
                if isinstance(child, (list, tuple, deque, dict)):
                    visit(child, include_finished, seen)
    # Use distinct seen sets because the same Req can occur in a stale batch and
    # pending GPU output simultaneously; the latter must win.
    for field in QUEUED_FIELDS + ('result_queue',):
        visit(getattr(scheduler, field, None), True, set())
    for field in BATCH_FIELDS:
        visit(getattr(scheduler, field, None), False, set())
    return found


def get_manager(scheduler):
    cache = scheduler.tree_cache
    manager = getattr(cache, 'toolslack_kv_manager', None)
    if manager is not None:
        return manager
    args = scheduler.server_args
    if args.tp_size != 1 or args.dp_size != 1 or args.pp_size != 1:
        raise ValueError('system_kv_v1 supports TP=DP=PP=1 only')
    if not scheduler.spec_algorithm.is_none():
        raise ValueError('system_kv_v1 target-only exact token prefixes required')
    if str(args.disaggregation_mode) not in ('null', 'none', 'None'):
        raise ValueError('system_kv_v1 disaggregated serving unsupported')
    path = os.environ.get('TOOLSLACK_KV_PROFILE_PATH')
    if not path or not Path(path).is_absolute():
        raise ValueError('TOOLSLACK_KV_PROFILE_PATH must name the frozen guard profile')
    profile = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    manager = PrefixManager(cache, NativeTransfers(cache), profile,
                            active_rids=lambda: active_rids(scheduler),
                            foreground_reservation=lambda: foreground_reservation(scheduler))
    cache.toolslack_kv_manager = manager
    return manager


def handle(scheduler, request):
    from sglang.srt.managers.io_struct import ToolSlackKVControlOutput
    result = get_manager(scheduler).control(request.payload)
    return ToolSlackKVControlOutput(transport_id=request.transport_id, payload=result)


def tick(scheduler):
    manager = getattr(scheduler.tree_cache, 'toolslack_kv_manager', None)
    if manager is not None:
        manager.tick()


def can_flush(scheduler):
    manager = getattr(scheduler.tree_cache, 'toolslack_kv_manager', None)
    if manager is None:
        return True
    manager.tick()
    return not manager.reserved_device_tokens and all(h['released'] for h in manager.handles.values())


def after_flush(scheduler):
    # Resetting while all handles are released invalidates old epoch handles.
    if hasattr(scheduler.tree_cache, 'toolslack_kv_manager'):
        del scheduler.tree_cache.toolslack_kv_manager
