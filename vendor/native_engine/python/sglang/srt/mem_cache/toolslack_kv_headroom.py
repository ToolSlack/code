"""CPU-only reservation of already-admitted native requests.

Called in the scheduler owner thread. No GPU synchronization, tensor reads,
request mutation or scheduling changes. Unadmitted queues are deliberately not
reserved; existing native admission already sees optional cache protection.
"""
from collections import deque


SCHEMA = "native-admitted-request-headroom.v1"
ADMITTED_FIELDS = ("chunked_req", "running_batch", "cur_batch", "last_batch", "result_queue")


def foreground_reservation(scheduler):
    cache = scheduler.tree_cache
    page = getattr(cache, "page_size", None)
    if type(page) is not int or page <= 0:
        return dict(schema=SCHEMA, supported=False, reason="invalid_native_page_size")
    requests = {}
    conflict = False

    def visit(value, seen):
        nonlocal conflict
        if value is None or id(value) in seen:
            return
        seen.add(id(value))
        rid = getattr(value, "rid", None)
        if isinstance(rid, str):
            finished = getattr(value, "finished", None)
            if not callable(finished):
                conflict = True
                return
            if finished():
                # Pending output can still own allocated GPU slots. Those slots
                # remain in allocator accounting, but need no future reservation.
                return
            if rid in requests and requests[rid] is not value:
                conflict = True
            else:
                requests[rid] = value
            return
        if isinstance(value, dict):
            for item in value.values():
                visit(item, seen)
        elif isinstance(value, (list, tuple, deque)):
            for item in value:
                visit(item, seen)
        else:
            for name in ("reqs", "queue"):
                child = getattr(value, name, None)
                if isinstance(child, (list, tuple, deque, dict)):
                    visit(child, seen)

    try:
        for field in ADMITTED_FIELDS:
            visit(getattr(scheduler, field, None), set())
        if conflict:
            return dict(schema=SCHEMA, supported=False, reason="ambiguous_admitted_request")
        rows = []
        for rid, req in sorted(requests.items()):
            ids = getattr(req, "origin_input_ids", None)
            allocated = getattr(req, "kv_allocated_len", None)
            maximum = getattr(getattr(req, "sampling_params", None), "max_new_tokens", None)
            freed = getattr(req, "kv_committed_freed", None)
            if (not isinstance(ids, (list, tuple)) or type(allocated) is not int or allocated < 0
                    or type(maximum) is not int or maximum < 0 or freed is not False):
                return dict(schema=SCHEMA, supported=False, reason="unknown_admitted_allocation", request_id=rid)
            # kv_allocated_len is set to seq_len by prepare_for_extend and is
            # incremented for decode allocation. It already includes input,
            # output and any overallocated slots: do not reserve them twice.
            target = len(ids) + maximum
            remaining = max(0, target - allocated)
            reserve = ((remaining + page - 1) // page) * page
            rows.append(dict(request_id=rid, origin_input_tokens=len(ids),
                output_cap=maximum, kv_allocated_tokens=allocated,
                remaining_tokens=remaining, reserved_tokens=reserve))
        return dict(schema=SCHEMA, supported=True, page_size=page,
            admitted_request_count=len(rows), reserve_tokens=sum(row["reserved_tokens"] for row in rows),
            admitted_requests=rows)
    except (AttributeError, TypeError, ValueError, RuntimeError):
        return dict(schema=SCHEMA, supported=False, reason="admitted_snapshot_unavailable")
