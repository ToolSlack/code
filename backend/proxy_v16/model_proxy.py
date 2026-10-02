#!/usr/bin/env python3
"""Profiled local proxy with arrival, dispatch, generated/visible SSE timing.

No GPU operations on import. Request kind must be explicit for QPS attribution.
Streaming first-generated-token includes reasoning/tool tokens; visible content
is reported separately. Nonstream responses never invent a TTFT measurement.
"""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import threading
import time
import uuid
from server_profile import ServerProfile
from family_profile import load_profile, make_counter, template_kwargs
from cancellation_owner import CancellationOwner, shield_owned
from admission import AdmissionQueue
from stable_prefix import stable_known_request_prefix
from ttft_gain_http import TTFTGainHTTP
from transport_terminal import confirm_foreground_terminal, validate_wait_seconds
from prefix_budget import PrefixBudget, PrefixCost, BudgetDeclined, finite_number
from prefill_deadline import (
    PrefillTerminalUnconfirmed,
    run_deadline_bound_prefill,
)


# No upstream error status alone proves pre-admission rejection. A source-bound
# terminal proof is required before any future exception can be registered.
PRE_ADMISSION_TERMINAL_HTTP = frozenset()
RETRYABLE_KV_BUSY_CODES = frozenset({'busy', 'handle_busy', 'transfer_busy'})


KINDS = {'foreground', 'memory', 'kv_prefill', 'warmup', 'unknown'}
PRIORITIES = {'foreground': 100, 'memory_required': 100, 'memory': 0,
              'kv_prefill': -10, 'warmup': 100}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def sha(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def priority_body(body, kind, required=False):
    """System-only ordering; generation/content parameters are left intact.

    All experimental arms use this same mapping. Native required compaction and
    foreground calls have equal protection. External memory ordering is managed
    by the benefit scheduler, rather than accepting arbitrary caller priorities.
    """
    key = 'memory_required' if kind == 'memory' and required else kind
    if key not in PRIORITIES:
        raise ValueError('An explicit known request kind is required')
    expected = PRIORITIES[key]
    if 'priority' in body and (type(body['priority']) is not int or body['priority'] != expected):
        raise ValueError('Priority differs from the frozen system mapping')
    return dict(body, priority=expected)


def normalize(body, profile=None):
    profile = profile or ServerProfile()
    body = dict(body)
    if body.get('model') not in (None, profile.model_name):
        raise ValueError('Model differs from the frozen profile')
    body['model'] = profile.model_name
    body['chat_template_kwargs'] = template_kwargs(body.get('chat_template_kwargs'), profile)
    maximum = body.get('max_tokens', body.get('max_completion_tokens', profile.output_reserve))
    if type(maximum) is not int or not 0 < maximum <= profile.output_reserve:
        raise ValueError('Output exceeds the frozen reserve or is invalid')
    body.pop('max_completion_tokens', None)
    body['max_tokens'] = maximum
    n = body.get('n', 1)
    if n is not None and (type(n) is not int or n != 1):
        raise ValueError('This measured single-consumer profile requires n=1')
    if body.get('stream'):
        body['stream_options'] = {**(body.get('stream_options') or {}), 'include_usage': True}
    if not isinstance(body.get('messages'), list) or not body['messages']:
        raise ValueError('Complete nonempty messages required')
    return body


def token_kinds(delta):
    delta = delta or {}
    kinds = set()
    if delta.get('content'):
        kinds.add('visible')
    if delta.get('reasoning_content'):
        kinds.add('reasoning')
    if any((call.get('function') or {}).get('name') or
           (call.get('function') or {}).get('arguments') for call in delta.get('tool_calls') or []):
        kinds.add('tool')
    return kinds


def first_delta(delta):
    return bool(token_kinds(delta))


def latency_fields(arrival_ns, sent_ns, first_ns, visible_ns):
    return dict(
        proxy_arrival_to_first_token_seconds=(first_ns-arrival_ns)/1e9 if first_ns else None,
        upstream_send_to_first_token_seconds=(first_ns-sent_ns)/1e9 if first_ns else None,
        proxy_arrival_to_first_visible_seconds=(visible_ns-arrival_ns)/1e9 if visible_ns else None,
        proxy_pre_dispatch_seconds=(sent_ns-arrival_ns)/1e9)


def validate_usage(usage, measured, maximum):
    if not isinstance(usage, dict) or usage.get('prompt_tokens') != measured['input_tokens']:
        raise ValueError('Actual model prompt usage differs from exact token count')
    n = usage.get('completion_tokens')
    if type(n) is not int or not 0 <= n <= maximum:
        raise ValueError('Invalid completion token usage')
    if usage.get('total_tokens') != usage['prompt_tokens'] + n:
        raise ValueError('Inconsistent actual total token usage')


def validate_terminal(*, seen_done, finished_choices, usage, measured, body):
    if not seen_done or finished_choices != {0}:
        raise ValueError('SSE ended without DONE and an explicit finish reason')
    validate_usage(usage, measured, body['max_tokens'])


async def serve(args):
    import aiohttp
    from aiohttp import web
    profile = load_profile(args.profile)
    exact_counter = make_counter(profile)
    tokenizer = exact_counter.tokenizer
    # The shared fast tokenizer is protected from concurrent mutable truncation
    # setup. This cost remains included in arrival TTFT and separately reported.
    count_lock = threading.Lock()
    capacity = dict(profile.capacity(),
        model_name=profile.model_name,
        model_profile_sha256=sha(profile.__dict__),
        tokenizer_sha256=hashlib.sha256((Path(profile.model_path)/'tokenizer.json').read_bytes()).hexdigest(),
        chat_template_sha256=hashlib.sha256(str(tokenizer.chat_template).encode()).hexdigest())
    capacity['native_preprocessing_source_sha256'] = getattr(exact_counter, 'source_hashes', {})
    kv_profile = getattr(args, 'kv_service_profile', None)
    kv_service_sha = hashlib.sha256(kv_profile.read_bytes()).hexdigest() if kv_profile else None
    capacity['kv_service_profile_sha256'] = kv_service_sha
    cost_path = getattr(args, 'kv_cost_profile', None)
    prefix_cost = PrefixCost.load(cost_path, kv_service_sha) if cost_path else None
    capacity['kv_cost_profile_sha256'] = prefix_cost.source_sha if prefix_cost else None
    kv_max_prefills = getattr(args, 'kv_max_prefills', 1)
    if type(kv_max_prefills) is not int or not 1 <= kv_max_prefills <= 16:
        raise ValueError('Invalid KV prefill concurrency limit')
    capacity['kv_max_prefills'] = kv_max_prefills
    kv_prefill_gate = asyncio.Semaphore(kv_max_prefills)
    prefill_terminal_wait = getattr(args, 'kv_prefill_terminal_wait', None)
    if prefill_terminal_wait is None:
        prefill_terminal_wait = min(float(profile.request_timeout_seconds), 1800.0)
    if (type(prefill_terminal_wait) not in (int, float)
            or not 0.05 <= float(prefill_terminal_wait) <= 1800):
        raise ValueError('Invalid KV prefill terminal confirmation interval')
    capacity['kv_prefill_terminal_wait_seconds'] = float(prefill_terminal_wait)
    kv_runtime_failed = False
    prefill_slots = {}
    unconfirmed_prefill_tasks = {}
    owned_requests = {}
    submitted_request_ids = set()
    candidate_rows = {}
    candidate_tombstones = {}
    # A confirmed logical-handle retirement is remembered so the framework
    # side may perform an idempotent drain check after the proxy-owned
    # terminal release.  This record does not flush or own the engine's
    # natural radix cache, which may remain reusable by unrelated requests.
    one_shot_releases = {}
    # A one-shot consumer owns the handle lifecycle once its foreground
    # request is admitted.  Framework drain calls may observe this state, but
    # must not race the proxy by issuing a second native cancel.
    one_shot_claims = {}
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output/'proxy_capacity.json').open('x') as f:
        f.write(json.dumps(capacity, indent=2)+'\n')
    log = (args.output/'proxy_events.jsonl').open('x', buffering=1)

    def event(kind, **data):
        stamp = time.monotonic_ns()
        log.write(json.dumps(dict(event=kind, monotonic_ns=stamp, wall_time=time.time(), **data), ensure_ascii=False)+'\n')
        return stamp

    ttft_history = TTFTGainHTTP(
        emit=event,
        request_started=lambda request_id: (
            request_id in owned_requests or request_id in submitted_request_ids
        ),
    )

    def count(body, include_ids=False):
        queued = time.monotonic_ns()
        with count_lock:
            started = time.monotonic_ns()
            ids = exact_counter.token_ids(body)
            n = len(ids)
            closed = exact_counter.token_ids(body, add_generation_prompt=False) if include_ids else None
            if closed is not None and ids[:len(closed)] != closed:
                raise ValueError('Closed serialization is not an exact prefix of native request')
            finished = time.monotonic_ns()
        result = dict(capacity, input_tokens=n, at_or_above_trigger=n >= profile.trigger_tokens,
                    hard_input_overflow=n > profile.max_input_tokens,
                    payload_sha256=hashlib.sha256(canonical(body).encode()).hexdigest(),
                    exact_count_queue_seconds=(started-queued)/1e9,
                    exact_count_seconds=(finished-started)/1e9)
        if include_ids:
            result.update(input_ids=ids, input_ids_sha256=sha(ids),
                          closed_prefix_ids=closed, closed_prefix_ids_sha256=sha(closed),
                          position_base=0, future_consumer_match_required=True)
        return result

    trace_config = aiohttp.TraceConfig()
    async def headers_sent(session, context, params):
        info = getattr(context, 'trace_request_ctx', None)
        if info and 'common' in info:
            info['timing']['headers_sent_ns'] = event('upstream_request_headers_sent', **info['common'])
    trace_config.on_request_headers_sent.append(headers_sent)
    client = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(
        total=profile.request_timeout_seconds, connect=15), trust_env=False,
        connector=aiohttp.TCPConnector(limit=128, limit_per_host=128, force_close=True), trace_configs=[trace_config])

    async def counter(request):
        try:
            result = await asyncio.to_thread(count, normalize(await request.json(), profile))
        except (ValueError, TypeError, KeyError) as exc:
            return web.json_response({'error': str(exc)}, status=400)
        event('count_tokens', request_id=request.headers.get('x-toolslack-request-id'),
              session_id=request.headers.get('x-toolslack-session-id'), **result)
        return web.json_response(result)

    async def exact_tokens(request):
        try:
            body = normalize(await request.json(), profile)
            result = await asyncio.to_thread(count, body, True)
        except (ValueError, TypeError, KeyError) as exc:
            return web.json_response({'error': str(exc)}, status=400)
        event('exact_tokens', request_id=request.headers.get('x-toolslack-request-id'),
              **{k: v for k, v in result.items() if k not in ('input_ids', 'closed_prefix_ids')})
        return web.json_response(result)

    async def engine_control(body):
        if kv_service_sha is None:
            raise ValueError('This proxy has no bound KV service profile')
        body = dict(body)
        supplied = body.get('service_profile_sha256', kv_service_sha)
        if supplied != kv_service_sha:
            raise ValueError('KV service profile identity differs')
        body['service_profile_sha256'] = kv_service_sha
        if body.get('action') not in {'register', 'backup', 'offload', 'prefetch',
                                     'status', 'acquire', 'release_consumer', 'cancel', 'request_status', 'fingerprint'}:
            raise ValueError('Unknown KV action')
        if not isinstance(body.get('request_id'), str) or not body['request_id']:
            raise ValueError('A unique KV operation request ID is required')
        started = event('kv_control_sent', action=body['action'], request_id=body['request_id'],
                        handle_id=body.get('handle_id'), control_payload_sha256=sha(body))
        async with client.post(args.upstream+'/toolslack/kv/control', json=body,
                               timeout=aiohttp.ClientTimeout(total=30)) as upstream:
            result = await upstream.json()
            if upstream.status != 200:
                raise RuntimeError('KV control HTTP error: '+str(upstream.status)+' '+str(result)[:1000])
        if result.get('service_profile_sha256') != kv_service_sha:
            raise RuntimeError('KV control response profile binding differs')
        if result.get('request_id') != body['request_id']:
            raise RuntimeError('KV response request identity differs')
        if result.get('action') != body['action']:
            raise RuntimeError('KV response action differs')
        if body.get('handle_id') and result.get('handle_id') != body['handle_id']:
            raise RuntimeError('KV response handle identity differs')
        event('kv_control_finished', action=body['action'], request_id=body['request_id'],
              handle_id=result.get('handle_id'), receipt=result,
              control_seconds=(time.monotonic_ns()-started)/1e9)
        return result

    async def retire_one_shot_handle(handle_id, request_id):
        """Retire a one-consumer handle only after exact terminal/reference proof."""
        previous = one_shot_releases.get(handle_id)
        if previous is not None:
            return dict(previous, request_id=request_id + ':retire',
                        action='cancel', idempotent=True)
        deadline = time.monotonic() + 30
        attempt = 0
        while True:
            if handle_id in one_shot_claims:
                one_shot_claims[handle_id]['state'] = 'retiring'
            suffix = ':retire' if attempt == 0 else ':retire:' + str(attempt)
            receipt = await engine_control(dict(
                request_id=request_id + suffix,
                action='cancel',
                handle_id=handle_id,
            ))
            released = bool(
                receipt.get('ok') is True
                and receipt.get('state') == 'RELEASED'
                and receipt.get('consumer_refs') == 0
                and receipt.get('pending_operations') == []
            )
            event('kv_one_shot_released' if released else 'kv_one_shot_cleanup_pending',
                  request_id=request_id, handle_id=handle_id,
                  receipt=receipt, cleanup_attempt=attempt)
            if released:
                one_shot_releases[handle_id] = dict(receipt)
                if handle_id in one_shot_claims:
                    one_shot_claims[handle_id]['state'] = 'released'
                return receipt
            error_code = (receipt.get('error') or {}).get('code')
            # Dense and sparse managers use different, but equally
            # retryable, names while a consumer or DMA owns the handle.
            if error_code not in RETRYABLE_KV_BUSY_CODES or time.monotonic() >= deadline:
                raise RuntimeError('One-shot KV handle release unconfirmed; guard required')
            attempt += 1
            await asyncio.sleep(.02)

    async def kv_control(request):
        try:
            body = await request.json()
            handle = body.get('handle_id')
            if body.get('action') == 'cancel' and (handle in one_shot_releases or handle in one_shot_claims):
                # The synthetic reply is still a control-plane receipt.  Apply
                # the same identity/profile checks as engine_control before
                # returning the cached terminal proof; otherwise a missing
                # request ID or a receipt from another service profile could be
                # laundered through the idempotent path.
                if not isinstance(body.get('request_id'), str) or not body['request_id']:
                    raise ValueError('A unique KV operation request ID is required')
                supplied = body.get('service_profile_sha256', kv_service_sha)
                if supplied != kv_service_sha:
                    raise ValueError('KV service profile identity differs')
                if handle in one_shot_releases:
                    prior = one_shot_releases[handle]
                    result = dict(prior, request_id=body['request_id'],
                                  action='cancel', service_profile_sha256=kv_service_sha,
                                  idempotent=True)
                    event_name = 'kv_one_shot_idempotent_release'
                else:
                    claim = one_shot_claims[handle]
                    result = dict(request_id=body['request_id'], action='cancel',
                                  handle_id=handle, service_profile_sha256=kv_service_sha,
                                  service_epoch=claim.get('service_epoch'), ok=False,
                                  prefix_sha256=claim.get('prefix_sha256'),
                                  requested_prefix_sha256=claim.get('requested_prefix_sha256'),
                                  requested_prefix_tokens=claim.get('requested_prefix_tokens'),
                                  cached_prefix_tokens=claim.get('cached_prefix_tokens'),
                                  identity_source=claim.get('identity_source'),
                                  state='DRAINING', consumer_refs=claim.get('consumer_refs', 1),
                                  pending_operations=['proxy_terminal_cleanup'],
                                  error={'code':'handle_busy',
                                         'message':'proxy owns one-shot terminal cleanup'},
                                  idempotent=True, native_cancel_issued=False)
                    event_name = 'kv_one_shot_cleanup_owned_by_proxy'
                event(event_name, request_id=body['request_id'],
                      handle_id=handle, receipt=result)
                return web.json_response(result)
            if kv_runtime_failed and body.get('action') not in {'status', 'request_status', 'cancel', 'release_consumer'}:
                return web.json_response({'error':'Unconfirmed operation; only cleanup/status allowed'},status=503)
            result = await engine_control(body)
            return web.json_response(result)
        except (ValueError, TypeError, KeyError) as exc:
            return web.json_response({'error': str(exc)}, status=400)

    async def kv_prefill(request):
        nonlocal kv_runtime_failed
        spec = await request.json()
        budget = None
        try:
            if 'deadline_unix_ms' in spec:
                budget = PrefixBudget(spec, prefix_cost)
                budget.select()
            elif 'safety_ms' in spec:
                raise ValueError('safety_ms requires deadline_unix_ms')
        except (ValueError, TypeError, KeyError) as exc:
            event('kv_prefill_declined', request_id=spec.get('request_id'), reason=str(exc), native_submitted=False)
            return web.json_response(dict(error=str(exc), optional_declined=True, native_submitted=False), status=400)
        if kv_runtime_failed:
            return web.json_response(
                {'error':'Unconfirmed prior KV operation; run requires guard cleanup'},
                status=503,
            )
        queued = event('kv_prefill_queued', request_id=spec.get('request_id'),
                       maximum_running=kv_max_prefills)
        await kv_prefill_gate.acquire()
        release_slot = True
        rid = spec.get('request_id')
        prefill_slots[rid] = dict(
            request_id=rid,
            state='admitted',
            slot_held=True,
            native_request_id=None,
            deadline_unix_ms=spec.get('deadline_unix_ms'),
            terminal_confirmed=False,
        )
        try:
            if kv_runtime_failed:
                return web.json_response({'error':'Unconfirmed prior KV operation; run requires guard cleanup'},status=503)
            event('kv_prefill_dispatched', request_id=spec.get('request_id'),
                  queue_seconds=(time.monotonic_ns()-queued)/1e9)
            try:
                return await run_kv_prefill(spec, budget)
            except PrefillTerminalUnconfirmed as exc:
                # The exact native request may still execute.  Keep this
                # semaphore permit permanently owned until the independent
                # guard closes the service; an abort HTTP 200 is not release.
                release_slot = False
                kv_runtime_failed = True
                prefill_slots[rid].update(
                    state='native_terminal_unconfirmed',
                    native_request_id=exc.native_rid,
                    terminal_confirmed=False,
                    terminal_audit=exc.audit,
                )
                unconfirmed_prefill_tasks[exc.native_rid] = exc.native_task
                exc.native_task.add_done_callback(
                    lambda task: task.exception() if not task.cancelled() else None
                )
                event('kv_runtime_cleanup_required', request_id=rid,
                      native_request_id=exc.native_rid,
                      reason='deadline_abort_terminal_unconfirmed',
                      prefill_slot_released=False)
                return web.json_response({
                    'error':'Native prefix-prefill terminal state unconfirmed; guard cleanup required',
                    'native_submitted':True,
                    'native_terminal_confirmed':False,
                    'prefill_slot_released':False,
                }, status=503)
            except BaseException:
                # Transport loss does not prove GPU completion. Do not admit
                # another KV prefill against the same nominal resource slot.
                kv_runtime_failed = True
                event('kv_runtime_cleanup_required', request_id=spec.get('request_id'))
                raise
        finally:
            if release_slot:
                kv_prefill_gate.release()
                prefill_slots.pop(rid, None)

    async def run_kv_prefill(spec, budget=None):
        """Build only already known, exactly serialized closed-message tokens."""
        nonlocal kv_runtime_failed
        try:
            rid = spec['request_id']
            if not isinstance(rid, str) or not rid or kv_service_sha is None:
                raise ValueError('KV build needs request identity and a bound service')
            if budget is not None:
                budget.select()  # Charge semaphore queueing before exact serialization.
            body = normalize(spec['body'], profile)
            known_future_body = normalize(spec['known_future_body'], profile)
            measured = await asyncio.to_thread(count, body, True)
            ids = measured['closed_prefix_ids']
            def select_stable():
                with count_lock:
                    return stable_known_request_prefix(
                        body, known_future_body, ids,
                        lambda future: exact_counter.token_ids(
                            future, add_generation_prompt=False
                        ),
                        future_position_prefix=getattr(exact_counter, 'future_position_prefix', None),
                    )
            ids, stable_scope = await asyncio.to_thread(select_stable)
            if not ids or len(ids) > profile.max_input_tokens:
                raise ValueError('Known prefix is empty or over hard input capacity')
            lease = spec.get('lease_seconds', 600)
            if type(lease) not in (int, float) or not 1 <= lease <= 1800:
                raise ValueError('Invalid KV handle lease duration')
            limit = spec.get('max_prefix_tokens', len(ids))
            if type(limit) is not int or limit <= 0:
                raise ValueError('max_prefix_tokens must be a positive exact integer')
            bounded_tokens = min(len(ids), limit)
            budget_selection = budget.select(bounded_tokens) if budget is not None else None
            if budget_selection is not None:
                ids = ids[:budget_selection['selected_prefix_tokens']]
                event('kv_budget_selected', request_id=rid, **budget_selection)
            else:
                ids = ids[:bounded_tokens]
        except BudgetDeclined as exc:
            event('kv_prefill_declined', request_id=spec.get('request_id'), reason=str(exc), native_submitted=False)
            return web.json_response(dict(error=str(exc), optional_declined=True, native_submitted=False), status=400)
        except (ValueError, TypeError, KeyError) as exc:
            return web.json_response({'error': str(exc)}, status=400)
        native = dict(input_ids=ids, sampling_params={'max_new_tokens': 0, 'temperature': 0},
                      stream=False, priority=PRIORITIES['kv_prefill'], rid=rid+':prefill')
        prefill_slots[rid].update(
            state='native_submitted',
            native_request_id=native['rid'],
        )
        started = event('kv_prefill_sent', request_id=rid, request_kind='kv_prefill',
                        native_payload_sha256=sha(native), payload_sha256=measured['payload_sha256'],
                        prefix_sha256=sha(ids), prefix_tokens=len(ids),
                        known_context_payload=body, model_profile_sha256=capacity['model_profile_sha256'])
        event('kv_stable_scope', request_id=rid, **stable_scope,
              known_future_body_sha256=sha(known_future_body))
        def exact_prefill_terminal(status, generated):
            meta = generated.get('meta_info', {}) if isinstance(generated, dict) else {}
            return bool(
                status == 200
                and meta.get('id') == native['rid']
                and meta.get('prompt_tokens') == len(ids)
                and meta.get('completion_tokens') == 0
            )

        outcome = await run_deadline_bound_prefill(
            client=client,
            upstream=args.upstream,
            native=native,
            deadline_mono=budget.deadline_mono if budget is not None else None,
            terminal_wait_seconds=float(prefill_terminal_wait),
            engine_control=engine_control,
            emit=event,
            validate_http_terminal=exact_prefill_terminal,
        )
        generated = outcome.body
        if outcome.deadline_expired:
            prefill_slots[rid].update(
                state='deadline_terminal_confirmed',
                terminal_confirmed=True,
                terminal_proof=outcome.proof,
            )
            event('kv_prefill_deadline_terminal_confirmed', request_id=rid,
                  native_request_id=native['rid'], proof=outcome.proof,
                  engine_epoch=outcome.engine_epoch,
                  status_attempts=outcome.status_attempts,
                  abort_attempts=outcome.abort_attempts,
                  prefill_slot_released=True,
                  registration_attempted=False)
            return web.json_response({
                'error':'Optional prefix-prefill exceeded the tool deadline',
                'optional_declined':True,
                'native_submitted':True,
                'native_terminal_confirmed':True,
                'terminal_proof':outcome.proof,
                'registration_attempted':False,
            }, status=408)
        if outcome.status != 200 or not outcome.terminal_confirmed:
            known_terminal = outcome.terminal_confirmed
            if not known_terminal:
                kv_runtime_failed = True
                event('kv_runtime_cleanup_required', request_id=rid,
                      reason='native_prefill_http_terminal_unconfirmed',
                      status=outcome.status)
            event('kv_prefill_failed', request_id=rid, status=outcome.status,
                  response=generated, upstream_terminal_observed=known_terminal)
            return web.json_response({'error': 'Native prefill failed',
                'native_terminal_confirmed':known_terminal}, status=502)
        meta = generated.get('meta_info', {})
        if (meta.get('id') != native['rid'] or meta.get('prompt_tokens') != len(ids)
            or meta.get('completion_tokens') != 0):
            raise RuntimeError('Native prefill token/zero-generation receipt differs')
        event('kv_prefill_finished', request_id=rid, receipt=generated,
              prefill_seconds=(time.monotonic_ns()-started)/1e9)
        registered = await engine_control(dict(request_id=rid+':register', action='register',
            input_ids=ids, prefix_sha256=sha(ids), position_base=0, lease_seconds=lease))
        if registered.get('ok'):
            covered = registered.get('cached_prefix_tokens')
            if (type(covered) is not int or not 0 < covered <= len(ids) or
                registered.get('requested_prefix_tokens') != len(ids) or
                registered.get('requested_prefix_sha256') != sha(ids) or
                registered.get('prefix_sha256') != sha(ids[:covered])):
                raise RuntimeError('Registered full/page-aligned prefix identity differs from native build')
        return web.json_response(dict(prefill=generated, registration=registered,
            selected_prefix_ids=ids, max_prefix_tokens=spec.get('max_prefix_tokens'),
            requested_prefix_sha256=sha(ids), requested_prefix_tokens=len(ids),
            prefix_sha256=registered.get('prefix_sha256'),
            cached_prefix_tokens=registered.get('cached_prefix_tokens'),
            known_payload_sha256=measured['payload_sha256'],
            service_profile_sha256=kv_service_sha,
            stable_scope=stable_scope,
            budget_selection=budget_selection,
            completed_after_deadline=(time.monotonic() > budget.deadline_mono) if budget else None,
            future_consumer_match_required=True))

    async def register_existing_prefix(request):
        """Pin only already resident exact KV; never perform optional generation.

        This path is independent of a native-memory candidate. It can reclaim
        tool-wait HBM for an unchanged L0 context when a caller subsequently
        performs the existing backup/offload/prefetch lifecycle.
        """
        nonlocal kv_runtime_failed
        if kv_runtime_failed:
            return web.json_response({'error': 'Unconfirmed operation; guard cleanup required'}, status=503)
        try:
            spec = await request.json()
            rid = spec['request_id']
            if not isinstance(rid, str) or not rid or kv_service_sha is None:
                raise ValueError('Existing KV registration requires request and service identity')
            deadline = None
            if 'deadline_unix_ms' in spec:
                deadline = time.monotonic() + finite_number(spec['deadline_unix_ms'], 'deadline', True)/1000 - time.time()
                deadline -= finite_number(spec.get('safety_ms', 0), 'safety')/1000
                if time.monotonic() >= deadline:
                    raise ValueError('Existing KV registration tool window expired')
            body = normalize(spec['body'], profile)
            future = normalize(spec['known_future_body'], profile)
            measured = await asyncio.to_thread(count, body, True)
            def exact_stable():
                with count_lock:
                    return stable_known_request_prefix(body, future, measured['closed_prefix_ids'],
                        lambda value: exact_counter.token_ids(value, add_generation_prompt=False),
                        future_position_prefix=getattr(exact_counter, 'future_position_prefix', None))
            ids, stable_scope = await asyncio.to_thread(exact_stable)
            if not ids or len(ids) > profile.max_input_tokens:
                raise ValueError('Existing stable prefix is empty or over capacity')
            lease = spec.get('lease_seconds', 600)
            if type(lease) not in (int, float) or not 1 <= lease <= 1800:
                raise ValueError('Invalid existing KV lease')
            if deadline is not None and time.monotonic() >= deadline:
                raise ValueError('Existing KV serialization exhausted tool window')
        except (ValueError, KeyError, TypeError) as exc:
            return web.json_response({'error': str(exc), 'native_submitted': False}, status=400)
        # A cache miss is an optional no-op. The native allocator alone decides
        # which exact leading tokens are still resident; no model call repairs it.
        try:
            registered = await engine_control(dict(action='register', request_id=rid+':register',
                input_ids=ids, prefix_sha256=sha(ids), position_base=0, lease_seconds=lease))
            if registered.get('ok') is True:
                covered = registered.get('cached_prefix_tokens')
                if (type(covered) is not int or not 0 < covered <= len(ids)
                        or registered.get('requested_prefix_tokens') != len(ids)
                        or registered.get('requested_prefix_sha256') != sha(ids)
                        or registered.get('prefix_sha256') != sha(ids[:covered])):
                    raise RuntimeError('Existing KV identity differs from native cached prefix')
        except BaseException:
            kv_runtime_failed = True
            event('kv_runtime_cleanup_required', request_id=rid, reason='existing_registration_unconfirmed')
            raise
        event('kv_existing_prefix_registered', request_id=rid, registration=registered,
            full_stable_prefix_tokens=len(ids), native_generate_submitted=False,
            semantic_context_changed=False, stable_scope=stable_scope)
        return web.json_response(dict(registration=registered,
            existing_prefix_only=True, native_generate_submitted=False,
            semantic_context_changed=False, optional_cache_miss=not registered.get('ok'),
            requested_prefix_sha256=sha(ids), requested_prefix_tokens=len(ids),
            prefix_sha256=registered.get('prefix_sha256'),
            cached_prefix_tokens=registered.get('cached_prefix_tokens'),
            known_payload_sha256=measured['payload_sha256'], service_profile_sha256=kv_service_sha,
            stable_scope=stable_scope, future_consumer_match_required=True))

    memory_workers = getattr(args, 'optional_memory_workers', 0)
    if type(memory_workers) is not int or not 0 <= memory_workers <= 16:
        raise ValueError('optional memory workers outside [0,16]')
    memory_gate = AdmissionQueue(max_running=max(1, memory_workers), emit=event)
    transport_terminal_wait = validate_wait_seconds(
        getattr(args, 'transport_terminal_wait', 5.0)
    )

    def optional_spec(request, owner, as_of_monotonic_ns):
        if not memory_workers or owner.common['request_kind'] != 'memory':
            return None
        if request.headers.get('x-toolslack-memory-required') == 'true':
            return None
        raw = request.headers.get('x-toolslack-maintenance')
        if not raw or len(raw) > 4096:
            raise ValueError('optional memory needs bounded scheduling metadata')
        spec = json.loads(raw)
        if 'expected_relative_ttft_gain' in spec:
            raise ValueError('caller-supplied TTFT gain is forbidden; explicit history is authoritative')
        if spec.get('clock_domain') != 'same-host-wall':
            raise ValueError('unsupported tool deadline clock domain')
        deadline = spec['deadline_unix_s']
        if type(deadline) not in (float, int) or not __import__('math').isfinite(deadline):
            raise ValueError('finite deadline required')
        history_estimate = ttft_history.estimate_for_maintenance(
            raw_query=spec.get('ttft_gain_query'),
            common=owner.common,
            as_of_monotonic_ns=as_of_monotonic_ns,
            actual_model_profile_hash=capacity['model_profile_sha256'],
        )
        cost_source = spec['estimate_source']
        if not isinstance(cost_source, str) or not cost_source:
            raise ValueError('pipeline-cost estimate provenance is required')
        benefit_source = history_estimate['receipt_sha256'] or 'unknown-neutral'
        # Convert once on receipt; count/queue time subsequently consumes this budget.
        return dict(request_id=owner.common['request_id'],
            session_id=owner.common['session_id'], snapshot_id=spec['snapshot_id'],
            deadline=time.monotonic() + deadline - time.time() - 0.1,
            remaining_cost_s=spec['remaining_pipeline_cost_s'],
            expected_relative_ttft_gain=history_estimate['scheduler_value'],
            estimate_source=f"cost:{cost_source};ttft:{benefit_source}")

    async def optional_admission(spec, owner):
        ticket = memory_gate.submit(**spec)
        while ticket.state == 'pending':
            if kv_runtime_failed:
                memory_gate.cancel(ticket.request_id, reason='native_state_unconfirmed')
                return ticket
            if owner.disconnected() or owner.cancelled_at is not None:
                memory_gate.cancel(ticket.request_id)
                return ticket
            memory_gate.dispatch()
            if ticket.state == 'pending':
                await asyncio.sleep(.005)
        return ticket

    def candidate_snapshot(candidate_id):
        rows = list((candidate_rows.get(candidate_id) or {}).values())
        exists = candidate_id in candidate_rows or candidate_id in candidate_tombstones
        terminal = bool(candidate_id in candidate_tombstones) and all(
            row.get('terminal_confirmed') is True for row in rows)
        state = ('terminal' if terminal else 'unconfirmed'
                 if any(row.get('state') == 'unconfirmed' for row in rows)
                 else 'cancel_requested' if candidate_id in candidate_tombstones
                 else 'active' if exists else 'unknown')
        return dict(candidate_id=candidate_id, exists=exists, terminal=terminal,
                    state=state, cancelled=candidate_id in candidate_tombstones,
                    budget_releasable=terminal,
                    requests=[dict(row) for row in rows])

    def mark_candidate_terminal(common, reason):
        candidate_id = common.get('candidate_id')
        if not candidate_id:
            return
        row = candidate_rows[candidate_id][common['request_id']]
        row.update(state='terminal', terminal_reason=reason,
                   terminal_at=time.monotonic(), terminal_confirmed=True)

    async def completion_owned(request, owner):
        nonlocal kv_runtime_failed
        if kv_runtime_failed:
            return web.json_response({'error':'Unconfirmed KV operation; run requires guard cleanup'},status=503)
        arrival = time.monotonic_ns()
        common = owner.common
        event('request_arrived', **common, arrival_monotonic_ns=arrival)
        try:
            if common['request_kind'] not in KINDS:
                raise ValueError('Unknown request-kind header')
            maintenance_spec = optional_spec(request, owner, arrival)
            body = normalize(await request.json(), profile)
            wire_body = priority_body(body, common['request_kind'],
                                      request.headers.get('x-toolslack-memory-required') == 'true')
            measured = await asyncio.to_thread(count, body)
        except (ValueError, TypeError, KeyError) as exc:
            event('request_invalid', **common, error=str(exc))
            return web.json_response({'error': {'message': str(exc), 'type': 'invalid_request_error'}}, status=400)
        event('request_payload', **common, payload=body,
              system_priority=wire_body['priority'], **measured)
        if measured['hard_input_overflow']:
            event('hard_input_rejected', **common, **measured)
            return web.json_response({'error': {'message': 'Verified maximum input capacity exceeded',
                'type': 'invalid_request_error', 'code': 'context_length_exceeded'}}, status=400)
        if owner.disconnected() or owner.cancelled_at is not None:
            owner.cancel('downstream_gone_before_native_submission')
            event('request_not_submitted', **common)
            return web.json_response({'error': 'consumer disconnected before submission'}, status=499)
        sent = None
        first = visible = None
        kv_handle = request.headers.get('x-toolslack-kv-handle')
        kv_one_shot = request.headers.get('x-toolslack-kv-one-shot')
        if kv_one_shot not in {None, 'true'} or (kv_one_shot == 'true' and not kv_handle):
            return web.json_response({'error': {'message': 'Invalid one-shot KV binding',
                'type': 'invalid_request_error'}}, status=400)
        kv_one_shot = kv_one_shot == 'true'
        if kv_one_shot and common['request_kind'] != 'foreground':
            return web.json_response({'error': {'message': 'One-shot KV is foreground-only',
                'type': 'invalid_request_error'}}, status=400)
        if kv_one_shot:
            prior_claim = one_shot_claims.get(kv_handle)
            if prior_claim is not None and prior_claim.get('consumer_request_id') != common['request_id']:
                return web.json_response({'error': {'message': 'One-shot KV already has a consumer owner',
                    'type': 'invalid_request_error'}}, status=409)
            one_shot_claims[kv_handle] = dict(
                consumer_request_id=common['request_id'], state='claimed',
                service_epoch=None, consumer_refs=0,
            )
        acquired = False
        one_shot_retired = False
        upstream_terminal = False
        transport_timing = {}
        maintenance_ticket = None
        try:
            if maintenance_spec is not None:
                try:
                    maintenance_ticket = await optional_admission(maintenance_spec, owner)
                except (ValueError, OverflowError) as exc:
                    return web.json_response({'error': {'type': 'invalid_request_error', 'code': 'toolslack_admission_denied', 'message': str(exc)}, 'native_submitted': False}, status=400)
                if kv_runtime_failed:
                    return web.json_response({'error':'Native terminal unconfirmed; guard cleanup required'}, status=503)
                if maintenance_ticket.state != 'running':
                    return web.json_response({'error': {'type': 'invalid_request_error', 'code': 'toolslack_optional_window_closed', 'message': 'Optional maintenance expired or cancelled before native admission'}, 'native_submitted': False}, status=400)
                if owner.disconnected() or owner.cancelled_at is not None:
                    return web.json_response({'error': 'optional consumer no longer active', 'native_submitted': False}, status=499)
            if 'rid' in wire_body and wire_body['rid'] != common['request_id']:
                raise ValueError('Native request ID differs from the observed foreground ID')
            wire_body['rid'] = common['request_id']
            event('wire_request', **common, wire_payload=wire_body, wire_payload_sha256=sha(wire_body))
            if kv_handle:
                exact = await asyncio.to_thread(count, body, True)
                receipt = await engine_control(dict(request_id=common['request_id']+':acquire',
                    action='acquire', handle_id=kv_handle,
                    consumer_request_id=common['request_id'], input_ids=exact['input_ids']))
                acquired = receipt.get('ok') is True
                if kv_one_shot:
                    one_shot_claims[kv_handle].update(
                        state='acquired' if acquired else 'acquire_rejected',
                        service_epoch=receipt.get('service_epoch'),
                        prefix_sha256=receipt.get('prefix_sha256'),
                        requested_prefix_sha256=receipt.get('requested_prefix_sha256'),
                        requested_prefix_tokens=receipt.get('requested_prefix_tokens'),
                        cached_prefix_tokens=receipt.get('cached_prefix_tokens'),
                        identity_source='validated_native_acquire_receipt',
                        consumer_refs=1 if acquired else 0,
                    )
                event('kv_consumer_acquired' if acquired else 'kv_consumer_fallback',
                      **common, handle_id=kv_handle, receipt=receipt)
                # A definitive rejection is safe: normal model inference can
                # compute the exact new context without claiming a KV hit.
                if kv_one_shot and not acquired:
                    await retire_one_shot_handle(kv_handle, common['request_id'])
                    one_shot_retired = True
            # Counting/acquiring await CPU/control work. Another in-flight
            # request may have poisoned this service during those waits.
            if kv_runtime_failed:
                raise RuntimeError('Run poisoned before native model submission; guard required')
            sent = event('request_sent', **common, boundary='client_post_attempt_includes_pool_queue')
            owner.submitted = True
            if common.get('candidate_id'):
                candidate_rows[common['candidate_id']][common['request_id']].update(
                    state='submitted', submitted_at=time.monotonic()
                )
            submitted_request_ids.add(common['request_id'])
            async with client.post(args.upstream+'/v1/chat/completions', json=wire_body,
                trace_request_ctx=dict(common=common, timing=transport_timing)) as upstream:
                event('upstream_headers', **common, status=upstream.status)
                if upstream.status != 200:
                    data = await upstream.read()
                    upstream_terminal = upstream.status in PRE_ADMISSION_TERMINAL_HTTP
                    if not upstream_terminal:
                        kv_runtime_failed = True
                        event('kv_runtime_cleanup_required', **common,
                              reason='native_http_terminal_unconfirmed',status=upstream.status)
                    event('request_failed', **common, status=upstream.status, error=data.decode(errors='replace')[:3000],
                          upstream_terminal_observed=upstream_terminal)
                    # Do not forward a native 4xx as a proxy pre-admission 4xx:
                    # callers may otherwise free their own model permits early.
                    return web.json_response({'error':{'type':'unconfirmed_upstream',
                        'code':'toolslack_guard_required',
                        'message':'Native terminal state unconfirmed; guard cleanup required'},
                        'upstream_status':upstream.status},status=502)
                if not body.get('stream'):
                    data = await upstream.json()
                    if data.get('model') != profile.model_name:
                        raise ValueError('Nonstream response model differs')
                    if data.get('id') != common['request_id']:
                        raise ValueError('Nonstream response does not identify the owned native request')
                    choices = data.get('choices') or []
                    if len(choices) != 1 or choices[0].get('finish_reason') is None:
                        raise ValueError('Nonstream response has no explicit terminal choice')
                    validate_usage(data.get('usage'), measured, body['max_tokens'])
                    upstream_terminal = True
                    owner.confirm_terminal()
                    mark_candidate_terminal(common, choices[0].get('finish_reason'))
                    event('nonstream_response', **common, usage=data.get('usage'), response=data)
                    event('request_finished', **common, stream=False, ttft_observable=False,
                          consumer_cancelled=owner.cancelled_at is not None,
                          usage=data.get('usage'), **latency_fields(arrival, sent, None, None))
                    return web.json_response(data)
                response = web.StreamResponse(status=200, headers={
                    'Content-Type': 'text/event-stream', 'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})
                await owner.prepare(response, request)
                usage = None
                buffer = b''
                chunks = []
                seen_done = False
                finished_choices = set()
                stream_identity = None
                async for chunk in upstream.content.iter_any():
                    received = time.monotonic_ns()
                    buffer += chunk
                    outgoing = bytearray()
                    while b'\n' in buffer:
                        line, buffer = buffer.split(b'\n', 1)
                        if not line.startswith(b'data:'):
                            if not seen_done:
                                outgoing.extend(line+b'\n')
                            continue
                        raw = line[5:].strip()
                        if not raw:
                            outgoing.extend(line+b'\n')
                            continue
                        if raw == b'[DONE]':
                            seen_done = True
                            continue
                        if seen_done:
                            raise ValueError('Unexpected SSE data after DONE')
                        try:
                            item = json.loads(raw)
                        except json.JSONDecodeError as exc:
                            raise ValueError('Malformed SSE JSON') from exc
                        if item.get('error') or item.get('model') != profile.model_name:
                            raise ValueError('SSE error or response model mismatch')
                        if not isinstance(item.get('id'), str) or not item['id']:
                            raise ValueError('SSE response identity is missing')
                        if item['id'] != common['request_id']:
                            raise ValueError('SSE response does not identify the owned native request')
                        if stream_identity is None:
                            stream_identity = item['id']
                        elif stream_identity != item['id']:
                            raise ValueError('SSE response identity changed within a stream')
                        for choice in item.get('choices') or []:
                            if choice.get('index') != 0:
                                raise ValueError('Unexpected SSE choice index')
                            if choice.get('finish_reason') is not None:
                                finished_choices.add(0)
                        if item.get('usage'):
                            usage = item['usage']
                        chunks.append(item)
                        kinds = set().union(*(token_kinds(c.get('delta')) for c in item.get('choices') or []))
                        if first is None and kinds:
                            first = received
                            first_latency = latency_fields(arrival, sent, first, visible)
                            event('first_token', **common, received_monotonic_ns=received, kinds=sorted(kinds),
                                  **first_latency)
                            ttft_history.observe_first_token(
                                common=common,
                                received_monotonic_ns=received,
                                proxy_arrival_to_first_token_seconds=first_latency[
                                    'proxy_arrival_to_first_token_seconds'
                                ],
                            )
                        if visible is None and 'visible' in kinds:
                            visible = received
                            event('first_visible_token', **common, received_monotonic_ns=received,
                                  **latency_fields(arrival, sent, first, visible))
                        outgoing.extend(line+b'\n')
                    # Observe upstream first-token arrival before backpressure.
                    # Withhold DONE until native EOF and terminal/usage proof
                    # pass, so a client cannot release its permit prematurely.
                    if outgoing:
                        await owner.write(response, bytes(outgoing))
                if buffer.strip():
                    raise ValueError('Truncated SSE line at transport EOF')
                validate_terminal(seen_done=seen_done, finished_choices=finished_choices,
                                  usage=usage, measured=measured, body=body)
                upstream_terminal = True
                owner.confirm_terminal()
                mark_candidate_terminal(common, 'validated_stream_terminal')
                await owner.write(response, b'data: [DONE]\n\n')
                await owner.write(response, b'', eof=True)
                event('response_chunks', **common, chunks=chunks, usage=usage)
                event('request_finished', **common, stream=True, usage=usage,
                      consumer_cancelled=owner.cancelled_at is not None,
                      ttft_observable=first is not None, first_token_seen=first is not None,
                      tokenizer_prompt_tokens=measured['input_tokens'],
                      prompt_count_matches_server=(usage.get('prompt_tokens') == measured['input_tokens']) if usage else None,
                      upstream_headers_sent_to_first_token_seconds=(first-transport_timing['headers_sent_ns'])/1e9
                          if first and transport_timing.get('headers_sent_ns') else None,
                      **latency_fields(arrival, sent, first, visible))
                return response
        except asyncio.CancelledError:
            event('request_cancelled', **common)
            raise
        except Exception as exc:
            event('request_failed', **common, error=f'{type(exc).__name__}: {exc}')
            is_transport_failure = isinstance(
                exc, (aiohttp.ClientConnectionError, ConnectionError, asyncio.TimeoutError)
            )
            if (is_transport_failure and common['request_kind'] == 'foreground'
                    and owner.submitted):
                try:
                    recovery = await confirm_foreground_terminal(
                        rid=common['request_id'], common=common, owner=owner,
                        engine_control=engine_control, emit=event,
                        wait_seconds=transport_terminal_wait,
                    )
                except Exception as recovery_exc:
                    recovery = {
                        'terminal_confirmed': False,
                        'proof': None,
                        'error_type': type(recovery_exc).__name__,
                        'error': str(recovery_exc)[:500],
                    }
                if recovery.get('terminal_confirmed') is True:
                    upstream_terminal = True
                    owner.confirm_terminal()
                    event('native_transport_terminal_confirmed', **common,
                          recovery=recovery, original_error_type=type(exc).__name__,
                          request_still_failed=True, model_request_retried=False)
                else:
                    event('native_transport_terminal_unconfirmed', **common,
                          recovery=recovery, original_error_type=type(exc).__name__,
                          request_still_failed=True, model_request_retried=False)
            raise
        finally:
            if not upstream_terminal and (sent is not None or acquired):
                kv_runtime_failed = True
                event('kv_runtime_cleanup_required', **common,
                      reason='native_terminal_unconfirmed_no_further_model_admission')
            if acquired:
                if upstream_terminal:
                    # HTTP EOF plus fully validated 200 completion proves
                    # GPU terminal. Upstream error statuses do not. Scheduler
                    # bookkeeping may still briefly report busy. Only this
                    # proven-terminal path retries release, never inference.
                    deadline=time.monotonic()+30
                    attempt=0
                    try:
                        while True:
                            suffix=':release' if attempt==0 else ':release:'+str(attempt)
                            receipt = await engine_control(dict(request_id=common['request_id']+suffix,
                                action='release_consumer', handle_id=kv_handle,
                                consumer_request_id=common['request_id']))
                            event('kv_consumer_release' if receipt.get('ok') else 'kv_consumer_cleanup_pending',
                                  **common, handle_id=kv_handle, receipt=receipt,release_attempt=attempt)
                            if receipt.get('ok'):break
                            if (receipt.get('error') or {}).get('code') not in {'busy','consumer_active'} or time.monotonic()>=deadline:
                                raise RuntimeError('Proven-terminal consumer release unconfirmed; guard required')
                            attempt+=1
                            await asyncio.sleep(.02)
                        if kv_one_shot:
                            await retire_one_shot_handle(kv_handle, common['request_id'])
                            one_shot_retired = True
                    except BaseException:
                        kv_runtime_failed=True
                        event('kv_runtime_cleanup_required', **common,
                              reason='consumer_release_unconfirmed_after_proven_terminal')
                        raise
                else:
                    # Do not mistake a cancelled HTTP coroutine for completed
                    # GPU work. Keep its consumer reference until engine-owned
                    # cleanup confirms quiescence, or the guard closes service.
                    event('kv_consumer_cleanup_pending', **common, handle_id=kv_handle,
                          reason='upstream terminal state not observed; reference retained')
            elif kv_one_shot and kv_handle and not one_shot_retired:
                # The only normal path here is a definitive acquire rejection
                # already retired above.  Any other path is an ownership bug.
                kv_runtime_failed = True
                event('kv_runtime_cleanup_required', **common, handle_id=kv_handle,
                      reason='one_shot_handle_not_retired')

            if maintenance_ticket is not None and maintenance_ticket.state == 'running':
                if upstream_terminal and owner.terminal:
                    memory_gate.acknowledge_terminal(maintenance_ticket.request_id,
                        terminal_request_id=common['request_id'], terminal_complete=True,
                        native_quiescent=True, reason='validated_native_terminal')
                elif not owner.submitted:
                    memory_gate.abandon_before_native_submission(maintenance_ticket.request_id,
                        native_submitted=False)
                # Otherwise the reservation remains held and existing cleanup poisons the run.
            candidate_id = common.get('candidate_id')
            if candidate_id:
                row = candidate_rows[candidate_id][common['request_id']]
                if not owner.submitted and not row.get('terminal_confirmed'):
                    row.update(state='terminal', terminal_reason='never_submitted_to_native_engine',
                               terminal_at=time.monotonic(), terminal_confirmed=True)
                elif owner.submitted and not upstream_terminal:
                    row.update(state='unconfirmed', terminal_reason='native_terminal_unconfirmed',
                               terminal_at=None, terminal_confirmed=False)

    async def completion(request):
        common = dict(request_id=request.headers.get('x-toolslack-request-id') or str(uuid.uuid4()),
                      agent=request.headers.get('x-toolslack-agent', 'unknown'),
                      request_kind=request.headers.get('x-toolslack-request-kind', 'unknown'),
                      session_id=request.headers.get('x-toolslack-session-id'),
                      cohort_id=request.headers.get('x-toolslack-cohort-id'),
                      arm=request.headers.get('x-toolslack-arm'),
                      task_id=request.headers.get('x-toolslack-task-id'),
                      candidate_id=request.headers.get('x-toolslack-candidate-id'))
        rid = common['request_id']
        if rid in owned_requests or rid in submitted_request_ids:
            return web.json_response({'error': 'native request ID was already used'}, status=409)
        candidate_id = common['candidate_id']
        toolslack_rid = request.headers.get('x-toolslack-native-request-id')
        toolslack_kind = request.headers.get('x-toolslack-native-request-kind')
        if any((candidate_id, toolslack_rid, toolslack_kind)):
            if (not candidate_id or toolslack_rid != rid or toolslack_kind != 'native-memory'
                    or common['request_kind'] != 'memory'):
                return web.json_response({'error': 'incomplete or conflicting native-memory ownership'}, status=400)
            if candidate_id in candidate_tombstones:
                return web.json_response({'error': 'candidate was already cancelled'}, status=409)
            if rid in candidate_rows.setdefault(candidate_id, {}):
                return web.json_response({'error': 'candidate request identity was already used'}, status=409)
            candidate_rows[candidate_id][rid] = dict(
                rid=rid, candidate_id=candidate_id, state='registered',
                created_at=time.monotonic(), submitted_at=None, terminal_at=None,
                terminal_reason=None, terminal_confirmed=False,
                abort_enqueued=0, cancel_at=None,
            )
        owner = CancellationOwner(request, client, args.upstream, common, event)

        async def run():
            owner.start()
            try:
                return await completion_owned(request, owner)
            finally:
                await owner.finish()
                owned_requests.pop(rid, None)

        worker = asyncio.create_task(run())
        owned_requests[rid] = (owner, worker)
        # Retrieve detached exceptions after downstream cancellation while
        # preserving the worker's own structured failure/guard logs.
        worker.add_done_callback(lambda task: task.exception() if not task.cancelled() else None)
        return await shield_owned(worker, owner)

    async def candidate_cancel(request):
        candidate_id = request.match_info['candidate_id']
        if not candidate_id or len(candidate_id) > 180:
            return web.json_response({'error': 'invalid candidate identity'}, status=400)
        payload = await request.json()
        candidate_tombstones.setdefault(candidate_id, time.monotonic())
        for rid, row in (candidate_rows.get(candidate_id) or {}).items():
            row['cancel_at'] = row.get('cancel_at') or time.monotonic()
            pair = owned_requests.get(rid)
            if pair is not None:
                owner, worker = pair
                owner.cancel(str(payload.get('reason', 'tool_result_ready'))[:120])
                row['abort_enqueued'] = owner.abort_attempts
        deadline = time.monotonic() + getattr(args, 'candidate_terminal_wait', 5.0)
        snapshot = candidate_snapshot(candidate_id)
        while not snapshot['terminal'] and time.monotonic() < deadline:
            await asyncio.sleep(.02)
            for rid, row in (candidate_rows.get(candidate_id) or {}).items():
                pair = owned_requests.get(rid)
                if pair is not None:
                    row['abort_enqueued'] = pair[0].abort_attempts
            snapshot = candidate_snapshot(candidate_id)
        return web.json_response(snapshot, status=200 if snapshot['terminal'] else 202)

    async def candidate_status(request):
        snapshot = candidate_snapshot(request.match_info['candidate_id'])
        return web.json_response(snapshot, status=200 if snapshot['exists'] else 404)

    async def models(request):
        return web.json_response({'object': 'list', 'data': [
            {'id': profile.model_name, 'object': 'model', 'owned_by': 'local'}]})

    async def health(request):
        return web.json_response({'status': 'tokenizer_ready', 'capacity': capacity, 'profile': profile.__dict__})

    async def toolslack_capabilities(request):
        return web.json_response(dict(
            bounded_prefix_prefill=kv_service_sha is not None,
            deadline_terminal_drain=True,
            existing_prefix_registration=kv_service_sha is not None,
            service_profile_sha256=kv_service_sha,
            implementation='v15_explicit_prefix_cap_v1'))

    async def lifecycle_status(request):
        # A workload ending is not proof its cancelled native requests ended.
        # Drivers must drain these owners before cache reset or arm handoff;
        # that cleanup wall time belongs to the arm which created the work.
        rows = [dict(request_id=rid, request_kind=owner.common['request_kind'],
                     submitted=owner.submitted, terminal_observed=owner.terminal,
                     cancel_requested=owner.cancelled_at is not None,
                     cleanup_timed_out=owner.cleanup_timed_out)
                for rid, (owner, worker) in owned_requests.items()]
        return web.json_response(dict(guard_cleanup_required=kv_runtime_failed,
            active_owners=len(rows), quiescent=not rows and not kv_runtime_failed,
            requests=rows,
            optional_prefix_prefills=list(prefill_slots.values()),
            held_prefill_slots=sum(1 for row in prefill_slots.values()
                                   if row.get('slot_held')),
            optional_memory_admission=memory_gate.status()))

    async def cleanup(app):
        owners = list(owned_requests.values())
        for owner, worker in owners:
            owner.cancel('proxy_shutdown')
        if owners:
            await asyncio.gather(*(worker for owner, worker in owners), return_exceptions=True)
        native_prefills = list(unconfirmed_prefill_tasks.values())
        for task in native_prefills:
            task.cancel()
        if native_prefills:
            await asyncio.gather(*native_prefills, return_exceptions=True)
        await client.close()
        log.close()

    app = web.Application(client_max_size=32*1024*1024)
    app.router.add_post('/count_tokens', counter)
    app.router.add_post('/exact_tokens', exact_tokens)
    app.router.add_post('/kv/control', kv_control)
    app.router.add_post('/kv/prefill', kv_prefill)
    app.router.add_post('/kv/register-existing', register_existing_prefix)
    app.router.add_post('/v1/chat/completions', completion)
    app.router.add_post('/toolslack/v1/candidates/{candidate_id}/cancel', candidate_cancel)
    app.router.add_get('/toolslack/v1/candidates/{candidate_id}', candidate_status)
    ttft_history.install(app)
    app.router.add_get('/v1/models', models)
    app.router.add_get('/health', health)
    app.router.add_get('/toolslack/capabilities', toolslack_capabilities)
    app.router.add_get('/lifecycle/status', lifecycle_status)
    app.on_cleanup.append(cleanup)
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    await web.TCPSite(runner, '127.0.0.1', args.port).start()
    event('proxy_ready', **capacity, port=args.port)
    try:
        await asyncio.Event().wait()
    finally:
        await runner.cleanup()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--port', type=int, default=32100)
    parser.add_argument('--upstream', default='http://127.0.0.1:32000')
    parser.add_argument('--kv-service-profile', type=Path)
    parser.add_argument('--kv-cost-profile', type=Path)
    parser.add_argument('--kv-max-prefills', type=int, default=1)
    parser.add_argument('--optional-memory-workers', type=int, default=0)
    parser.add_argument('--candidate-terminal-wait', type=float, default=5.0)
    parser.add_argument('--transport-terminal-wait', type=float, default=5.0)
    parser.add_argument('--kv-prefill-terminal-wait', type=float)
    asyncio.run(serve(parser.parse_args()))
