"""Real native KV lifecycle/semantic gate, not an agent-QPS experiment."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time
import urllib.request
import uuid

from calibration.prefix import digest, http, retire


def run(proxy, upstream, service_sha, tokens):
    rows = []
    body = dict(model='Qwen3-8B', messages=[dict(role='user', content=
        'Remember that the secret color is violet.\n' + '\n'.join(
        f'Record {i}: cedar, copper, river, quartz, and violet.' for i in range(tokens // 8 + 64)))],
        max_tokens=32, temperature=0., seed=20260930, stream=False,
        chat_template_kwargs={'enable_thinking': False})
    known = deepcopy(body)
    known['messages'].append(dict(role='assistant', content='', tool_calls=[
        dict(id='color_probe', type='function', function=dict(name='probe', arguments='{}'))]))
    consumer = deepcopy(known)
    consumer['messages'].append(dict(role='tool', tool_call_id='color_probe',
        content='What is the secret color? Answer with the color only.'))

    def control(action, handle, epoch, **fields):
        rid = 'lifecycle:' + uuid.uuid4().hex + ':' + action
        started = time.monotonic()
        status, result = http(proxy, '/kv/control', dict(action=action,
            request_id=rid, handle_id=handle, service_profile_sha256=service_sha, **fields))
        if status != 200 or result.get('ok') is not True or any(result.get(k) != v for k, v in dict(
            action=action, request_id=rid, handle_id=handle,
            service_profile_sha256=service_sha, service_epoch=epoch).items()):
            raise RuntimeError('Native lifecycle control rejected: ' + str(result)[:1000])
        rows.append(dict(action=action, wall_s=time.monotonic()-started, receipt=result))
        return result

    def wait(handle, epoch, field):
        until = time.monotonic() + 180
        while True:
            result = control('status', handle, epoch)
            if result.get(field) is True and result.get('pending_operations') == []:
                return result
            if time.monotonic() >= until:
                raise RuntimeError('Native lifecycle operation did not drain')
            time.sleep(.02)

    def create():
        status, result = http(upstream, '/flush_cache?timeout=60', {})
        if status != 200 or result is False:
            raise RuntimeError('Lifecycle exclusive cache flush failed')
        status, result = http(proxy, '/kv/prefill', dict(body=body, known_future_body=known,
            request_id='lifecycle:' + uuid.uuid4().hex, max_prefix_tokens=tokens, lease_seconds=180))
        reg = result.get('registration') or {}
        ids = result.get('selected_prefix_ids')
        n = reg.get('cached_prefix_tokens')
        if status != 200 or reg.get('ok') is not True or reg.get('service_profile_sha256') != service_sha:
            raise RuntimeError('Lifecycle KV preparation failed')
        if type(n) is not int or not isinstance(ids, list) or not 0 < n <= len(ids) <= tokens or reg.get('prefix_sha256') != digest(ids[:n]):
            raise RuntimeError('Lifecycle exact prefix mismatch')
        return reg

    def infer(handle):
        rid = 'lifecycle:' + uuid.uuid4().hex + ':consumer'
        headers = {'Content-Type': 'application/json', 'x-toolslack-request-kind': 'foreground',
            'x-toolslack-request-id': rid, 'x-toolslack-session-id': 'native-lifecycle-canary',
            'x-toolslack-kv-handle': handle, 'x-toolslack-kv-one-shot': 'true'}
        request = urllib.request.Request(proxy + '/v1/chat/completions',
            json.dumps(consumer).encode(), headers, method='POST')
        with urllib.request.urlopen(request, timeout=180) as response:
            return json.load(response)

    reg = create()
    handle, epoch = reg['handle_id'], reg['service_epoch']
    try:
        resident = infer(handle)
    finally:
        retire(proxy, handle, service_sha, epoch)
    reg = create()
    handle, epoch = reg['handle_id'], reg['service_epoch']
    try:
        before = control('fingerprint', handle, epoch, source='device')
        started = time.monotonic()
        control('backup', handle, epoch)
        wait(handle, epoch, 'host_ready')
        backup_s = time.monotonic() - started
        unloaded = control('offload', handle, epoch)
        unloaded = control('status', handle, epoch)
        if unloaded.get('device_ready') or not unloaded.get('host_ready') or unloaded.get('controlled_reserved_device_tokens') != 0:
            raise RuntimeError('Native offload did not release device ownership')
        freed = unloaded['allocator_free_tokens'] - reg['allocator_free_tokens']
        if freed <= 0:
            raise RuntimeError('Native offload reclaimed no allocator capacity')
        started = time.monotonic()
        control('prefetch', handle, epoch)
        wait(handle, epoch, 'device_ready')
        prefetch_s = time.monotonic() - started
        after = control('fingerprint', handle, epoch, source='device')
        if not before.get('payload_sha256') or before['payload_sha256'] != after.get('payload_sha256'):
            raise RuntimeError('Native KV bytes changed across tiers')
        restored = infer(handle)
        if resident.get('choices') != restored.get('choices'):
            raise RuntimeError('Native-resident/restored greedy outputs differ')
        return dict(state='passed', service_profile_sha256=service_sha,
            cached_prefix_tokens=reg['cached_prefix_tokens'], allocator_tokens_reclaimed=freed,
            prefix_bytes_unchanged=True, resident_restored_greedy_equal=True,
            backup_wall_s=backup_s, prefetch_wall_s=prefetch_s,
            measured_transfer_lead_s=2 * max(backup_s, prefetch_s) + .05,
            reference_kind='native_same_prefix_resident_vs_restored',
            calibration_only=True, task_quality_result=False, task_qps_result=False,
            resident=resident, restored=restored, controls=rows)
    finally:
        retire(proxy, handle, service_sha, epoch)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--proxy', default='http://127.0.0.1:34203')
    parser.add_argument('--upstream', default='http://127.0.0.1:34200')
    parser.add_argument('--service-profile', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--tokens', type=int, default=16384)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    service_sha = hashlib.sha256(args.service_profile.read_bytes()).hexdigest()
    result = run(args.proxy, args.upstream, service_sha, args.tokens)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k not in ('controls', 'resident', 'restored')}), flush=True)


if __name__ == '__main__':
    main()
