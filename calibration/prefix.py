"""Measure same-service bounded zero-generation KV preparation.

These neutral synthetic prompts calibrate serving cost only. They are never
tool workloads, task-quality examples, or throughput benchmark observations.
Each sample drains its native handle before the next cache flush.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time
import urllib.error
import urllib.request
import uuid


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def http(url, path, body=None, timeout=180):
    req = urllib.request.Request(url.rstrip('/') + path,
        data=None if body is None else json.dumps(body).encode(),
        headers={'Content-Type': 'application/json'}, method='GET' if body is None else 'POST')
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            raw = response.read()
            try:
                value = json.loads(raw)
            except json.JSONDecodeError:
                # Some native /flush_cache revisions return plain text. Control
                # receipts still fail their required structured identity checks.
                value = {'raw_response': raw.decode('utf-8', errors='replace')}
            return response.status, value
    except urllib.error.HTTPError as error:
        return error.code, json.load(error)


def retire(proxy, handle, service_sha, epoch):
    until = time.monotonic() + 180
    while True:
        rid = 'calibration:' + uuid.uuid4().hex + ':cancel'
        status, result = http(proxy, '/kv/control', dict(action='cancel',
            request_id=rid, handle_id=handle, service_profile_sha256=service_sha))
        if status != 200 or any(result.get(k) != v for k, v in dict(action='cancel',
                request_id=rid, handle_id=handle, service_profile_sha256=service_sha,
                service_epoch=epoch).items()):
            raise RuntimeError('Calibration cleanup identity is unconfirmed')
        if result.get('ok') is True and result.get('state') == 'RELEASED' and result.get('consumer_refs') == 0 and result.get('pending_operations') == []:
            return result
        if time.monotonic() >= until:
            raise RuntimeError('Calibration handle did not drain')
        time.sleep(.02)


def measure(proxy, upstream, model, service_sha, count, repetition):
    # A new first-message nonce and a native cache flush prevent another
    # calibration sample from silently turning this into a warm-cost curve.
    status, flushed = http(upstream, '/flush_cache?timeout=60', {})
    if status != 200 or flushed is False:
        raise RuntimeError('Exclusive calibration cache flush declined')
    nonce = uuid.uuid4().hex
    content = 'Serving-cost calibration ' + nonce + '.\n' + '\n'.join(
        f'Record {i}: cedar, copper, river, quartz, and violet.' for i in range(count // 8 + 64))
    body = dict(model=model, messages=[dict(role='user', content=content)],
        max_tokens=64, temperature=0, stream=False,
        chat_template_kwargs={'enable_thinking': False})
    future = deepcopy(body)
    future['messages'].append(dict(role='assistant', content='', tool_calls=[
        dict(id='calibration_probe', type='function', function=dict(name='probe', arguments='{}'))]))
    rid = 'calibration:' + nonce + ':kv'
    started = time.monotonic()
    status, result = http(proxy, '/kv/prefill', dict(body=body, known_future_body=future,
        request_id=rid, max_prefix_tokens=count, lease_seconds=180))
    elapsed = (time.monotonic() - started) * 1000
    registration = result.get('registration') or {}
    handle, epoch = registration.get('handle_id'), registration.get('service_epoch')
    try:
        if status != 200 or registration.get('ok') is not True:
            raise RuntimeError('Native calibration prefix failed: ' + str(result)[:1200])
        ids = result.get('selected_prefix_ids')
        covered = registration.get('cached_prefix_tokens')
        expected = dict(action='register', request_id=rid + ':register',
            service_profile_sha256=service_sha, requested_prefix_tokens=count,
            requested_prefix_sha256=digest(ids), device_ready=True)
        if not isinstance(ids, list) or len(ids) != count or any(registration.get(k) != v for k, v in expected.items()):
            raise RuntimeError('Calibrated exact token/profile identity differs')
        if type(covered) is not int or not 0 < covered <= count or registration.get('prefix_sha256') != digest(ids[:covered]):
            raise RuntimeError('Calibrated cached prefix identity differs')
        meta = (result.get('prefill') or {}).get('meta_info') or {}
        if meta.get('id') != rid + ':prefill' or meta.get('prompt_tokens') != count or meta.get('completion_tokens') != 0:
            raise RuntimeError('Calibration requires exact zero-generation terminal proof')
        return dict(tokens=count, repetition=repetition, wall_ms=elapsed,
            cached_prefix_tokens=covered, source='native_proxy_zero_generation_plus_registration',
            native_receipt=result, task_quality_observation=False,
            benchmark_observation=False)
    finally:
        if handle and epoch:
            retire(proxy, handle, service_sha, epoch)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--proxy', default='http://127.0.0.1:34203')
    parser.add_argument('--upstream', default='http://127.0.0.1:34200')
    parser.add_argument('--service-profile', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', default='Qwen3-8B')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--knots', nargs='+', type=int,
        default=[32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384])
    args = parser.parse_args()
    if args.repeats < 2 or any(x <= 0 for x in args.knots):
        parser.error('At least two repeats and positive exact token knots required')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    raw_path = args.output.with_suffix('.samples.jsonl')
    if args.output.exists() or raw_path.exists():
        raise FileExistsError('Calibration observations must be preserved; choose a fresh output')
    service_sha = hashlib.sha256(args.service_profile.read_bytes()).hexdigest()
    samples, knots, upper = [], [], 0.
    with raw_path.open('x') as log:
        for count in sorted(set(args.knots)):
            values = []
            for repetition in range(args.repeats):
                row = measure(args.proxy, args.upstream, args.model, service_sha, count, repetition)
                log.write(json.dumps(row) + '\n'); log.flush()
                values.append(row['wall_ms']); samples.append(row)
                print(json.dumps({k: row[k] for k in ('tokens', 'repetition', 'wall_ms', 'cached_prefix_tokens')}), flush=True)
            upper = max(upper, max(values))
            knots.append(dict(tokens=count, monotone_upper_ms=upper,
                samples_ms=values, sample_count=len(values)))
    args.output.write_text(json.dumps(dict(schema='toolslack.native-prefix-prefill-cost.v1',
        service_profile_sha256=service_sha, extrapolation_allowed=False, knots=knots,
        model=args.model, observation_kind='measured_cold_device_ready_cost',
        raw_samples=str(raw_path), task_quality_result=False, task_qps_result=False), indent=2))


if __name__ == '__main__':
    main()
