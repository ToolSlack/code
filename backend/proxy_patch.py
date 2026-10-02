"""Create a new v15-derived proxy with an enforced explicit KV prefix cap.

This changes system scope only. Native memory, stable token serialization,
same-service calibration admission and exact terminal drainage are preserved.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil


def build(source: Path, target: Path):
    if target.exists():
        raise ValueError('Use a new target; frozen source is never overwritten')
    original = (source / 'model_proxy.py').read_text()
    text = original
    edits = [
        ("            budget_selection = budget.select(len(ids)) if budget is not None else None\n",
         "            limit = spec.get('max_prefix_tokens', len(ids))\n"
         "            if type(limit) is not int or limit <= 0:\n"
         "                raise ValueError('max_prefix_tokens must be a positive exact integer')\n"
         "            bounded_tokens = min(len(ids), limit)\n"
         "            budget_selection = budget.select(bounded_tokens) if budget is not None else None\n"),
        ("                event('kv_budget_selected', request_id=rid, **budget_selection)\n",
         "                event('kv_budget_selected', request_id=rid, **budget_selection)\n"
         "            else:\n"
         "                ids = ids[:bounded_tokens]\n"),
        ("        return web.json_response(dict(prefill=generated, registration=registered,\n",
         "        return web.json_response(dict(prefill=generated, registration=registered,\n"
         "            selected_prefix_ids=ids, max_prefix_tokens=spec.get('max_prefix_tokens'),\n"),
        ("    async def lifecycle_status(request):\n",
         "    async def toolslack_capabilities(request):\n"
         "        return web.json_response(dict(\n"
         "            bounded_prefix_prefill=kv_service_sha is not None,\n"
         "            deadline_terminal_drain=True,\n"
         "            existing_prefix_registration=kv_service_sha is not None,\n"
         "            service_profile_sha256=kv_service_sha,\n"
         "            implementation='v15_explicit_prefix_cap_v1'))\n\n"
         "    async def lifecycle_status(request):\n"),
        ("    app.router.add_get('/health', health)\n",
         "    app.router.add_get('/health', health)\n"
         "    app.router.add_get('/toolslack/capabilities', toolslack_capabilities)\n"),
    ]
    for old, new in edits:
        if text.count(old) != 1:
            raise ValueError('Expected v15 source anchor is missing or ambiguous: ' + old[:70])
        text = text.replace(old, new, 1)
    ast.parse(text)
    shutil.copytree(source, target, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    (target / 'model_proxy.py').write_text(text)
    manifest = dict(source=str(source), target=str(target),
                    source_sha256=hashlib.sha256(original.encode()).hexdigest(),
                    patched_sha256=hashlib.sha256(text.encode()).hexdigest(),
                    engine_algorithms_unchanged=True,
                    additional_guard='explicit token cap alongside existing calibrated deadline guard',
                    GPU_validation_performed=False)
    (target / 'TOOLSLACK_CAP_PATCH.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--target', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.target), indent=2))
