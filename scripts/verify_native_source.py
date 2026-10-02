#!/usr/bin/env python3
"""Check renamed frozen-engine source and native KV contract with CPU stubs.

No SGLang or PyTorch package is imported and no subprocess or network call occurs.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import runpy
from types import SimpleNamespace


def verify(engine):
    manifest = json.loads((engine / "SOURCE_MANIFEST.json").read_text())
    for row in manifest["files"]:
        file = engine / row["path"]
        if not file.is_file() or hashlib.sha256(file.read_bytes()).hexdigest() != row["sha256"]:
            raise RuntimeError("Frozen native source bytes differ: " + row["path"])
    files = sorted((engine / "python/sglang").rglob("*.py"))
    for file in files:
        ast.parse(file.read_bytes(), filename=str(file), feature_version=(3, 11))
    package = engine / "python/sglang"
    native = package / "srt/mem_cache/toolslack_kv_control.py"
    module = runpy.run_path(str(native))
    profile_hash = hashlib.sha256(b'{"portable_profile":true}\n').hexdigest()
    cache = SimpleNamespace(cache_controller=SimpleNamespace(ack_write_queue=[]),
        writing_check=lambda: None, loading_check=lambda: None)
    manager = module["PrefixManager"](cache, SimpleNamespace(), profile_hash,
        active_rids=lambda: set(), clock=lambda: 12345)
    body = dict(action="request_status", request_id="cpu-probe-1",
        consumer_request_id="cpu-nonexistent-1", service_profile_sha256=profile_hash)
    response = manager.control(body)
    assert response["ok"] is True and response["active"] is False
    assert response["service_profile_sha256"] == profile_hash
    assert response["engine_epoch"] == response["service_epoch"]
    assert response["terminal_proof"] is False
    assert manager.control(body) == response
    conflict = manager.control(dict(body, consumer_request_id="different"))
    assert conflict["ok"] is False and conflict["error"]["code"] == "idempotency_conflict"
    mismatch = manager.control(dict(body, request_id="cpu-probe-2", service_profile_sha256="0" * 64))
    assert mismatch["ok"] is False and mismatch["error"]["code"] == "profile_mismatch"
    # Verify names statically to avoid importing GPU-serving entry points.
    server = (package / "srt/entrypoints/http_server.py").read_text()
    runtime = (package / "srt/mem_cache/toolslack_kv_runtime.py").read_text()
    assert '/toolslack/kv/control' in server
    assert 'TOOLSLACK_KV_PROFILE_PATH' in runtime
    assert 'toolslack_kv_control' in runtime
    return dict(schema="toolslack.native-source-cpu-receipt.v1",
        verified_manifest_files=len(manifest["files"]), parsed_python_files=len(files),
        parser_feature_version="3.11", checks=["source_manifest_integrity", "source_parse",
        "renamed_native_route_and_profile_env", "nonexistent_request_status_schema",
        "profile_binding", "idempotent_replay", "idempotency_conflict", "profile_mismatch"],
        GPU_operations=0, imports_engine=False, validates_fresh_gpu_installation=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine-dir", type=Path,
        default=Path(__file__).resolve().parents[1] / "vendor/native_engine")
    args = parser.parse_args()
    print(json.dumps(verify(args.engine_dir.resolve()), indent=2))


if __name__ == "__main__":
    main()
