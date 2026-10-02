"""Construct a dedicated dense-Qwen service from the bundled frozen engine.

This module never reuses, stops, or flushes an unrelated service. Installation
and model downloads are performed only after an empty physical GPU is checked.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess

MODEL_ID = "Qwen/Qwen3-8B"
MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"


def content_sha(file):
    digest = hashlib.sha256()
    with Path(file).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_model_weights(model, root):
    lock = json.loads((Path(root) / "resources/model-weights-lock.json").read_text())
    if lock["repository"] != MODEL_ID or lock["revision"] != MODEL_REVISION:
        raise RuntimeError("The model weight lock is not bound to this artifact revision.")
    for row in [lock["index"], *lock["shards"]]:
        file = Path(model) / row["path"]
        if not file.is_file() or file.stat().st_size != row["bytes"] or content_sha(file) != row["sha256"]:
            raise RuntimeError("The checkpoint differs from the pinned official model: " + row["path"])


def verify_native_interpreter(python, root):
    expected = {}
    for line in (Path(root) / "requirements/engine.in").read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            name, version = line.split("==", 1)
            expected[name] = version
    expected["sglang"] = "0.1.dev33+g46ef0661e"
    # Read static wheel metadata; importing torch here can initialize native
    # libraries before the selected device has been bound for service startup.
    script = """import ast, sys, json, importlib.metadata as m
names = json.loads(sys.argv[1])
version_file = m.distribution('torch').locate_file('torch/version.py')
build = {}
for node in ast.parse(version_file.read_text()).body:
    if isinstance(node, (ast.Assign, ast.AnnAssign)):
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            if isinstance(target, ast.Name) and target.id in ('__version__', 'cuda', 'hip'):
                build[target.id] = ast.literal_eval(node.value)
print(json.dumps({'python': sys.version.split()[0],
                  'packages': {name: m.version(name) for name in names},
                  'torch_build': build}))
"""
    try:
        result = subprocess.run([str(python), "-c", script, json.dumps(sorted(expected))],
            capture_output=True, text=True, check=True, timeout=30)
        actual = json.loads(result.stdout)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        raise RuntimeError("Cannot verify the frozen native interpreter dependency closure.") from error
    if actual["python"] != "3.11.13":
        raise RuntimeError("The native engine requires the frozen Python 3.11.13 interpreter.")
    for name, version in expected.items():
        observed = actual["packages"].get(name, "")
        if observed != version and not (name == "torch" and observed == version + "+cu128"):
            raise RuntimeError("The native dependency differs from the frozen environment: " + name)
    build = actual.get("torch_build", {})
    if (build.get("__version__") != "2.9.1+cu128" or build.get("cuda") != "12.8"
            or "hip" not in build or build["hip"] is not None):
        raise RuntimeError("The native engine requires the frozen PyTorch CUDA 12.8 build.")
    return actual


def check_empty_gpu(index):
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise RuntimeError("The bundled CUDA engine requires Linux x86_64; use cpu on this machine.")
    try:
        result = subprocess.run(["nvidia-smi", "-i", str(index),
            "--query-gpu=uuid,memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=15)
        rows = list(csv.reader(result.stdout.splitlines()))
        if len(rows) != 1 or len(rows[0]) != 3:
            raise RuntimeError("Cannot identify one physical GPU for the native service.")
        uuid, used, total = (v.strip() for v in rows[0])
        if not re.fullmatch(r"GPU-[0-9a-fA-F-]{36}", uuid):
            raise RuntimeError("nvidia-smi did not return a valid full GPU UUID.")
        used, total = float(used), float(total)
        if not math.isfinite(used) or not math.isfinite(total) or used < 0 or total <= 0 or used > total:
            raise RuntimeError("nvidia-smi returned invalid memory observations.")
        apps = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True, timeout=15)
        if any(row and row[0].strip() == uuid for row in csv.reader(apps.stdout.splitlines())):
            raise RuntimeError("The selected GPU has an existing compute process. Reserve an empty GPU before starting this artifact.")
        if used > 1024:
            raise RuntimeError("The selected GPU has more than 1 GiB of existing allocations; automatic startup declined.")
        if total < 45000:
            raise RuntimeError("The default dense-Qwen profile requires at least 45,000 MiB device memory.")
        return uuid
    except (OSError, subprocess.CalledProcessError, ValueError) as error:
        raise RuntimeError("Cannot verify the selected GPU is empty with nvidia-smi.") from error


def engine_argv(python, model, engine_port=34200, nccl_port=34202):
    rope = {"max_position_embeddings": 131072,
            "rope_scaling": {"rope_type": "yarn", "factor": 4.0,
                             "original_max_position_embeddings": 32768}}
    return [str(python), "-m", "sglang.launch_server", "--model-path", str(model),
        "--served-model-name", "Qwen3-8B", "--host", "127.0.0.1", "--port", str(engine_port),
        "--dtype", "bfloat16", "--context-length", "131072",
        "--json-model-override-args", json.dumps(rope, separators=(",", ":")),
        "--mem-fraction-static", "0.75", "--max-running-requests", "16",
        "--max-queued-requests", "256", "--chunked-prefill-size", "8192",
        "--max-prefill-tokens", "16384", "--attention-backend", "triton",
        "--reasoning-parser", "qwen3", "--tool-call-parser", "qwen",
        "--enable-metrics", "--enable-cache-report", "--cuda-graph-max-bs", "16",
        "--random-seed", "20260924", "--nccl-port", str(nccl_port),
        "--enable-hierarchical-cache", "--hicache-size", "128",
        "--hicache-write-policy", "write_back", "--hicache-io-backend", "direct",
        "--hicache-mem-layout", "layer_first", "--enable-priority-scheduling",
        "--schedule-policy", "fcfs", "--tp-size", "1"]


def verify_frozen_sources(root):
    manifest = root / "vendor/native_engine/SOURCE_MANIFEST.json"
    if not manifest.is_file():
        raise RuntimeError("The frozen engine source manifest is missing.")
    rows = json.loads(manifest.read_text())["files"]
    base = root / "vendor/native_engine"
    for row in rows:
        file = base / row["path"]
        if not file.is_file() or hashlib.sha256(file.read_bytes()).hexdigest() != row["sha256"]:
            raise RuntimeError("Frozen engine source changed: " + row["path"])


def prepare_default_plan(args, root, output, resources):
    root, output = Path(root).resolve(), Path(output).resolve()
    if not args.exclusive_gpus:
        raise RuntimeError("Use --exclusive-gpus for an independently reserved GPU; this run will start a dedicated service.")
    indices = str(args.gpu_indices or "").split(",")
    if len(indices) != 1 or not indices[0].strip().isdigit():
        raise RuntimeError("The bundled native KV backend supports one physical --gpu-indices value (TP=DP=PP=1).")
    gpu = int(indices[0].strip())
    gpu_uuid = check_empty_gpu(gpu)
    # The tested HiCache profile allocates 128 GiB host memory. Fail before
    # downloading large CUDA packages or checkpoint files on smaller machines.
    meminfo = Path("/proc/meminfo").read_text()
    available = next(int(line.split()[1]) for line in meminfo.splitlines() if line.startswith("MemAvailable:"))
    if available < 150 * 1024 * 1024:
        raise RuntimeError("The historical 128 GiB HiCache profile needs at least 150 GiB available host RAM; supply a separately validated service plan for another profile.")
    verify_frozen_sources(root)
    engine_python = Path(args.engine_python).resolve() if args.engine_python else root / ".engine-venv/bin/python"
    if not args.engine_python:
        subprocess.run(["bash", str(root / "scripts/bootstrap.sh"), "engine"], check=True)
    if not engine_python.is_file():
        raise RuntimeError("The native engine interpreter is missing after bootstrap.")
    runtime = verify_native_interpreter(engine_python, root)
    (output / "native_runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
    if args.model_path:
        model = Path(args.model_path).resolve()
        if not model.is_dir():
            raise RuntimeError("--model-path must be a complete local checkpoint directory.")
    else:
        model = root / "resources/cache/models/Qwen3-8B"
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=MODEL_ID, revision=MODEL_REVISION,
                          local_dir=str(model), local_files_only=args.offline)
    required = ("config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors.index.json")
    if any(not (model / name).is_file() for name in required):
        raise RuntimeError("The complete Qwen3-8B checkpoint metadata is missing.")
    index = json.loads((model / "model.safetensors.index.json").read_text())
    if any(not (model / name).is_file() for name in set(index["weight_map"].values())):
        raise RuntimeError("A Qwen3-8B checkpoint shard is missing; tokenizer-only files cannot start inference.")
    verify_model_weights(model, root)
    # A local checkpoint must use the same tokenizer as the prepared agent.
    tokenizer = Path(resources["TOOLSLACK_TOKENIZER"])
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        if hashlib.sha256((model / name).read_bytes()).digest() != hashlib.sha256((tokenizer / name).read_bytes()).digest():
            raise RuntimeError("The local model and pinned agent tokenizer differ: " + name)
    model_profile = output / "model_profile.local.json"
    from dataclasses import asdict
    import sys
    sys.path.insert(0, str(root / "backend/proxy_v16"))
    from server_profile import ServerProfile
    profile = ServerProfile(model_path=str(model), server_python=str(engine_python),
        server_cwd=str(root / "vendor/native_engine"), random_seed=20260924)
    model_profile.write_text(json.dumps(asdict(profile), indent=2) + "\n")
    engine_port, proxy_port = args.engine_port, args.proxy_port
    if not (1024 <= engine_port <= 65532 and 1024 <= proxy_port <= 65535):
        raise RuntimeError("Use unprivileged engine/proxy ports with room for engine coordination ports.")
    if proxy_port in (engine_port, engine_port + 1, engine_port + 2):
        raise RuntimeError("Proxy and native engine coordination ports overlap.")
    engine_url, proxy_url = f"http://127.0.0.1:{engine_port}", f"http://127.0.0.1:{proxy_port}"
    service_profile = output / "service_profile.local.json"
    argv = engine_argv(engine_python, model, engine_port, engine_port + 2)
    binding = {"schema": "toolslack.artifact.native-service.v1", "model": MODEL_ID,
        "model_revision": MODEL_REVISION, "engine_argv": argv,
        "engine_source_manifest_sha256": hashlib.sha256((root / "vendor/native_engine/SOURCE_MANIFEST.json").read_bytes()).hexdigest(),
        "host_cache_gib": 128, "gpu_indices": [gpu], "tensor_parallel": 1,
        "data_parallel": 1, "pipeline_parallel": 1, "storage_backend": None}
    service_profile.write_text(json.dumps(binding, indent=2, sort_keys=True) + "\n")
    cost = output / "calibration/prefix_cost.json"
    native_env = {"PYTHONPATH": str(root / "vendor/native_engine/python"),
        "TOOLSLACK_KV_PROFILE_PATH": str(service_profile),
        "TOOLSLACK_KV_ENABLE_DIAGNOSTICS": "1", "PYTHONUNBUFFERED": "1"}
    base_proxy = [str(engine_python), str(root / "backend/proxy_v16/model_proxy.py"),
        "--profile", str(model_profile), "--upstream", engine_url,
        "--port", str(proxy_port), "--kv-service-profile", str(service_profile),
        # The benchmark controller already owns optional worker/deadline
        # scheduling. Its transport does not implement the distinct proxy
        # maintenance-metadata contract, so leave that second scheduler off.
        "--kv-max-prefills", "1", "--optional-memory-workers", "0"]
    plan = {"schema": "toolslack.artifact.service-plan.v1", "service_profile": str(service_profile),
        "cost_profile": str(cost), "engine_url": engine_url, "proxy_url": proxy_url,
        "gpu_indices": [gpu], "gpu_uuid": gpu_uuid, "startup_timeout_s": 1200,
        "engine": {"argv": argv, "cwd": str(root), "env": native_env},
        "calibration_proxy": {"argv": base_proxy + ["--output", str(output / "calibration_proxy")],
                              "cwd": str(root), "env": native_env},
        "proxy": {"argv": base_proxy + ["--output", str(output / "proxy"), "--kv-cost-profile", str(cost)],
                  "cwd": str(root), "env": native_env}, "calibration": {}}
    path = output / "service-plan.local.json"
    path.write_text(json.dumps(plan, indent=2) + "\n")
    args.service_plan = path
    return plan
