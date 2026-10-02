"""One entry point for setup, CPU correctness, and real owned-service runs."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import uuid

from .report import (Redactor, file_sha, machine_summary, measured_report,
                     save_json, utc_now, write_report)

ROOT = Path(__file__).resolve().parents[1]
GPU_PACKAGES = ("aiohttp", "langgraph", "langmem", "langchain-core", "transformers", "httpx")


class PreflightError(RuntimeError):
    pass


def env_default(name, default=None):
    return os.environ.get("TOOLSLACK_" + name, default)


def env_flag(name):
    return str(env_default(name, "0")).lower() in ("1", "true", "yes")


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("mode", choices=("cpu", "prepare", "smoke", "full"))
    result.add_argument("--output", type=Path, help="fresh result directory (default: results/<UTC>-<mode>)")
    result.add_argument("--no-bootstrap", action="store_true", help="use the active interpreter; fail if dependencies are missing")
    result.add_argument("--bootstrapped", action="store_true", help=argparse.SUPPRESS)
    result.add_argument("--data-root", type=Path, default=env_default("DATASET_ROOT"))
    result.add_argument("--subsets", type=Path, default=env_default("SUBSETS"))
    result.add_argument("--tokenizer", type=Path, default=env_default("TOKENIZER"))
    result.add_argument("--model", default=env_default("MODEL", "Qwen3-8B"))
    result.add_argument("--model-path", type=Path, default=env_default("MODEL_PATH"), help="optional cached full official checkpoint for tracked engine startup")
    result.add_argument("--engine-python", type=Path, default=env_default("ENGINE_PYTHON"), help="optional separate verified native engine interpreter")
    result.add_argument("--engine-port", type=int, default=34200)
    result.add_argument("--proxy-port", type=int, default=34203)
    result.add_argument("--offline", action="store_true", help="require cached official resources; do not download")
    result.add_argument("--service-profile", type=Path, default=env_default("SERVICE_PROFILE"))
    result.add_argument("--cost-profile", type=Path, default=env_default("COST_PROFILE"))
    result.add_argument("--engine-url", default=env_default("ENGINE_URL"))
    result.add_argument("--proxy-url", default=env_default("PROXY_URL"))
    result.add_argument("--gpu-indices", default=env_default("GPU_INDICES"), help="physical nvidia-smi indices, comma separated")
    result.add_argument("--owned-service", action="store_true", default=env_flag("OWNED_SERVICE"), help="affirm these endpoints are dedicated to this run; authorizes cache flushes")
    result.add_argument("--exclusive-gpus", action="store_true", default=env_flag("EXCLUSIVE_GPUS"), help="affirm the selected physical GPUs are independently reserved for this run")
    result.add_argument("--service-plan", type=Path, help="native engine/proxy process plan JSON; start and stop only tracked processes")
    result.add_argument("--native-control-path", default="/toolslack/kv/control")
    result.add_argument("--tool-mode", choices=("cpu", "model"), default=env_default("TOOL_MODE", "model"))
    result.add_argument("--selection", choices=("frozen-subset", "long-context"), default=env_default("SELECTION", "frozen-subset"))
    result.add_argument("--eval-tasks", type=int, default=int(env_default("EVAL_TASKS", "16")))
    result.add_argument("--cal-tasks", type=int, default=int(env_default("CAL_TASKS", "3")))
    result.add_argument("--repeats", type=int, default=int(env_default("REPEATS", "2")))
    result.add_argument("--concurrency", type=int, default=int(env_default("CONCURRENCY", "4")))
    result.add_argument("--memory-workers", type=int, default=int(env_default("MEMORY_WORKERS", "1")))
    result.add_argument("--foreground-limit", type=int, default=int(env_default("FOREGROUND_LIMIT", "16")))
    result.add_argument("--memory-max-tokens", type=int, default=int(env_default("MEMORY_MAX_TOKENS", "8192")))
    result.add_argument("--memory-trigger-tokens", type=int, default=int(env_default("MEMORY_TRIGGER_TOKENS", "4096")))
    result.add_argument("--summary-tokens", type=int, default=int(env_default("SUMMARY_TOKENS", "384")))
    result.add_argument("--foreground-tokens", type=int, default=int(env_default("FOREGROUND_TOKENS", "640")))
    result.add_argument("--max-scopes", type=int, default=int(env_default("MAX_SCOPES", "3")))
    result.add_argument("--request-timeout-s", type=float, default=float(env_default("REQUEST_TIMEOUT_S", "1200")))
    result.add_argument("--safety-margin-s", type=float, default=float(env_default("SAFETY_MARGIN_S", "0.05")))
    result.add_argument("--benefit-reference", choices=("serial_native_memory_to_first_token", "original_model_ttft"),
                        default=env_default("BENEFIT_REFERENCE", "serial_native_memory_to_first_token"))
    return result


def parse_gpus(value):
    if not value:
        raise PreflightError("Select independently reserved physical GPUs with --gpu-indices.")
    try:
        values = [int(part.strip()) for part in str(value).split(",")]
    except ValueError as error:
        raise PreflightError("GPU indices must be nonnegative physical integer indices.") from error
    if not values or min(values) < 0 or len(values) != len(set(values)):
        raise PreflightError("GPU indices must be distinct nonnegative physical indices.")
    return values


def require_measurement_ownership(args):
    if args.mode in ("smoke", "full"):
        if not (args.owned_service or args.service_plan):
            raise PreflightError("Measurement flushes native caches. Supply --owned-service only for a dedicated endpoint, or --service-plan to create tracked services.")
        if not args.exclusive_gpus:
            raise PreflightError("Measurement requires --exclusive-gpus and an independent GPU allocation; shared services cannot be measured safely.")


def get_json(url, path, timeout=10):
    parsed = urllib.parse.urlparse(url or "")
    if parsed.scheme not in ("http", "https") or not parsed.netloc or parsed.username or parsed.password:
        raise PreflightError("Provide an HTTP(S) endpoint without embedded credentials.")
    request = urllib.request.Request(url.rstrip("/") + path, method="GET")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=timeout) as response:
            if response.status != 200:
                raise PreflightError("Read-only endpoint preflight did not return HTTP 200.")
            value = json.load(response)
    except (OSError, ValueError) as error:
        raise PreflightError("Read-only endpoint preflight failed: " + str(error)) from error
    if not isinstance(value, dict):
        raise PreflightError("Read-only endpoint must return a JSON object.")
    return value


def native_probe(args, service_sha):
    """Observe a never-submitted request; do not register/cancel/flush anything."""
    rid = "probe-" + uuid.uuid4().hex
    consumer = "probe-nonexistent-" + uuid.uuid4().hex
    body = dict(request_id=rid, action="request_status", consumer_request_id=consumer,
                service_profile_sha256=service_sha)
    request = urllib.request.Request(args.engine_url.rstrip("/") + args.native_control_path,
               data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}, method="POST")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=10) as response:
            value = json.load(response)
    except (OSError, ValueError) as error:
        raise PreflightError("Observational native request_status probe failed: " + str(error)) from error
    validate_native_probe(value, body)
    return value


def validate_native_probe(receipt, body):
    if not isinstance(receipt, dict) or receipt.get("ok") is not True:
        raise PreflightError("Native request_status did not confirm a successful observational probe.")
    for key in ("request_id", "action", "consumer_request_id", "service_profile_sha256"):
        if receipt.get(key) != body[key]:
            raise PreflightError("Native request_status identity/profile differs: " + key)
    epoch = receipt.get("service_epoch")
    if not isinstance(epoch, str) or not epoch or receipt.get("engine_epoch") != epoch:
        raise PreflightError("Native request_status service epoch is absent or inconsistent.")
    if receipt.get("active") is not False or receipt.get("terminal_proof") is not False:
        raise PreflightError("Native request_status for a fresh nonexistent request must be inactive and cannot prove termination.")


def validate_service_snapshots(health, proxycaps, lifecycle, nativecaps, service_sha, cost_sha, model):
    capacity = health.get("capacity", {})
    if health.get("status") != "tokenizer_ready" or capacity.get("model_name") != model:
        raise PreflightError("Proxy is not ready for the selected model.")
    if capacity.get("kv_service_profile_sha256") != service_sha:
        raise PreflightError("Proxy service profile differs from the supplied native service profile.")
    if capacity.get("kv_cost_profile_sha256") != cost_sha:
        raise PreflightError("Proxy cost profile differs from the supplied measured cost profile. Restart the owned proxy with that profile.")
    if capacity.get("context_window") != 131072:
        raise PreflightError("This benchmark requires the verified Qwen3-8B 131072-token native service.")
    if proxycaps.get("service_profile_sha256") != service_sha:
        raise PreflightError("Proxy native capability binding differs.")
    for name in ("bounded_prefix_prefill", "deadline_terminal_drain", "existing_prefix_registration"):
        if proxycaps.get(name) is not True:
            raise PreflightError("Proxy is missing required native capability: " + name)
    if (lifecycle.get("quiescent") is not True or lifecycle.get("active_owners") != 0 or
            lifecycle.get("guard_cleanup_required") is not False or lifecycle.get("held_prefill_slots") != 0):
        raise PreflightError("Owned proxy has undrained requests or unconfirmed native cleanup; no cache flush is allowed.")
    if nativecaps.get("service_profile_sha256") != service_sha:
        raise PreflightError("Native observational probe service profile differs.")


def check_dependencies():
    missing = []
    for package in GPU_PACKAGES:
        try:
            importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            missing.append(package)
    if missing:
        raise PreflightError("Official runtime dependencies missing: " + ", ".join(missing))


def gpu_snapshot(indices):
    try:
        process = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version",
                                  "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=15, check=True)
    except (OSError, subprocess.SubprocessError) as error:
        raise PreflightError("Real GPU measurement requires working local NVIDIA GPUs and nvidia-smi.") from error
    rows = []
    for line in process.stdout.splitlines():
        fields = [part.strip() for part in line.split(",")]
        if len(fields) == 4 and fields[0].isdigit() and int(fields[0]) in indices:
            rows.append(dict(index=int(fields[0]), name=fields[1], memory_total_mib=int(fields[2]), driver_version=fields[3]))
    if {r["index"] for r in rows} != set(indices):
        raise PreflightError("Selected physical GPUs are unavailable in nvidia-smi.")
    return rows


def build_environment(args, resources):
    env = os.environ.copy()
    env.update({key: str(value) for key, value in resources.items() if key.startswith("TOOLSLACK_")})
    mapping = dict(SERVICE_PROFILE=args.service_profile, COST_PROFILE=args.cost_profile,
                   ENGINE_URL=args.engine_url, PROXY_URL=args.proxy_url, MODEL=args.model,
                   GPU_INDICES=",".join(map(str, parse_gpus(args.gpu_indices))), TOOL_MODE=args.tool_mode,
                   SELECTION=args.selection, EVAL_TASKS=args.eval_tasks, CAL_TASKS=args.cal_tasks,
                   REPEATS=args.repeats, CONCURRENCY=args.concurrency, MEMORY_WORKERS=args.memory_workers,
                   FOREGROUND_LIMIT=args.foreground_limit, MEMORY_MAX_TOKENS=args.memory_max_tokens,
                   MEMORY_TRIGGER_TOKENS=args.memory_trigger_tokens, SUMMARY_TOKENS=args.summary_tokens,
                   FOREGROUND_TOKENS=args.foreground_tokens, MAX_SCOPES=args.max_scopes,
                   REQUEST_TIMEOUT_S=args.request_timeout_s, SAFETY_MARGIN_S=args.safety_margin_s,
                   BENEFIT_REFERENCE=args.benefit_reference)
    for key, value in mapping.items():
        if value is not None:
            env["TOOLSLACK_" + key] = str(value)
    # Compatibility with the original guarded runner is explicit and truthful.
    if args.mode in ("smoke", "full"):
        env["TOOLSLACK_ARTIFACT_OWNED_SERVICE"] = "1"
        env["TOOLSLACK_ARTIFACT_EXCLUSIVE_GPU"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    return env


def parameters(args):
    names = ("mode", "model", "tool_mode", "selection", "eval_tasks", "cal_tasks", "repeats", "concurrency",
             "memory_workers", "foreground_limit", "memory_max_tokens", "memory_trigger_tokens", "summary_tokens",
             "foreground_tokens", "max_scopes", "request_timeout_s", "safety_margin_s", "benefit_reference")
    return {name: getattr(args, name) for name in names}


class Session:
    def __init__(self, output, mode):
        self.output = output
        self.redactor = Redactor(ROOT)
        self.children = []
        self.log_readers = []
        self.status = dict(schema="toolslack.artifact.status.v1", mode=mode, state="running",
                           started_at=utc_now(), performance_evidence=False)
        previous = output / "status.json"
        if previous.is_file():
            self.status["started_at"] = json.loads(previous.read_text())["started_at"]
        self.phase("starting")

    def phase(self, value, **extra):
        self.status.update(phase=value, updated_at=utc_now(), **extra)
        save_json(self.output / "status.json", self.redactor.value(self.status))
        print(json.dumps(dict(phase=value, **self.redactor.value(extra))), flush=True)

    def command(self, argv, name, env=None, cwd=ROOT):
        target = self.output / (name + ".log")
        with target.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(argv, cwd=cwd, env=env, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                                       start_new_session=True)
            self.children.append(process)
            try:
                for line in process.stdout:
                    safe = self.redactor.text(line)
                    log.write(safe)
                    log.flush()
                    print(safe, end="", flush=True)
                return process.wait()
            finally:
                if process.poll() is not None:
                    self.children.remove(process)

    def finish(self, state, evidence=False, error=None):
        self.status.update(state=state, performance_evidence=evidence, finished_at=utc_now())
        if error:
            self.status["error"] = self.redactor.text(error)
        self.phase("finished")

    def cleanup(self):
        # Only process groups created by this exact session can be signalled.
        for process in list(reversed(self.children)):
            if process.poll() is not None:
                continue
            try:
                os.killpg(process.pid, signal.SIGTERM)
                # A nested runner receives SIGTERM, records failure, and drains
                # its own tracked engine/proxy groups before it exits.
                process.wait(timeout=45)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
            except ProcessLookupError:
                pass
        self.children.clear()
        for reader in self.log_readers:
            reader.join(timeout=5)
        self.log_readers.clear()


def load_plan(args, plan=None):
    if plan is None:
        if not args.service_plan:
            return None
        plan = json.loads(args.service_plan.read_text())
    if plan.get("schema") != "toolslack.artifact.service-plan.v1":
        raise PreflightError("Unsupported native service process plan schema.")
    planned_gpus = plan.get("gpu_indices")
    if planned_gpus is not None and args.gpu_indices is not None:
        planned = ",".join(map(str, planned_gpus)) if isinstance(planned_gpus, list) else planned_gpus
        if parse_gpus(planned) != parse_gpus(args.gpu_indices):
            raise PreflightError("Service plan GPUs differ from the explicitly selected physical GPU allocation.")
    for name in ("service_profile", "cost_profile", "engine_url", "proxy_url", "gpu_indices"):
        if getattr(args, name) is None and plan.get(name) is not None:
            value = plan[name]
            if name in ("service_profile", "cost_profile"):
                value = Path(value)
                if not value.is_absolute():
                    value = ((args.service_plan.parent if args.service_plan else ROOT) / value).resolve()
            if name == "gpu_indices" and isinstance(value, list):
                value = ",".join(map(str, value))
            setattr(args, name, value)
    for name in ("engine", "proxy", *(('calibration_proxy',) if plan.get('calibration') is not None else ())):
        spec = plan.get(name, {})
        if not isinstance(spec.get("argv"), list) or not spec["argv"] or not all(isinstance(v, str) for v in spec["argv"]):
            raise PreflightError("Native process plan requires explicit argument lists for " + name + "; shell commands are not accepted.")
    return plan


def expand_plan(value, env):
    def replace(match):
        key = match.group(1)
        if not key.startswith("TOOLSLACK_") or key not in env:
            raise PreflightError("Native process plan has an unknown resource variable: " + key)
        return env[key]
    return re.sub(r"\$\{([A-Z][A-Z0-9_]*)\}", replace, value)


def stop_tracked_process(session, process):
    if process not in session.children:
        raise PreflightError("Refusing to stop an untracked native service process.")
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)
    session.children.remove(process)


def endpoint_responds(url):
    parsed = urllib.parse.urlparse(url or "")
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise PreflightError("The service plan must supply valid HTTP(S) endpoints.")
    import socket
    try:
        with socket.create_connection((parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80)), timeout=2):
            return True
    except OSError:
        return False


def start_plan(session, args, plan, env, service_sha):
    if not args.exclusive_gpus:
        raise PreflightError("Starting native services requires --exclusive-gpus.")
    for endpoint in (args.engine_url, args.proxy_url):
        if endpoint_responds(endpoint):
            raise PreflightError("A service already responds at a planned endpoint; tracked startup will not take it over.")
    names = ("engine", "calibration_proxy", "proxy") if plan.get("calibration") is not None else ("engine", "proxy")
    for name in names:
        spec = plan[name]
        child_env = env.copy()
        for key, value in spec.get("env", {}).items():
            child_env[key] = expand_plan(str(value), env)
        child_env["CUDA_VISIBLE_DEVICES"] = str(plan.get("gpu_uuid", env["TOOLSLACK_GPU_INDICES"]))
        child_env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        if name == "engine":
            from .native import check_empty_gpu
            indices = parse_gpus(args.gpu_indices)
            if len(indices) != 1:
                raise PreflightError("Tracked startup is validated for one independently reserved GPU; use a separately validated owned service for other layouts.")
            # Model downloads/bootstrap may take time. Recheck immediately
            # before the first engine Popen, using the stable physical UUID.
            current_uuid = check_empty_gpu(indices[0])
            if plan.get("gpu_uuid") and current_uuid != plan["gpu_uuid"]:
                raise PreflightError("Selected physical GPU identity changed after preparation; the engine will not start.")
            if not isinstance(current_uuid, str) or not current_uuid.startswith("GPU-"):
                raise PreflightError("GPU recheck did not return a validated physical GPU UUID.")
            child_env["CUDA_VISIBLE_DEVICES"] = current_uuid
        argv = [expand_plan(value, child_env) for value in spec["argv"]]
        cwd = Path(expand_plan(spec.get("cwd", str(ROOT)), child_env))
        # A reader process writes redacted output; never create a raw log file.
        process = subprocess.Popen(argv, cwd=cwd, env=child_env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                                   start_new_session=True)
        session.children.append(process)
        import threading
        def copy_log(proc=process, filename=name):
            with (session.output / (filename + ".log")).open("w", encoding="utf-8") as stream:
                for line in proc.stdout:
                    stream.write(session.redactor.text(line)); stream.flush()
        reader = threading.Thread(target=copy_log, daemon=True)
        session.log_readers.append(reader)
        reader.start()
        # Calibration can take longer than startup. Each newly started
        # component receives its own readiness budget.
        deadline = time.monotonic() + float(plan.get("startup_timeout_s", 600))
        endpoint = args.engine_url if name == "engine" else args.proxy_url
        while True:
            if process.poll() is not None:
                raise PreflightError("Tracked native " + name + " exited before readiness; inspect its redacted log.")
            try:
                if name == "engine":
                    native_probe(args, service_sha)
                else:
                    health = get_json(endpoint, "/health", timeout=2)
                    capacity = health.get("capacity", {})
                    if health.get("status") != "tokenizer_ready" or capacity.get("kv_service_profile_sha256") != service_sha:
                        raise PreflightError("Tracked proxy service profile differs or tokenizer is not ready.")
                break
            except PreflightError:
                if time.monotonic() >= deadline:
                    raise PreflightError("Tracked native " + name + " did not become ready before the startup deadline.")
                time.sleep(1)
        session.phase(name + "_ready", tracked_process=True)
        if name == "calibration_proxy":
            # These serving-cost probes flush only this session's tracked engine.
            lifecycle = get_json(args.proxy_url, "/lifecycle/status")
            if lifecycle.get("quiescent") is not True or lifecycle.get("guard_cleanup_required") is not False:
                raise PreflightError("Tracked calibration proxy has unconfirmed native cleanup.")
            calibration = plan["calibration"]
            if not isinstance(calibration, dict):
                raise PreflightError("Service plan calibration must be an object.")
            cost_path = args.cost_profile.resolve()
            cost_path.parent.mkdir(parents=True, exist_ok=True)
            gate_path = cost_path.with_name("KV_LIFECYCLE_GATE.json")
            if cost_path.exists() or gate_path.exists():
                raise PreflightError("Fresh native calibration paths are required; previous observations will not be overwritten.")
            common = ["--proxy", args.proxy_url, "--upstream", args.engine_url,
                      "--service-profile", str(args.service_profile)]
            prefix = calibration.get("prefix_argv", [sys.executable, "-m", "calibration.prefix",
                     *common, "--output", str(cost_path)])
            lifecycle_argv = calibration.get("lifecycle_argv", [sys.executable, "-m", "calibration.lifecycle",
                            *common, "--output", str(gate_path)])
            for kind, argv in (("prefix_cost", prefix), ("kv_lifecycle", lifecycle_argv)):
                if not isinstance(argv, list) or not argv or not all(isinstance(v, str) for v in argv):
                    raise PreflightError("Calibration requires explicit argument lists; shell commands are not accepted.")
                session.phase("calibrating_" + kind)
                code = session.command([expand_plan(v, env) for v in argv], "calibration_" + kind, env=env)
                if code:
                    raise PreflightError("Real native " + kind + " calibration failed; no performance is valid.")
            from prefix_budget import PrefixCost
            PrefixCost.load(cost_path, service_sha)
            gate = json.loads(gate_path.read_text())
            if (gate.get("state") != "passed" or gate.get("service_profile_sha256") != service_sha or
                    gate.get("prefix_bytes_unchanged") is not True or gate.get("resident_restored_greedy_equal") is not True):
                raise PreflightError("Real same-service native lifecycle calibration did not pass.")
            lifecycle = get_json(args.proxy_url, "/lifecycle/status")
            if lifecycle.get("quiescent") is not True:
                raise PreflightError("Native calibration did not drain before proxy handoff.")
            save_json(session.output / "serving_calibration.json", dict(state="passed", calibration_only=True,
                        task_qps_result=False, cost_profile_sha256=file_sha(cost_path), lifecycle_gate_sha256=file_sha(gate_path),
                        native_service_epoch=native_probe(args, service_sha)["service_epoch"]))
            stop_tracked_process(session, process)


def preflight(args, session, resources, plan):
    require_measurement_ownership(args)
    check_dependencies()
    if args.model != "Qwen3-8B":
        raise PreflightError("The real agent benchmark is closed for Qwen3-8B; other profiles require separate validation.")
    if args.cal_tasks < 3 or args.eval_tasks < 4 or args.repeats < 1:
        raise PreflightError("Require at least 3 disjoint calibration tasks, 4 evaluation tasks, and 1 repeat.")
    for value in (args.concurrency, args.foreground_limit, args.memory_max_tokens, args.memory_trigger_tokens,
                  args.summary_tokens, args.foreground_tokens, args.max_scopes):
        if value <= 0:
            raise PreflightError("Concurrency, native thresholds, output lengths, and scopes must be positive.")
    if not 0 < args.memory_trigger_tokens <= args.memory_max_tokens or args.memory_workers < 1:
        raise PreflightError("Native memory threshold or worker count is invalid.")
    if not args.concurrency <= args.foreground_limit <= 16:
        raise PreflightError("Foreground concurrency must fit the verified at-most-16-request service.")
    if args.request_timeout_s <= 0 or args.safety_margin_s < 0:
        raise PreflightError("Request timeout must be positive and safety margin nonnegative.")
    for name in ("service_profile", "cost_profile"):
        value = getattr(args, name)
        will_calibrate = name == "cost_profile" and plan and plan.get("calibration") is not None
        if not value or (not value.is_file() and not will_calibrate):
            raise PreflightError("A real existing --" + name.replace("_", "-") + " JSON is required; no example profile is substituted.")
    service_sha = file_sha(args.service_profile)
    sys.path.insert(0, str(ROOT / "backend/proxy_v16"))
    from prefix_budget import PrefixCost
    gpu_rows = gpu_snapshot(parse_gpus(args.gpu_indices))
    session.phase("gpu_verified")
    env = build_environment(args, resources)
    if plan:
        start_plan(session, args, plan, env, service_sha)
    PrefixCost.load(args.cost_profile, service_sha)
    cost_sha = file_sha(args.cost_profile)
    if not args.engine_url or not args.proxy_url:
        raise PreflightError("Set dedicated --engine-url and --proxy-url, or provide a closed native --service-plan.")
    snapshots = dict(health=get_json(args.proxy_url, "/health"),
                     proxy_capabilities=get_json(args.proxy_url, "/toolslack/capabilities"),
                     lifecycle=get_json(args.proxy_url, "/lifecycle/status"),
                     native_capabilities=native_probe(args, service_sha))
    validate_service_snapshots(snapshots["health"], snapshots["proxy_capabilities"], snapshots["lifecycle"],
                               snapshots["native_capabilities"], service_sha, cost_sha, args.model)
    save_json(session.output / "preflight.json", session.redactor.value(dict(
        state="passed", observational_native_probe=True, service_profile_sha256=service_sha, cost_profile_sha256=cost_sha,
        profile_hash_basis="original_runtime_bytes_before_result_anonymization",
        ownership="tracked_processes" if plan else "external_dedicated_service",
        owned_service_declared=bool(args.owned_service or plan), exclusive_gpus_declared=args.exclusive_gpus,
        native_service_restore_guard_present=bool(os.environ.get("TOOLSLACK_RUNTIME_ROOT")),
        gpus=gpu_rows, snapshots=snapshots)))
    return env


def cpu_run(session):
    session.phase("frozen_native_source")
    code = session.command([sys.executable, str(ROOT / "scripts/verify_native_source.py")], "native_source")
    if code:
        raise PreflightError("Frozen native source hashes, syntax or observational control checks failed.")
    suites = [("core", "tests", 70), ("proxy", "backend/proxy_v16", 69),
              ("benchmark_protocol", "benchmark", 9), ("artifact_cli", "artifact_tests", None)]
    results = {}
    for name, directory, expected in suites:
        session.phase("cpu_" + name)
        summary = session.output / (name + "_tests.json")
        code = session.command([sys.executable, "-m", "artifact.testing", "--directory", directory,
                                "--summary", str(summary)], "cpu_" + name)
        if not summary.is_file():
            raise PreflightError("CPU test runner did not write the " + name + " result.")
        value = json.loads(summary.read_text())
        value["expected_tests"] = expected
        results[name] = value
        save_json(session.output / "cpu.json", dict(cpu_only=True, performance_evidence=False,
                    artificial_protocol_fixtures=True, gpu_operations=0, suites=results))
        if code or value["skipped"] or (expected is not None and value["tests_run"] != expected):
            raise PreflightError("CPU suite " + name + " failed, skipped tests, or changed its audited test count.")
    return dict(schema="toolslack.artifact.report.v1", mode="cpu", state="cpu_passed",
                performance_evidence=False, performance_valid=False, suites=results,
                validity_reasons=["CPU protocol/model/tokenizer fixtures establish correctness only; no GPU performance measured."])


def main(argv=None):
    args = parser().parse_args(argv)
    minimum = (3, 11) if args.no_bootstrap or args.bootstrapped else (3, 9)
    if sys.version_info < minimum:
        print("Python " + ".".join(map(str, minimum)) + " or newer is required.", file=sys.stderr)
        return 2
    if args.output:
        output = args.output.resolve()
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output = ROOT / "results" / f"{stamp}-{args.mode}-{uuid.uuid4().hex[:6]}"
    if output.exists() and not args.bootstrapped:
        print("Result directory already exists; use a fresh --output.", file=sys.stderr)
        return 2
    output.mkdir(parents=True, exist_ok=args.bootstrapped)
    session = Session(output, args.mode)
    previous_sigterm = signal.getsignal(signal.SIGTERM)
    def terminate_run(signum, frame):
        raise KeyboardInterrupt("Run termination requested")
    signal.signal(signal.SIGTERM, terminate_run)
    try:
        save_json(output / "parameters.json", parameters(args))
        save_json(output / "machine.json", machine_summary())
        if not args.no_bootstrap and not args.bootstrapped:
            session.phase("bootstrap")
            code = session.command(["bash", str(ROOT / "scripts/bootstrap.sh"),
                                    "cpu" if args.mode == "cpu" else "gpu"], "bootstrap")
            if code:
                raise PreflightError("Environment bootstrap failed. Inspect bootstrap.log; the run is not performance evidence.")
            executable = ROOT / ".venv/bin/python"
            if not executable.is_file():
                raise PreflightError("Bootstrap did not create .venv/bin/python.")
            next_args = list(argv if argv is not None else sys.argv[1:])
            # Avoid repeating a user output argument, preserving other options verbatim.
            if "--output" in next_args:
                index = next_args.index("--output"); del next_args[index:index + 2]
            next_args = [value for value in next_args if not value.startswith("--output=")]
            code = session.command([str(executable), "-m", "artifact.runner", *next_args,
                                    "--bootstrapped", "--output", str(output)], "run")
            return code
        if args.mode == "cpu":
            report = cpu_run(session)
            write_report(output, report)
            session.finish("cpu_passed")
            return 0
        plan = load_plan(args)
        if args.mode in ("smoke", "full") and not args.exclusive_gpus:
            require_measurement_ownership(args)
        if not plan and (args.engine_url or args.proxy_url):
            require_measurement_ownership(args)
        session.phase("resources")
        from .resources import ensure_resources
        resources = ensure_resources(args, ROOT)
        if not isinstance(resources, dict):
            raise PreflightError("Official resource preparation did not return an environment mapping.")
        if not plan and not args.engine_url and not args.proxy_url:
            session.phase("native_process_plan")
            from .native import prepare_default_plan
            plan = load_plan(args, prepare_default_plan(args, ROOT, output, resources))
            args.owned_service = True
        require_measurement_ownership(args)
        session.phase("preflight")
        env = preflight(args, session, resources, plan)
        session.phase("prepare" if args.mode == "prepare" else "measuring")
        command = [sys.executable, "-m", "benchmark.run_series", "--output", str(output / "benchmark")]
        if args.mode == "prepare":
            command.append("--prepare-only")
        else:
            command += ["--stage", args.mode]
        code = session.command(command, "benchmark", env=env)
        if code:
            raise PreflightError("Original real benchmark exited unsuccessfully. Preserve partial cells and inspect benchmark.log.")
        if args.mode == "prepare":
            report = dict(schema="toolslack.artifact.report.v1", mode="prepare", state="prepared_only",
                          performance_evidence=False, performance_valid=False,
                          validity_reasons=["Official inputs, dependency closure, GPU service and frozen config were checked; no performance was measured."])
            write_report(output, report)
            session.finish("prepared_only")
            return 0
        series_path = output / "benchmark/series.json"
        if not series_path.is_file():
            raise PreflightError("Real benchmark did not write a series checkpoint.")
        report = measured_report(json.loads(series_path.read_text()), args.mode, args.repeats)
        report["state"] = "completed" if report["performance_valid"] else "invalid_measurement"
        write_report(output, report)
        session.finish(report["state"], evidence=report["performance_valid"])
        return 0 if report["performance_valid"] else 1
    except (Exception, KeyboardInterrupt) as error:
        message = f"{type(error).__name__}: {error}"
        session.finish("failed", error=message)
        report = dict(schema="toolslack.artifact.report.v1", mode=args.mode, state="failed",
                      performance_evidence=False, performance_valid=False,
                      validity_reasons=[session.redactor.text(message)])
        partial = output / "benchmark/series.json"
        if partial.is_file():
            try:
                report["partial_measured_cells"] = measured_report(json.loads(partial.read_text()), args.mode, args.repeats)["cells"]
            except (ValueError, TypeError):
                pass
        write_report(output, report)
        print(session.redactor.text(message), file=sys.stderr)
        return 130 if isinstance(error, KeyboardInterrupt) else 1
    finally:
        session.cleanup()
        session.redactor.scrub_tree(output)
        distributed_files = sorted(p for p in output.rglob("*") if p.is_file() and p.name != "SHA256SUMS")
        (output / "SHA256SUMS").write_text("".join(file_sha(p) + "  " + str(p.relative_to(output)) + "\n"
                                                   for p in distributed_files), encoding="utf-8")
        signal.signal(signal.SIGTERM, previous_sigterm)
        print("Result: " + str(output.relative_to(ROOT) if output.is_relative_to(ROOT) else "[EXTERNAL_RESULT_DIRECTORY]"), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
