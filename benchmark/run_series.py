"""Guard-workload entry: independent calibration, smoke, then balanced arms.

The existing guard owns GPU restoration. This program never launches/stops an
engine or occupancy worker. Its endpoints and service identity come from that
guard's TOOLSLACK_* environment. Every completed cell is checkpointed at once.
"""
from __future__ import annotations
import argparse
import asyncio
from copy import deepcopy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import sys

from .langgraph_hotpot import calibrate, measure, ARMS, SCHEMA


def subset_ids(path):
    data=json.loads(Path(path).read_text())
    group=data.get("datasets",data.get("frameworks",{}))["langgraph"]
    stages=group["stages"]
    return list(dict.fromkeys(list(stages.get("smoke",{}).get("task_ids",[]))+
                             list(stages["pilot"]["task_ids"])))


def build_config(output):
    model=os.environ.get("TOOLSLACK_MODEL","Qwen3-8B")
    profile=Path(os.environ["TOOLSLACK_SERVICE_PROFILE"])
    service_sha=hashlib.sha256(profile.read_bytes()).hexdigest()
    data_root=Path(os.environ["TOOLSLACK_DATASET_ROOT"])
    subset=os.environ["TOOLSLACK_SUBSETS"]
    count=int(os.environ.get("TOOLSLACK_EVAL_TASKS","16"))
    calibration_count=int(os.environ.get("TOOLSLACK_CAL_TASKS","3"))
    if calibration_count<3 or count<4:
        raise ValueError("At least three disjoint calibration tasks and four evaluation tasks required")
    selection=os.environ.get("TOOLSLACK_SELECTION","frozen-subset")
    if selection=="long-context":
        # Select using input size alone before observing outcomes, not QPS/labels.
        rows=[]
        with (data_root/"agent_inputs/distractor.jsonl").open() as source:
            for line in source:
                row=json.loads(line)
                rows.append((len(json.dumps(row["context"],ensure_ascii=False)),row["_id"]))
        rows.sort(key=lambda row:(-row[0],row[1]))
        ids=[identifier for _,identifier in rows[:calibration_count+count]]
        random.Random(20260930).shuffle(ids)
    elif selection=="frozen-subset":
        ids=subset_ids(subset)
    else:
        raise ValueError("Selection must be frozen-subset or long-context")
    if len(ids)<calibration_count+count:
        raise ValueError("Frozen input list too short for requested disjoint sets")
    config=dict(proxy_url=os.environ["TOOLSLACK_PROXY_URL"],model=model,
        service_profile_sha256=service_sha,dataset_root=str(data_root),
        tokenizer=os.environ["TOOLSLACK_TOKENIZER"],
        calibration_task_ids=ids[:calibration_count],evaluation_task_ids=ids[calibration_count:calibration_count+count],
        tool_mode=os.environ.get("TOOLSLACK_TOOL_MODE","model"),
        benefit_reference=os.environ.get("TOOLSLACK_BENEFIT_REFERENCE","serial_native_memory_to_first_token"),
        concurrency=int(os.environ.get("TOOLSLACK_CONCURRENCY","4")),
        memory_workers=int(os.environ.get("TOOLSLACK_MEMORY_WORKERS","1")),
        foreground_limit=int(os.environ.get("TOOLSLACK_FOREGROUND_LIMIT","16")),
        memory_max_tokens=int(os.environ.get("TOOLSLACK_MEMORY_MAX_TOKENS","8192")),
        memory_trigger_tokens=int(os.environ.get("TOOLSLACK_MEMORY_TRIGGER_TOKENS","4096")),
        memory_summary_tokens=int(os.environ.get("TOOLSLACK_SUMMARY_TOKENS","384")),
        foreground_tokens=int(os.environ.get("TOOLSLACK_FOREGROUND_TOKENS","640")),
        max_scopes=int(os.environ.get("TOOLSLACK_MAX_SCOPES","3")),context_length=131072,
        request_timeout_s=float(os.environ.get("TOOLSLACK_REQUEST_TIMEOUT_S","1200")),
        safety_margin_s=float(os.environ.get("TOOLSLACK_SAFETY_MARGIN_S","0.05")),
        seed=20260930,environment_key="guard-owned-sglang-host",profile_load_key="separate-calibration-isolated",
        gpu_indices=[int(x) for x in os.environ.get("TOOLSLACK_GPU_INDICES","").split(",") if x.strip()],
        enable_tiering=False,input_selection=selection,
        selection_rule="frozen existing subset" if selection=="frozen-subset" else "largest real official input-context character lengths; label/outcome blind cohort")
    gate_path=Path(os.environ["TOOLSLACK_COST_PROFILE"]).with_name("KV_LIFECYCLE_GATE.json")
    if gate_path.is_file():
        gate=json.loads(gate_path.read_text())
        if gate.get("state")=="passed" and gate.get("service_profile_sha256")==service_sha:
            lead=gate.get("measured_transfer_lead_s")
            if isinstance(lead,(int,float)) and lead>0:
                config.update(enable_tiering=True,transfer_lead_s=lead,
                              lifecycle_gate_sha256=hashlib.sha256(gate_path.read_bytes()).hexdigest())
    return config


def save(path,value):
    target=Path(path);temporary=target.with_suffix(target.suffix+".new")
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+"\n")
    temporary.replace(target)


def flush_receipt(response):
    """Accept the owned SGLang revision's explicit successful text receipt."""
    response.raise_for_status()
    try:
        receipt=response.json()
    except (ValueError, json.JSONDecodeError):
        text=response.text
        if not text.startswith("Cache flushed.\n"):
            raise RuntimeError("Unrecognized native cache flush response")
        receipt=dict(native_text_receipt=text)
    if receipt is False or (isinstance(receipt,dict) and receipt.get("ok") is False):
        raise RuntimeError(f"Native cache flush not confirmed: {receipt}")
    return receipt


async def flush_owned_engine():
    import httpx
    url=os.environ.get("TOOLSLACK_ENGINE_URL")
    if not url:
        raise ValueError("Cache isolation requires the guard-owned engine URL")
    async with httpx.AsyncClient(timeout=180,trust_env=False) as client:
        response=await client.post(url.rstrip("/")+"/flush_cache?timeout=120",json={})
        receipt=flush_receipt(response)
    return dict(endpoint=url.rstrip("/")+"/flush_cache",native_receipt=receipt,
                after_confirmed_previous_native_drain=True)


async def run(args):
    owned = (os.environ.get("TOOLSLACK_ARTIFACT_OWNED_SERVICE") == "1"
             and os.environ.get("TOOLSLACK_ARTIFACT_EXCLUSIVE_GPU") == "1")
    if not args.prepare_only and not (os.environ.get("TOOLSLACK_RUNTIME_ROOT") or owned):
        raise RuntimeError("Measurement requires a restoring guard or explicitly owned, exclusive artifact service")
    output=Path(args.output or os.environ.get("TOOLSLACK_BENCHMARK_OUTPUT","benchmark_results"))
    output.mkdir(parents=True,exist_ok=False)
    config=build_config(output);save(output/"config.json",config)
    versions={}
    for package in ("langgraph","langmem","langchain-core","transformers","httpx"):
        try:
            versions[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package]=None
    save(output/"runtime.json",dict(python=sys.version,packages=versions,argv=sys.argv,
        guard_runtime_present=bool(os.environ.get("TOOLSLACK_RUNTIME_ROOT")),
        external_owned_service_scope=owned))
    if args.prepare_only:
        print(json.dumps(dict(state="prepared_only",config=str(output/"config.json"))))
        return
    if any(value is None for value in versions.values()):
        raise RuntimeError("Official agent runtime dependencies missing: "+str(versions))
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(config["tokenizer"],local_files_only=True,trust_remote_code=False)
    checkpoint=dict(schema=SCHEMA,state="calibrating",cells=[],cache_flushes=[],config=config)
    save(output/"series.json",checkpoint)
    checkpoint["cache_flushes"].append(await flush_owned_engine())
    calibration=await calibrate(config,tokenizer,output/"calibration")
    checkpoint.update(state="smoke",calibration_scope_count=len(calibration["scope_observations"]),
                      calibration_failures=calibration["failures"])
    save(output/"series.json",checkpoint)
    smoke=deepcopy(config);smoke["evaluation_task_ids"]=config["evaluation_task_ids"][:4]
    for arm in ("off","full"):
        checkpoint["cache_flushes"].append(await flush_owned_engine())
        summary=await measure(smoke,tokenizer,calibration,arm,output/f"smoke_{arm}")
        checkpoint["cells"].append(dict(stage="smoke",repeat=0,**summary));save(output/"series.json",checkpoint)
        print(json.dumps(dict(stage="smoke",arm=arm,task_qps=summary["task_qps"],
                             successful_tasks=summary["successful_tasks"],prepared_l1=summary["prepared_l1"],prepared_l2=summary["prepared_l2"])),flush=True)
        if not summary["drain_confirmed"]:
            checkpoint["state"]="native_drain_failed";save(output/"series.json",checkpoint)
            raise RuntimeError("Unconfirmed native drainage; stop before further cells")
    stage=args.stage or os.environ.get("TOOLSLACK_SERIES_STAGE","smoke")
    if stage=="full":
        checkpoint["state"]="measuring";save(output/"series.json",checkpoint)
        for repetition in range(int(os.environ.get("TOOLSLACK_REPEATS","2"))):
            order=ARMS if repetition%2==0 else tuple(reversed(ARMS))
            for arm in order:
                checkpoint["cache_flushes"].append(await flush_owned_engine())
                summary=await measure(config,tokenizer,calibration,arm,
                                      output/f"r{repetition:02d}_{arm}")
                checkpoint["cells"].append(dict(stage="main",repeat=repetition,**summary))
                save(output/"series.json",checkpoint)
                print(json.dumps(dict(stage="main",repeat=repetition,arm=arm,
                    task_qps=summary["task_qps"],successful_tasks=summary["successful_tasks"],
                    quality=summary["task_quality"],prepared_l1=summary["prepared_l1"],prepared_l2=summary["prepared_l2"])),flush=True)
                if not summary["drain_confirmed"]:
                    raise RuntimeError("Unconfirmed native drainage; stop before next cell")
    checkpoint["state"]="completed";save(output/"series.json",checkpoint)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output");parser.add_argument("--prepare-only",action="store_true")
    parser.add_argument("--stage",choices=("smoke","full"))
    asyncio.run(run(parser.parse_args()))


if __name__=="__main__":
    main()
