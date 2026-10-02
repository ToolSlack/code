"""Anonymous result records and conservative measured-series validation."""
from __future__ import annotations

import csv
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import socket
import statistics


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".new")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Redactor:
    """Redact before writing logs; benchmark-generated files are scrubbed too."""
    def __init__(self, root=None, extra_paths=()):
        paths = [str(Path.home()), str(Path.cwd())]
        if root:
            paths.append(str(Path(root).resolve()))
        paths.extend(str(p) for p in extra_paths if p)
        self.paths = sorted(set(p for p in paths if len(p) > 1), key=len, reverse=True)
        self.hostname = socket.gethostname()
        self.username = os.environ.get("USER") or os.environ.get("USERNAME")

    def text(self, value):
        value = str(value)
        for path in self.paths:
            value = value.replace(path, "[LOCAL_PATH]")
        # Personal paths outside the working directory may occur in tracebacks.
        value = re.sub(r"/(?:Users|home|root|mnt|opt|private|tmp|var|srv|scratch|workspace)(?:/[^\s\"'<>),;]+)*",
                       "[LOCAL_PATH]", value)
        value = re.sub(r"https?://(?:[^/@\s]+@)?[^/\s\"'<>]+", "[SERVICE_URL]", value)
        if self.hostname:
            value = value.replace(self.hostname, "[HOST]")
        if self.username and len(self.username) > 2:
            value = re.sub(r"(?<![\w-])" + re.escape(self.username) + r"(?![\w-])", "[USER]", value)
        return value

    def value(self, value):
        if isinstance(value, str):
            return self.text(value)
        if isinstance(value, dict):
            return {self.text(k): self.value(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self.value(v) for v in value]
        return value

    def scrub_tree(self, output):
        for path in Path(output).rglob("*"):
            if not path.is_file() or path.suffix not in (".json", ".jsonl", ".log", ".txt", ".md", ".csv"):
                continue
            original = path.read_text(encoding="utf-8", errors="replace")
            redacted = self.text(original)
            if original != redacted:
                path.write_text(redacted, encoding="utf-8")


def machine_summary():
    packages = {}
    for name in ("aiohttp", "langgraph", "langmem", "langchain-core", "transformers", "httpx", "torch", "sglang"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return dict(os=platform.system(), os_release=platform.release(),
                architecture=platform.machine(), logical_cpu_count=os.cpu_count(),
                python=platform.python_version(), packages=packages,
                hostname_recorded=False, username_recorded=False,
                personal_paths_recorded=False)


def measured_report(series, mode, repeats):
    """Never turn a prepared/partial run or failed drain into performance."""
    cells = series.get("cells", [])
    reasons = []
    if series.get("state") != "completed":
        reasons.append("series did not complete")
    if series.get("calibration_failures"):
        reasons.append("independent agent calibration reported failures")
    expected = [("smoke", 0, arm) for arm in ("off", "full")]
    if mode == "full":
        expected += [("main", repeat, arm) for repeat in range(repeats)
                     for arm in ("off", "full", "no_selector", "fifo", "no_kv")]
    observed = [(c.get("stage"), c.get("repeat"), c.get("arm")) for c in cells]
    if Counter(observed) != Counter(expected):
        reasons.append("expected measured cells are missing, duplicated, or unexpected")
    rows = []
    for cell in cells:
        qps = cell.get("task_qps")
        finite_qps = type(qps) in (int, float) and math.isfinite(qps) and qps > 0
        counts_ok = (type(cell.get("task_count")) is int and cell["task_count"] > 0 and
                     type(cell.get("successful_tasks")) is int and
                     0 <= cell["successful_tasks"] <= cell["task_count"])
        failures = cell["task_count"] - cell["successful_tasks"] if counts_ok else None
        valid = (cell.get("drain_confirmed") is True and finite_qps and counts_ok and failures == 0)
        if not valid:
            reasons.append("a measured cell has task failures, invalid throughput, or unconfirmed native drainage")
        row = {k: cell.get(k) for k in ("stage", "repeat", "arm", "task_count", "successful_tasks",
               "task_qps", "batch_wall_including_drain_s", "drain_confirmed", "mean_ttft_s",
               "mean_post_tool_ttft_s", "mean_tpot_s", "prepared_l1", "prepared_l2", "native_memory_calls",
               "task_quality", "e2e_latency_s", "consumed_levels", "cache_tiering_enabled")}
        row.update(failed_tasks=failures, valid=valid)
        rows.append(row)
    valid = not reasons
    ratios = {}
    if valid and mode == "full":
        for arm in ("full", "no_selector", "fifo", "no_kv"):
            values = []
            for repeat in range(repeats):
                matched = {r["arm"]: r["task_qps"] for r in rows
                           if r["stage"] == "main" and r["repeat"] == repeat}
                values.append(matched[arm] / matched["off"])
            ratios[arm] = dict(per_repeat=values, median=statistics.median(values))
    return dict(schema="toolslack.artifact.report.v1", mode=mode,
                performance_evidence=valid, performance_valid=valid,
                publication_statistics_established=False,
                validity_reasons=list(dict.fromkeys(reasons)), cells=rows,
                main_qps_ratio_to_off=ratios,
                limitations=["smoke is a transport/correctness gate",
                             "calibration probes retain cache/order effects",
                             "small samples do not establish publication-level statistics",
                             "model tools contend on the serving GPU",
                             "no GPU measurement is inferred from CPU tests or preparation"])


def write_report(output, report):
    output = Path(output)
    save_json(output / "report.json", report)
    if report.get("cells"):
        columns = ["stage", "repeat", "arm", "task_count", "successful_tasks", "failed_tasks",
                   "task_qps", "batch_wall_including_drain_s", "drain_confirmed", "valid"]
        with (output / "cells.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(report["cells"])
    lines = ["# ToolSlack artifact result", "", f"Mode: {report.get('mode', 'unknown')}",
             f"State: {report.get('state', 'completed' if report.get('performance_valid') else 'invalid')}",
             f"Performance evidence: {str(bool(report.get('performance_evidence'))).lower()}", ""]
    for reason in report.get("validity_reasons", []):
        lines.append("- " + str(reason))
    if report.get("cells"):
        lines += ["", "| Stage | Repeat | Arm | Successful/total | QPS | Valid |",
                  "|---|---:|---|---:|---:|---|"]
        for row in report["cells"]:
            qps = row.get("task_qps")
            value = f"{qps:.5f}" if type(qps) in (int, float) and math.isfinite(qps) else "unavailable"
            lines.append(f"| {row['stage']} | {row['repeat']} | {row['arm']} | "
                         f"{row['successful_tasks']}/{row['task_count']} | {value} | {row['valid']} |")
    lines += ["", "CPU checks and preparation are not GPU performance evidence.", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
