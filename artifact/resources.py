"""Pinned public inputs for the real HotpotQA benchmark; never invent data.

Downloads and generated inputs stay in resources/cache, outside the distributable
artifact. The public lock contains URLs and content hashes, never host paths.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import random
import re
import shutil
import tempfile
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener


class ResourceError(RuntimeError):
    """A real pinned dependency is absent, malformed, or fails verification."""


def _sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     prefix=".resource-", delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, ensure_ascii=False, sort_keys=True,
                  separators=(",", ":"), allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def _https_url(url):
    parsed = urlparse(url)
    if (parsed.scheme != "https" or not parsed.hostname or parsed.username
            or parsed.password or parsed.fragment):
        raise ResourceError("Resource downloads require an HTTPS URL without credentials")
    return url


class _HTTPSRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _https_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _fetch_https(url, target, maximum_bytes):
    opener = build_opener(_HTTPSRedirects())
    request = Request(_https_url(url), headers={"User-Agent": "ToolSlack-public-artifact/1"})
    total = 0
    with opener.open(request, timeout=60) as response, Path(target).open("wb") as stream:
        _https_url(response.geturl())
        if response.status != 200:
            raise ResourceError(f"Resource HTTP status {response.status}: {url}")
        for block in iter(lambda: response.read(1024 * 1024), b""):
            total += len(block)
            if total > maximum_bytes:
                raise ResourceError(f"Resource exceeded its locked size: {url}")
            stream.write(block)


def _record(record):
    _https_url(record.get("url", ""))
    if not re.fullmatch(r"[0-9a-f]{64}", record.get("sha256", "")):
        raise ResourceError("Every resource needs a frozen SHA256 from real bytes")
    if type(record.get("bytes")) is not int or record["bytes"] <= 0:
        raise ResourceError("Every resource needs its positive locked byte length")
    relative = PurePosixPath(record.get("path", ""))
    if (relative.is_absolute() or not relative.parts or ".." in relative.parts
            or "\\" in str(relative)):
        raise ResourceError("Resource paths must be safe relative filenames")
    return relative


def _verify(path, record):
    if Path(path).stat().st_size != record["bytes"] or _sha256(path) != record["sha256"]:
        raise ResourceError(f"Resource checksum mismatch: {path}; remove the corrupt cache file and retry")


def _download(record, blobs, *, offline=False):
    """Content-addressed cache: different versions never replace one another."""
    _record(record)
    blobs = Path(blobs)
    blobs.mkdir(parents=True, exist_ok=True)
    target = blobs / record["sha256"]
    if target.is_file():
        _verify(target, record)
        return target
    if offline:
        raise ResourceError(f"Offline resource unavailable: {record['url']} (SHA256 {record['sha256']})")
    descriptor, name = tempfile.mkstemp(prefix=".download-", suffix=".part", dir=blobs)
    os.close(descriptor)
    temporary = Path(name)
    try:
        _fetch_https(record["url"], temporary, record["bytes"])
        _verify(temporary, record)
        if target.exists():
            _verify(target, record)
        else:
            temporary.replace(target)
        return target
    except Exception as error:
        if isinstance(error, ResourceError):
            raise
        raise ResourceError(f"Cannot download real pinned resource {record['url']}: {error}") from error
    finally:
        temporary.unlink(missing_ok=True)


def _materialize(record, directory, blobs, *, offline=False):
    relative = _record(record)
    target = Path(directory).joinpath(*relative.parts)
    if target.exists():
        _verify(target, record)
        return target
    source = _download(record, blobs, offline=offline)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".copy-", dir=target.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        shutil.copyfile(source, temporary)
        _verify(temporary, record)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def _pairs(value, first, second, name):
    if isinstance(value, dict):
        left, right = value.get(first), value.get(second)
        if not isinstance(left, list) or not isinstance(right, list) or len(left) != len(right):
            raise ResourceError(f"Malformed HotpotQA {name} columns")
        return [[a, b] for a, b in zip(left, right)]
    if isinstance(value, list):
        result = []
        for item in value:
            if isinstance(item, dict):
                result.append([item.get(first), item.get(second)])
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                result.append(list(item))
            else:
                raise ResourceError(f"Malformed HotpotQA {name} pair")
        return result
    raise ResourceError(f"Missing HotpotQA {name}")


def _hotpot_row(row):
    """Restore the original public HotpotQA JSON schema from HF parquet rows."""
    identifier = row.get("_id", row.get("id"))
    if not isinstance(identifier, str) or not identifier:
        raise ResourceError("HotpotQA rows require a nonempty string ID")
    if not isinstance(row.get("question"), str) or not isinstance(row.get("answer"), str):
        raise ResourceError("HotpotQA question and gold answer must be strings")
    context = _pairs(row.get("context"), "title", "sentences", "context")
    for title, sentences in context:
        if (not isinstance(title, str) or not isinstance(sentences, list)
                or not all(isinstance(sentence, str) for sentence in sentences)):
            raise ResourceError("HotpotQA context must contain [title, list[str]]")
    supporting = _pairs(row.get("supporting_facts"), "title", "sent_id", "supporting facts")
    for title, index in supporting:
        if not isinstance(title, str) or type(index) is not int:
            raise ResourceError("HotpotQA supporting facts require [title, int] pairs")
    result = {"_id": identifier, "question": row["question"], "answer": row["answer"],
              "supporting_facts": supporting, "context": context}
    for key in ("type", "level"):
        if key in row:
            if not isinstance(row[key], str):
                raise ResourceError(f"HotpotQA {key} must be a string")
            result[key] = row[key]
    return result


def _read_parquet(path):
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise ResourceError("HotpotQA preprocessing requires pyarrow==21.0.0; run the artifact dependency bootstrap") from error
    for batch in parquet.ParquetFile(path).iter_batches(batch_size=512):
        yield from batch.to_pylist()


def _prepare_rows(rows, directory, dataset_metadata):
    directory = Path(directory)
    gold, identifiers = [], set()
    source_reference_anomalies = 0
    for raw in rows:
        row = _hotpot_row(raw)
        if row["_id"] in identifiers:
            raise ResourceError(f"Duplicate HotpotQA task ID: {row['_id']}")
        identifiers.add(row["_id"])
        context_lengths = {title: len(sentences) for title, sentences in row["context"]}
        source_reference_anomalies += sum(title not in context_lengths or index < 0
                                         or index >= context_lengths.get(title, 0)
                                         for title, index in row["supporting_facts"])
        gold.append(row)
    if len(gold) != dataset_metadata["expected_rows"]:
        raise ResourceError(f"HotpotQA row count {len(gold)} differs from lock {dataset_metadata['expected_rows']}")
    inputs = directory / "agent_inputs/distractor.jsonl"
    inputs.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".inputs-", dir=inputs.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            for row in gold:
                safe = {key: row[key] for key in ("_id", "question", "context")}
                stream.write(json.dumps(safe, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
        temporary.replace(inputs)
    finally:
        temporary.unlink(missing_ok=True)
    labels = directory / "evaluation/hotpot_dev_distractor_v1.json"
    _write_json(labels, gold)
    manifest = {"schema": "toolslack.prepared-hotpot-resources.v1", "dataset": dataset_metadata,
                "row_count": len(gold), "agent_fields": ["_id", "question", "context"],
                "gold_policy": "answer/supporting_facts appear only in the evaluation file consumed by the scorer",
                "source_gold_reference_anomalies": source_reference_anomalies,
                "source_gold_policy": "Official source annotations are preserved unchanged, including out-of-range supporting references",
                "derived_files": [dict(path=str(path.relative_to(directory)), sha256=_sha256(path), bytes=path.stat().st_size)
                                  for path in (inputs, labels)]}
    _write_json(directory / "resource-manifest.json", manifest)
    return sorted(identifiers)


def _input_ids(directory):
    path = Path(directory) / "agent_inputs/distractor.jsonl"
    if not path.is_file():
        raise ResourceError(f"Dataset must include label-free agent_inputs/distractor.jsonl: {directory}")
    identifiers = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if set(row) != {"_id", "question", "context"}:
                raise ResourceError("Agent input must contain only _id, question, context; gold labels are forbidden")
            if not isinstance(row["_id"], str) or not row["_id"]:
                raise ResourceError("Agent input IDs must be nonempty strings")
            identifiers.append(row["_id"])
    if not identifiers or len(identifiers) != len(set(identifiers)):
        raise ResourceError("Dataset needs nonempty unique task IDs")
    for relative in ("evaluation/hotpot_dev_distractor_v1.json", "evaluator/hotpot_evaluate_v1.py"):
        if not (Path(directory) / relative).is_file():
            raise ResourceError(f"Dataset override is incomplete: missing {relative}")
    return identifiers


def _cohort(identifiers, calibration_count=3, evaluation_count=16, seed=20260930):
    ids = list(identifiers)
    if not ids or len(ids) != len(set(ids)) or not all(isinstance(value, str) and value for value in ids):
        raise ResourceError("Cohort source IDs must be nonempty and unique")
    if min(calibration_count, evaluation_count) <= 0 or len(ids) < calibration_count + evaluation_count:
        raise ResourceError("Not enough distinct real tasks for disjoint calibration/evaluation cohorts")
    ids.sort()
    random.Random(seed).shuffle(ids)
    calibration = ids[:calibration_count]
    evaluation = ids[calibration_count:calibration_count + evaluation_count]
    return {"schema": "toolslack.public-cohort.v1", "seed": seed,
            "claim": "New public deterministic artifact cohort; not the original paper sample.",
            "selection": "lexicographically sorted IDs, Python random.Random(seed).shuffle, disjoint prefixes",
            "calibration_task_ids": calibration, "evaluation_task_ids": evaluation,
            "datasets": {"langgraph": {"stages": {"smoke": {"task_ids": calibration},
                                                    "pilot": {"task_ids": evaluation}}}}}


def _validate_subset(path, available, minimum):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    try:
        stages = data.get("datasets", data.get("frameworks", {}))["langgraph"]["stages"]
        ids = list(stages.get("smoke", {}).get("task_ids", [])) + list(stages["pilot"]["task_ids"])
    except (KeyError, TypeError) as error:
        raise ResourceError("Subset must have datasets.langgraph.stages.smoke/pilot.task_ids") from error
    if (not all(isinstance(value, str) and value for value in ids)
            or len(ids) != len(set(ids)) or len(ids) < minimum):
        raise ResourceError("Subset needs enough unique disjoint calibration/evaluation IDs")
    if not set(ids).issubset(set(available)):
        raise ResourceError("Subset references tasks absent from the real dataset")


def _tokenizer_valid(directory):
    directory = Path(directory)
    for name in ("tokenizer.json", "tokenizer_config.json", "config.json"):
        if not (directory / name).is_file():
            raise ResourceError(f"Local tokenizer override is missing {name}: {directory}")
    config = json.loads((directory / "tokenizer_config.json").read_text(encoding="utf-8"))
    if not config.get("chat_template"):
        raise ResourceError("The pinned Qwen tokenizer requires its native chat_template")


def ensure_resources(args, root):
    """Return benchmark environment paths, using explicit local overrides first.

    This function downloads public data/tokenizer resources only. It never fetches
    model weights, creates a service, or substitutes a synthetic workload.
    """
    root = Path(root).resolve()
    lock_path = root / "resources/resource-lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("schema") != "toolslack.public-resource-lock.v1":
        raise ResourceError("Unknown public resource lock schema")
    fingerprint = _sha256(lock_path)[:16]
    cache = root / "resources/cache"
    blobs = cache / "downloads"
    offline = bool(getattr(args, "offline", False))
    calibration_count = int(getattr(args, "cal_tasks", 3))
    evaluation_count = int(getattr(args, "eval_tasks", 16))
    if calibration_count < 3 or evaluation_count < 4:
        raise ResourceError("Benchmark requires at least 3 calibration and 4 evaluation tasks")
    override = getattr(args, "data_root", None) or getattr(args, "dataset_root", None)
    if override:
        dataset = Path(override).expanduser().resolve()
    else:
        dataset = cache / f"hotpotqa-distractor-{lock['dataset']['revision'][:12]}-{fingerprint}"
        for record in lock["dataset"]["files"] + lock["evaluator"]["files"]:
            _materialize(record, dataset, blobs, offline=offline)
        manifest = dataset / "resource-manifest.json"
        if manifest.exists():
            existing = json.loads(manifest.read_text(encoding="utf-8"))
            if existing.get("dataset") != lock["dataset"] or existing.get("row_count") != lock["dataset"]["expected_rows"]:
                raise ResourceError("Prepared dataset manifest differs from the pinned lock")
            for record in existing["derived_files"]:
                if not (dataset / record["path"]).is_file():
                    raise ResourceError("Prepared cache is incomplete; remove this versioned dataset directory and retry")
                _verify(dataset / record["path"], record)
        else:
            parquet_record = next(record for record in lock["dataset"]["files"] if record["path"].endswith(".parquet"))
            _prepare_rows(_read_parquet(dataset / parquet_record["path"]), dataset, lock["dataset"])
        _write_json(dataset / "SOURCE_AND_LICENSE.json", {"dataset": {
            key: lock["dataset"][key] for key in ("repository", "revision", "configuration", "split", "source_url", "mirror_url", "license", "license_url")},
            "evaluator": {key: lock["evaluator"][key] for key in ("repository", "revision", "license")},
            "conversion": "HF parquet normalized to original HotpotQA JSON; label-free agent input emitted separately"})
    available = _input_ids(dataset)
    supplied_subset = getattr(args, "subsets", None)
    if supplied_subset:
        subsets = Path(supplied_subset).expanduser().resolve()
    else:
        # Keep generated cohorts in our writable cache, including with a user dataset.
        source_digest = _sha256(dataset / "agent_inputs/distractor.jsonl")[:16]
        subsets = cache / "cohorts" / f"hotpotqa-{source_digest}-cal{calibration_count}-eval{evaluation_count}-seed20260930.json"
        expected = _cohort(available, calibration_count, evaluation_count)
        if subsets.exists() and json.loads(subsets.read_text(encoding="utf-8")) != expected:
            raise ResourceError("Cached public cohort differs from its deterministic definition")
        if not subsets.exists():
            _write_json(subsets, expected)
    _validate_subset(subsets, available, calibration_count + evaluation_count)
    supplied_tokenizer = getattr(args, "tokenizer", None)
    if supplied_tokenizer:
        tokenizer = Path(supplied_tokenizer).expanduser().resolve()
    else:
        tokenizer = cache / f"qwen3-8b-tokenizer-{lock['tokenizer']['revision'][:12]}-{fingerprint}"
        for record in lock["tokenizer"]["files"]:
            _materialize(record, tokenizer, blobs, offline=offline)
        _write_json(tokenizer / "resource-manifest.json", {"schema": "toolslack.tokenizer-resources.v1",
                                                        **lock["tokenizer"]})
    _tokenizer_valid(tokenizer)
    return {"TOOLSLACK_DATASET_ROOT": str(dataset), "TOOLSLACK_SUBSETS": str(subsets),
            "TOOLSLACK_TOKENIZER": str(tokenizer)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--data-root", "--dataset-root", dest="data_root")
    parser.add_argument("--subsets")
    parser.add_argument("--tokenizer")
    parser.add_argument("--cal-tasks", type=int, default=3)
    parser.add_argument("--eval-tasks", type=int, default=16)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    try:
        print(json.dumps(ensure_resources(args, args.root), sort_keys=True))
    except (ResourceError, OSError, ValueError) as error:
        parser.exit(1, f"Resource preparation failed: {error}\n")


if __name__ == "__main__":
    main()
