"""Unchanged historical HotpotQA protocol helpers, extracted by AST.
Source provenance is in reference_provenance.json. No historical result values
are imported by this module; only input/scoring/tokenization helpers are reused.
"""
from __future__ import annotations
import copy
import hashlib
import importlib.util
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence


FACT_PAIR_SCHEMA: dict[str, Any] = {
    "type": "array",
    "prefixItems": [{"type": "string"}, {"type": "integer"}],
    "items": False,
    "minItems": 2,
    "maxItems": 2,
}


STRUCTURED_OUTPUT_SCHEMAS: dict[str, dict[str, Any]] = {
    "plan": {
        "type": "object",
        "properties": {
            "queries": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "maxItems": 4,
            },
            "subquestions": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "maxItems": 8,
            },
        },
        "required": ["queries", "subquestions"],
        "additionalProperties": False,
    },
    "analysis": {
        "type": "object",
        "properties": {
            "findings": {"type": "string"},
            "supporting_facts": {
                "type": "array",
                "items": FACT_PAIR_SCHEMA,
            },
        },
        "required": ["findings", "supporting_facts"],
        "additionalProperties": False,
    },
    "answer": {
        "type": "object",
        "properties": {
            "answer": {"type": "string", "minLength": 1},
            "supporting_facts": {
                "type": "array",
                "items": FACT_PAIR_SCHEMA,
            },
        },
        "required": ["answer", "supporting_facts"],
        "additionalProperties": False,
    },
}


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def structured_response_format(kind: str) -> dict[str, Any]:
    if kind not in STRUCTURED_OUTPUT_SCHEMAS:
        raise ValueError(f"unknown structured-output kind: {kind}")
    return {
        "type": "json_schema",
        "json_schema": {
            "name": f"hotpot_{kind}",
            "strict": True,
            "schema": copy.deepcopy(STRUCTURED_OUTPUT_SCHEMAS[kind]),
        },
    }


def parse_structured_output(text: str, kind: str) -> dict[str, Any]:
    """Accept one complete JSON object and enforce the consumer contract."""

    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"{kind} output is not one complete JSON object: {error}"
        ) from error
    if not isinstance(value, dict):
        raise ValueError(f"{kind} output must be a JSON object")

    def facts_are_valid(raw: Any) -> bool:
        return isinstance(raw, list) and all(
            isinstance(item, list)
            and len(item) == 2
            and isinstance(item[0], str)
            and isinstance(item[1], int)
            and not isinstance(item[1], bool)
            for item in raw
        )

    if kind == "plan":
        if set(value) != {"queries", "subquestions"}:
            raise ValueError("plan JSON has unexpected or missing fields")
        if not all(
            isinstance(value[name], list)
            and value[name]
            and all(isinstance(item, str) for item in value[name])
            for name in ("queries", "subquestions")
        ):
            raise ValueError("plan JSON fields must be nonempty string arrays")
    elif kind == "analysis":
        if set(value) != {"findings", "supporting_facts"}:
            raise ValueError("analysis JSON has unexpected or missing fields")
        if not isinstance(value["findings"], str) or not facts_are_valid(
            value["supporting_facts"]
        ):
            raise ValueError("analysis JSON violates its consumer contract")
    elif kind == "answer":
        if set(value) != {"answer", "supporting_facts"}:
            raise ValueError("answer JSON has unexpected or missing fields")
        if (
            not isinstance(value["answer"], str)
            or not value["answer"].strip()
            or not facts_are_valid(value["supporting_facts"])
        ):
            raise ValueError("answer JSON violates its consumer contract")
    else:
        raise ValueError(f"unknown structured-output kind: {kind}")
    return value


def validate_model_plan(value: Mapping[str, Any], question: str, branches: int) -> dict[str, Any]:
    queries = value.get("queries")
    goals = value.get("subquestions")
    if not isinstance(queries, list) or not all(isinstance(x, str) for x in queries):
        queries = []
    if not isinstance(goals, list) or not all(isinstance(x, str) for x in goals):
        goals = []
    normalized_queries = [x.strip() for x in queries if x.strip()][:4] or [question]
    normalized_goals = [x.strip() for x in goals if x.strip()][:branches] or [question]
    return {"queries": normalized_queries, "subquestions": normalized_goals}


def rank_all_context(context: Sequence[Sequence[Any]], queries: Sequence[str]) -> list[dict[str, Any]]:
    """Rank every supplied distractor paragraph without dropping any input."""

    def tokenize(text: str) -> list[str]:
        return re.findall(r"\w+", text.lower())

    docs: list[tuple[str, Sequence[str], list[str]]] = []
    for raw in context:
        if len(raw) != 2 or not isinstance(raw[0], str) or not isinstance(raw[1], list):
            raise ValueError("HotpotQA context entry must be [title, sentence-list]")
        title, sentences = raw
        if not all(isinstance(sentence, str) for sentence in sentences):
            raise ValueError("HotpotQA sentences must be strings")
        docs.append((title, sentences, tokenize(title + " " + " ".join(sentences))))
    if not docs:
        raise ValueError("HotpotQA context is empty")
    terms = set(tokenize(" ".join(queries)))
    average_length = sum(len(tokens) for _, _, tokens in docs) / len(docs)
    document_frequency = {
        term: sum(term in tokens for _, _, tokens in docs) for term in terms
    }
    scored: list[tuple[float, int, dict[str, Any]]] = []
    for index, (title, sentences, tokens) in enumerate(docs):
        score = 0.0
        title_terms = set(tokenize(title))
        for term in terms:
            frequency = tokens.count(term)
            inverse = math.log(
                1 + (len(docs) - document_frequency[term] + 0.5)
                / (document_frequency[term] + 0.5)
            )
            denominator = frequency + 1.2 * (
                0.25 + 0.75 * len(tokens) / max(average_length, 1)
            )
            score += inverse * frequency * 2.2 / max(denominator, 1e-9)
            if term in title_terms:
                score += inverse
        materialized = {
            "title": title,
            "sentences": [
                {"sentence_id": sentence_id, "text": sentence}
                for sentence_id, sentence in enumerate(sentences)
            ],
        }
        scored.append((-score, index, materialized))
    ranked = [entry for _, _, entry in sorted(scored)]
    if len(ranked) != len(context):
        raise AssertionError("retrieval must preserve every supplied paragraph")
    return ranked


def assert_complete_tool_pairs(messages: Sequence[Mapping[str, Any]]) -> None:
    pending: set[str] = set()
    for message in messages:
        role = message.get("role")
        if pending and role != "tool":
            raise ValueError("unanswered tool call precedes a non-tool message")
        if role == "assistant" and message.get("tool_calls"):
            if pending:
                raise ValueError("nested unresolved tool-call batch")
            for call in message["tool_calls"]:
                call_id = call.get("id")
                if not isinstance(call_id, str) or not call_id or call_id in pending:
                    raise ValueError("tool-call IDs must be nonempty and unique")
                pending.add(call_id)
        elif role == "tool":
            call_id = message.get("tool_call_id")
            if call_id not in pending:
                raise ValueError("orphan or duplicate tool result")
            pending.remove(str(call_id))
    if pending:
        raise ValueError("snapshot contains an unresolved tool call")


def merge_leading_systems(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    wires = [copy.deepcopy(dict(message)) for message in messages]
    if not wires or wires[0].get("role") != "system":
        return wires
    merged = dict(wires[0])
    index = 1
    while index < len(wires) and wires[index].get("role") == "system":
        content = wires[index].get("content") or ""
        if content:
            merged["content"] = (merged.get("content") or "") + "\n\n" + str(content)
        index += 1
    return [merged, *wires[index:]]


def count_checkpoint_wire(
    tokenizer: Any, messages: Sequence[Mapping[str, Any]], model: str
) -> int:
    """Count the checkpoint's exact request serialization or fail closed.

    Qwen checkpoints publish a tokenizer chat template.  The released
    DeepSeek-V4 checkpoint instead publishes its reference ``encoding_dsv4``
    encoder next to the weights; using a whitespace/token heuristic here would
    invalidate both the LangMem trigger and the capacity check.
    """

    wires = merge_leading_systems(messages)
    if not wires:
        return 0
    if not any(message.get("role") == "user" for message in wires):
        wires.append({"role": "user", "content": ""})
    if getattr(tokenizer, "chat_template", None):
        encoded = tokenizer.apply_chat_template(
            wires,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        if hasattr(encoded, "get") and encoded.get("input_ids") is not None:
            encoded = encoded["input_ids"]
        return len(encoded)
    if model == "DeepSeek-V4-Flash-0731":
        encoder_path = Path(tokenizer.name_or_path) / "encoding" / "encoding_dsv4.py"
        if not encoder_path.is_file():
            raise ValueError(f"missing checkpoint encoder: {encoder_path}")
        spec = importlib.util.spec_from_file_location("encoding_dsv4", encoder_path)
        if spec is None or spec.loader is None:
            raise ValueError(f"cannot import checkpoint encoder: {encoder_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        rendered = module.encode_messages(wires, thinking_mode="chat")
        return len(tokenizer.encode(rendered, add_special_tokens=False))
    raise ValueError(f"no checkpoint-native chat serialization for {model}")


def score_hotpot_prediction(
    official: Any, prediction: Mapping[str, Any], gold: Mapping[str, Any]
) -> dict[str, float]:
    metrics = {
        name: 0.0
        for name in (
            "em",
            "f1",
            "prec",
            "recall",
            "sp_em",
            "sp_f1",
            "sp_prec",
            "sp_recall",
        )
    }
    official.update_answer(metrics, prediction["answer"], gold["answer"])
    official.update_sp(metrics, prediction["sp"], gold["supporting_facts"])
    metrics["joint_em"] = metrics["em"] * metrics["sp_em"]
    joint_precision = metrics["prec"] * metrics["sp_prec"]
    joint_recall = metrics["recall"] * metrics["sp_recall"]
    metrics["joint_f1"] = (
        2 * joint_precision * joint_recall / (joint_precision + joint_recall)
        if joint_precision + joint_recall
        else 0.0
    )
    return metrics


def zero_quality() -> dict[str, float]:
    return {
        name: 0.0
        for name in (
            "em",
            "f1",
            "prec",
            "recall",
            "sp_em",
            "sp_f1",
            "sp_prec",
            "sp_recall",
            "joint_em",
            "joint_f1",
        )
    }


def load_official_evaluator(dataset_root: Path) -> Any:
    evaluator_path = dataset_root / "evaluator" / "hotpot_evaluate_v1.py"
    spec = importlib.util.spec_from_file_location("hotpot_official", evaluator_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot import official evaluator: {evaluator_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

