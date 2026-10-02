"""CPU-only resource contracts, with explicit synthetic fixtures and no network."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from artifact import resources


def row(identifier):
    return dict(id=identifier, question="What connects the two documents?", answer="fixture gold",
                type="bridge", level="hard", context={"title": ["First", "Second"],
                "sentences": [["First evidence."], ["Second evidence."]]},
                supporting_facts={"title": ["First", "Second"], "sent_id": [0, 0]})


def record(path, body):
    return dict(path=path, url="https://example.org/pinned/" + path,
                sha256=hashlib.sha256(body).hexdigest(), bytes=len(body))


class ResourceTests(unittest.TestCase):
    def test_hf_columns_restore_original_format(self):
        value = resources._hotpot_row(row("task"))
        self.assertEqual(value["_id"], "task")
        self.assertEqual(value["context"], [["First", ["First evidence."]],
                                             ["Second", ["Second evidence."]]])
        self.assertEqual(value["supporting_facts"], [["First", 0], ["Second", 0]])
        self.assertNotIn("id", value)

    def test_struct_lists_also_restore_original_format(self):
        value = row("task")
        value["context"] = [dict(title="First", sentences=["First evidence."])]
        value["supporting_facts"] = [dict(title="First", sent_id=0)]
        self.assertEqual(resources._hotpot_row(value)["supporting_facts"], [["First", 0]])

    def test_gold_is_excluded_from_every_agent_input(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            resources._prepare_rows([row("a"), row("b")], target, dict(expected_rows=2))
            inputs = [json.loads(line) for line in (target / "agent_inputs/distractor.jsonl").read_text().splitlines()]
            self.assertTrue(all(set(value) == {"_id", "question", "context"} for value in inputs))
            self.assertNotIn("fixture gold", (target / "agent_inputs/distractor.jsonl").read_text())
            gold = json.loads((target / "evaluation/hotpot_dev_distractor_v1.json").read_text())
            self.assertTrue(all(value["answer"] == "fixture gold" and value["supporting_facts"] for value in gold))
            manifest = json.loads((target / "resource-manifest.json").read_text())
            self.assertEqual(manifest["row_count"], 2)
            for item in manifest["derived_files"]:
                resources._verify(target / item["path"], item)

    def test_duplicate_dataset_ids_fail_before_emitting_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(resources.ResourceError, "Duplicate"):
                resources._prepare_rows([row("a"), row("a")], directory, dict(expected_rows=2))
            self.assertFalse((Path(directory) / "agent_inputs/distractor.jsonl").exists())

    def test_locked_row_count_and_invalid_gold_type_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(resources.ResourceError, "row count"):
                resources._prepare_rows([row("a")], directory, dict(expected_rows=2))
        malformed = row("a")
        malformed["supporting_facts"]["sent_id"][0] = "bad"
        with self.assertRaisesRegex(resources.ResourceError, "pairs"):
            resources._hotpot_row(malformed)

    def test_original_source_gold_anomalies_are_preserved_not_repaired(self):
        anomalous = row("a")
        anomalous["supporting_facts"]["sent_id"][0] = 902
        with tempfile.TemporaryDirectory() as directory:
            resources._prepare_rows([anomalous], directory, dict(expected_rows=1))
            gold = json.loads((Path(directory) / "evaluation/hotpot_dev_distractor_v1.json").read_text())
            manifest = json.loads((Path(directory) / "resource-manifest.json").read_text())
            self.assertEqual(gold[0]["supporting_facts"][0], ["First", 902])
            self.assertEqual(manifest["source_gold_reference_anomalies"], 1)

    def test_cohort_is_order_independent_disjoint_and_compatible(self):
        ids = [f"task-{index:02}" for index in range(30)]
        first = resources._cohort(ids)
        second = resources._cohort(list(reversed(ids)))
        self.assertEqual(first, second)
        self.assertEqual(len(first["calibration_task_ids"]), 3)
        self.assertEqual(len(first["evaluation_task_ids"]), 16)
        self.assertFalse(set(first["calibration_task_ids"]) & set(first["evaluation_task_ids"]))
        stages = first["datasets"]["langgraph"]["stages"]
        self.assertEqual(stages["smoke"]["task_ids"] + stages["pilot"]["task_ids"],
                         first["calibration_task_ids"] + first["evaluation_task_ids"])
        self.assertIn("not the original paper sample", first["claim"])

    def test_cohort_duplicates_and_insufficient_rows_fail(self):
        for ids in (["a", "a"], ["a", "b"]):
            with self.subTest(ids=ids), self.assertRaises(resources.ResourceError):
                resources._cohort(ids)

    def test_subset_duplicates_missing_ids_and_insufficient_rows_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "subset.json"
            valid = resources._cohort([f"task-{i}" for i in range(20)])
            resources._write_json(target, valid)
            resources._validate_subset(target, [f"task-{i}" for i in range(20)], 19)
            duplicate = deepcopy(valid)
            duplicate["datasets"]["langgraph"]["stages"]["pilot"]["task_ids"][0] = valid["calibration_task_ids"][0]
            resources._write_json(target, duplicate)
            with self.assertRaisesRegex(resources.ResourceError, "unique"):
                resources._validate_subset(target, [f"task-{i}" for i in range(20)], 19)
            resources._write_json(target, valid)
            with self.assertRaisesRegex(resources.ResourceError, "absent"):
                resources._validate_subset(target, ["other"], 19)
            with self.assertRaisesRegex(resources.ResourceError, "enough"):
                resources._validate_subset(target, [f"task-{i}" for i in range(20)], 21)

    def test_only_https_and_safe_relative_resource_paths(self):
        for url in ("http://example.org/a", "ftp://example.org/a", "https://name:secret@example.org/a"):
            with self.subTest(url=url), self.assertRaises(resources.ResourceError):
                resources._https_url(url)
        for filename in ("/absolute", "../outside", "nested/../../outside", "bad\\path"):
            with self.subTest(filename=filename), self.assertRaises(resources.ResourceError):
                resources._record(record(filename, b"data"))

    def test_download_hash_failure_is_not_cached_and_never_falls_back(self):
        with tempfile.TemporaryDirectory() as directory:
            item = record("dataset.parquet", b"expected bytes")
            def fetch(url, target, maximum):
                Path(target).write_bytes(b"wrong bytes")
            with patch.object(resources, "_fetch_https", fetch):
                with self.assertRaisesRegex(resources.ResourceError, "checksum"):
                    resources._download(item, directory)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_offline_missing_resource_fails_and_valid_cached_bytes_work(self):
        with tempfile.TemporaryDirectory() as directory:
            item = record("tokenizer.json", b"tokenizer")
            with self.assertRaisesRegex(resources.ResourceError, "Offline"):
                resources._download(item, directory, offline=True)
            path = Path(directory) / item["sha256"]
            path.write_bytes(b"tokenizer")
            with patch.object(resources, "_fetch_https", side_effect=AssertionError("network forbidden")):
                self.assertEqual(resources._download(item, directory, offline=True), path)
            path.write_bytes(b"corrupt")
            with self.assertRaisesRegex(resources.ResourceError, "checksum"):
                resources._download(item, directory, offline=True)

    def test_explicit_overrides_take_precedence_and_do_not_download(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "resources").mkdir()
            (root / "resources/resource-lock.json").write_text(json.dumps({"schema": "toolslack.public-resource-lock.v1"}))
            data = root / "custom-data"
            resources._prepare_rows([row(str(index)) for index in range(7)], data, dict(expected_rows=7))
            (data / "evaluator").mkdir()
            (data / "evaluator/hotpot_evaluate_v1.py").write_text("# Explicit test fixture only\n")
            tokenizer = root / "custom-tokenizer"
            tokenizer.mkdir()
            for name in ("config.json", "tokenizer.json"):
                (tokenizer / name).write_text("{}")
            (tokenizer / "tokenizer_config.json").write_text(json.dumps(dict(chat_template="fixture")))
            subset = root / "custom-subset.json"
            resources._write_json(subset, resources._cohort([str(index) for index in range(7)], 3, 4))
            args = SimpleNamespace(data_root=str(data), dataset_root="must-not-be-used", subsets=str(subset),
                                   tokenizer=str(tokenizer), cal_tasks=3, eval_tasks=4, offline=True)
            with patch.object(resources, "_fetch_https", side_effect=AssertionError("network forbidden")):
                env = resources.ensure_resources(args, root)
            self.assertEqual(env, {"TOOLSLACK_DATASET_ROOT": str(data.resolve()), "TOOLSLACK_SUBSETS": str(subset.resolve()),
                                   "TOOLSLACK_TOKENIZER": str(tokenizer.resolve())})

    def test_dataset_override_with_gold_in_agent_input_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "agent_inputs/distractor.jsonl"
            path.parent.mkdir()
            path.write_text(json.dumps(resources._hotpot_row(row("a"))) + "\n")
            with self.assertRaisesRegex(resources.ResourceError, "gold labels"):
                resources._input_ids(directory)

    def test_public_lock_is_path_free_pinned_and_tokenizer_only(self):
        lock = json.loads((Path(__file__).resolve().parents[1] / "resources/resource-lock.json").read_text())
        self.assertNotIn("/Users/", json.dumps(lock))
        self.assertEqual(lock["dataset"]["expected_rows"], 7405)
        self.assertEqual(lock["dataset"]["revision"], "1908d6afbbead072334abe2965f91bd2709910ab")
        self.assertEqual(lock["tokenizer"]["revision"], "b968826d9c46dd6066d109eabc6255188de91218")
        self.assertFalse(lock["tokenizer"]["model_weights_included"])
        self.assertTrue(all("safetensors" not in item["path"] for item in lock["tokenizer"]["files"]))
        for group in ("dataset", "evaluator", "tokenizer"):
            for item in lock[group]["files"]:
                resources._record(item)
                self.assertNotEqual(item["sha256"], "0" * 64)


if __name__ == "__main__":
    unittest.main()
