import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from artifact.report import Redactor, measured_report
from artifact.runner import (PreflightError, expand_plan, load_plan, main, parse_gpus,
                             parser, require_measurement_ownership, validate_native_probe,
                             validate_service_snapshots, stop_tracked_process)


def cell(stage="smoke", arm="off", repeat=0):
    return dict(stage=stage, arm=arm, repeat=repeat, task_count=4, successful_tasks=4,
                task_qps=2., batch_wall_including_drain_s=2., drain_confirmed=True)


def complete_series(mode="smoke", repeats=2):
    rows = [cell(arm="off"), cell(arm="full")]
    if mode == "full":
        for repeat in range(repeats):
            for arm in ("off", "full", "no_selector", "fifo", "no_kv"):
                row = cell("main", arm, repeat)
                row["task_qps"] = 2. if arm == "off" else 3.
                rows.append(row)
    return dict(state="completed", cells=rows, calibration_failures=[])


class RunnerSafetyTests(unittest.TestCase):
    def test_distinct_physical_gpu_indices_are_required(self):
        self.assertEqual(parse_gpus("0, 2"), [0, 2])
        for value in (None, "", "-1", "1,1", "GPU-uuid", "0,"):
            with self.assertRaises(PreflightError):
                parse_gpus(value)

    def test_external_measurement_requires_both_explicit_ownership_and_isolation(self):
        args = SimpleNamespace(mode="full", owned_service=False, service_plan=None, exclusive_gpus=False)
        for owned, exclusive in ((False, False), (True, False), (False, True)):
            args.owned_service, args.exclusive_gpus = owned, exclusive
            with self.assertRaises(PreflightError):
                require_measurement_ownership(args)
        args.owned_service = args.exclusive_gpus = True
        require_measurement_ownership(args)

    def test_tracked_startup_still_requires_gpu_isolation(self):
        args = SimpleNamespace(mode="smoke", owned_service=False, service_plan=Path("plan.json"), exclusive_gpus=False)
        with self.assertRaises(PreflightError):
            require_measurement_ownership(args)
        args.exclusive_gpus = True
        require_measurement_ownership(args)

    def test_unowned_measurement_fails_before_resources_or_network(self):
        with tempfile.TemporaryDirectory() as temporary, patch("artifact.runner.get_json") as network:
            code = main(["full", "--no-bootstrap", "--output", str(Path(temporary) / "fresh"),
                         "--engine-url", "http://127.0.0.1:32000", "--proxy-url", "http://127.0.0.1:32100"])
            self.assertEqual(code, 1)
            network.assert_not_called()
            self.assertIn('"performance_evidence": false', (Path(temporary) / "fresh/status.json").read_text())

    def test_nonexistent_native_request_is_not_terminal_proof(self):
        body = dict(request_id="probe", action="request_status", consumer_request_id="never-submitted", service_profile_sha256="a" * 64)
        receipt = dict(body, ok=True, service_epoch="epoch", engine_epoch="epoch", active=False, terminal_proof=False)
        validate_native_probe(receipt, body)
        for key, value in (("request_id", "wrong"), ("service_profile_sha256", "b" * 64),
                           ("engine_epoch", "old"), ("active", True), ("terminal_proof", True), ("ok", False)):
            changed = dict(receipt); changed[key] = value
            with self.assertRaises(PreflightError):
                validate_native_probe(changed, body)

    def snapshots(self):
        sha, cost = "a" * 64, "b" * 64
        health = dict(status="tokenizer_ready", capacity=dict(model_name="Qwen3-8B", context_window=131072,
                       kv_service_profile_sha256=sha, kv_cost_profile_sha256=cost))
        caps = dict(service_profile_sha256=sha, bounded_prefix_prefill=True, deadline_terminal_drain=True,
                    existing_prefix_registration=True)
        lifecycle = dict(quiescent=True, active_owners=0, guard_cleanup_required=False, held_prefill_slots=0)
        native = dict(service_profile_sha256=sha)
        return [health, caps, lifecycle, native, sha, cost, "Qwen3-8B"]

    def test_matching_profiles_and_quiescent_service_pass(self):
        validate_service_snapshots(*self.snapshots())

    def test_changed_service_cost_and_missing_capability_fail(self):
        for changed_group, key, value in (("capacity", "kv_service_profile_sha256", "wrong"),
                                         ("capacity", "kv_cost_profile_sha256", "wrong"),
                                         ("caps", "deadline_terminal_drain", False)):
            values = self.snapshots()
            target = values[0]["capacity"] if changed_group == "capacity" else values[1]
            target[key] = value
            with self.assertRaises(PreflightError):
                validate_service_snapshots(*values)

    def test_active_native_work_blocks_any_experiment(self):
        for key, value in (("quiescent", False), ("active_owners", 1), ("held_prefill_slots", 1), ("guard_cleanup_required", True)):
            values = self.snapshots(); values[2][key] = value
            with self.assertRaises(PreflightError):
                validate_service_snapshots(*values)

    def test_process_plan_expands_only_declared_resource_variables(self):
        self.assertEqual(expand_plan("${TOOLSLACK_TOKENIZER}/tokenizer.json", {"TOOLSLACK_TOKENIZER": "/cache/tokens"}), "/cache/tokens/tokenizer.json")
        for value in ("${HOME}", "${TOOLSLACK_MISSING}"):
            with self.assertRaises(PreflightError):
                expand_plan(value, {})

    def test_process_plan_rejects_shell_commands_and_missing_calibration_proxy(self):
        args = parser().parse_args(["smoke"])
        args.service_plan = None
        base = dict(schema="toolslack.artifact.service-plan.v1", engine=dict(argv=["python", "-m", "engine"]), proxy=dict(argv=["python", "-m", "proxy"]))
        self.assertIs(load_plan(args, base), base)
        bad = copy.deepcopy(base); bad["engine"]["argv"] = "python -m engine"
        with self.assertRaises(PreflightError):
            load_plan(args, bad)
        bad = copy.deepcopy(base); bad["calibration"] = {}
        with self.assertRaises(PreflightError):
            load_plan(args, bad)

    def test_process_plan_must_use_the_explicit_gpu_allocation(self):
        args = parser().parse_args(["smoke", "--gpu-indices", "0"])
        plan = dict(schema="toolslack.artifact.service-plan.v1", gpu_indices=[1],
                    engine=dict(argv=["python", "-m", "engine"]), proxy=dict(argv=["python", "-m", "proxy"]))
        with self.assertRaises(PreflightError):
            load_plan(args, plan)

    def test_untracked_service_cannot_be_stopped(self):
        with patch("artifact.runner.os.killpg") as signal_group:
            with self.assertRaises(PreflightError):
                stop_tracked_process(SimpleNamespace(children=[]), SimpleNamespace(pid=123))
            signal_group.assert_not_called()
    def test_log_redaction_covers_personal_paths_host_and_endpoint(self):
        redactor = Redactor("/artifact")
        redactor.hostname = "private-workstation"
        redactor.username = "someone"
        value = redactor.text('/Users/someone/project/run.py /mnt/private/model host=private-workstation http://10.2.3.4:9000/health someone')
        for private in ("someone", "private-workstation", "10.2.3.4", "/Users/", "/mnt/"):
            self.assertNotIn(private, value)
        self.assertIn("/health", value)


class ReportValidityTests(unittest.TestCase):
    def test_complete_smoke_has_measured_evidence_but_no_main_speedup(self):
        report = measured_report(complete_series(), "smoke", 2)
        self.assertTrue(report["performance_valid"])
        self.assertEqual(report["main_qps_ratio_to_off"], {})
        self.assertFalse(report["publication_statistics_established"])

    def test_full_report_uses_same_repeat_observations(self):
        report = measured_report(complete_series("full"), "full", 2)
        self.assertTrue(report["performance_valid"])
        self.assertEqual(report["main_qps_ratio_to_off"]["full"], dict(per_repeat=[1.5, 1.5], median=1.5))

    def test_preparation_or_partial_state_is_not_performance(self):
        for state in ("prepared_only", "calibrating", "native_drain_failed"):
            series = complete_series(); series["state"] = state
            self.assertFalse(measured_report(series, "smoke", 2)["performance_valid"])

    def test_failed_native_drain_invalidates_throughput(self):
        series = complete_series(); series["cells"][0]["drain_confirmed"] = False
        self.assertFalse(measured_report(series, "smoke", 2)["performance_evidence"])

    def test_task_failure_is_preserved_and_invalidates_formal_result(self):
        series = complete_series(); series["cells"][0]["successful_tasks"] = 3
        report = measured_report(series, "smoke", 2)
        self.assertFalse(report["performance_valid"])
        self.assertEqual(report["cells"][0]["failed_tasks"], 1)

    def test_missing_duplicate_and_unknown_cells_are_invalid(self):
        series = complete_series()
        for rows in (series["cells"][:1], series["cells"] + [cell()], [cell(arm=None), cell(arm="full")]):
            changed = dict(series, cells=rows)
            self.assertFalse(measured_report(changed, "smoke", 2)["performance_valid"])

    def test_nan_or_null_qps_cannot_be_reported_as_valid(self):
        for value in (None, float("nan"), float("inf"), -1, True):
            series = complete_series(); series["cells"][0]["task_qps"] = value
            self.assertFalse(measured_report(series, "smoke", 2)["performance_valid"])

    def test_agent_calibration_failures_cannot_be_hidden(self):
        series = complete_series(); series["calibration_failures"] = [dict(task_id="cal", error="failed")]
        self.assertFalse(measured_report(series, "smoke", 2)["performance_valid"])


if __name__ == "__main__":
    unittest.main()
