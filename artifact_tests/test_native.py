"""Native startup safety checks using CPU-only process and resource stubs."""
import hashlib
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from artifact import native
from artifact.runner import PreflightError, start_plan


def options(**changes):
    values = dict(exclusive_gpus=True, gpu_indices="2", engine_python=None,
                  model_path=None, offline=True, engine_port=34200,
                  proxy_port=34203, memory_workers=1)
    values.update(changes)
    return SimpleNamespace(**values)


def successful_query(stdout):
    return subprocess.CompletedProcess([], 0, stdout=stdout, stderr="")


class NativeGpuSafetyTests(unittest.TestCase):
    def test_unsupported_platform_fails_before_hardware_or_installation(self):
        for system, machine in (("Darwin", "arm64"), ("Linux", "aarch64"), ("Windows", "x86_64")):
            with self.subTest(system=system, machine=machine), \
                    patch.object(native.platform, "system", return_value=system), \
                    patch.object(native.platform, "machine", return_value=machine), \
                    patch.object(native.subprocess, "run") as command:
                with self.assertRaises(RuntimeError):
                    native.check_empty_gpu(2)
                command.assert_not_called()

    def check_gpu(self, memory="0", total="81920", apps="", gpu_rows=None):
        commands = [successful_query(gpu_rows or "GPU-11111111-1111-1111-1111-111111111111, " + memory + ", " + total + "\n"),
                    successful_query(apps)]
        with patch.object(native.platform, "system", return_value="Linux"), \
                patch.object(native.platform, "machine", return_value="x86_64"), \
                patch.object(native.subprocess, "run", side_effect=commands):
            return native.check_empty_gpu(2)

    def test_other_gpu_process_does_not_change_selected_gpu_scope(self):
        self.assertEqual(self.check_gpu(apps="GPU-22222222-2222-2222-2222-222222222222, 9876\n"),
                         "GPU-11111111-1111-1111-1111-111111111111")

    def test_selected_gpu_process_blocks_even_with_tiny_memory_use(self):
        with self.assertRaises(RuntimeError):
            self.check_gpu(memory="1", apps="GPU-11111111-1111-1111-1111-111111111111, 9876\n")

    def test_existing_allocations_and_insufficient_capacity_are_rejected(self):
        for memory, total in (("1025", "81920"), ("0", "44999")):
            with self.subTest(memory=memory, total=total), self.assertRaises(RuntimeError):
                self.check_gpu(memory=memory, total=total)

    def test_nonfinite_or_negative_hardware_measurements_fail_closed(self):
        for memory, total in (("NaN", "81920"), ("-1", "81920"), ("0", "NaN"),
                              ("0", "inf"), ("0", "-1")):
            with self.subTest(memory=memory, total=total), self.assertRaises(RuntimeError):
                self.check_gpu(memory=memory, total=total)

    def test_ambiguous_hardware_query_is_not_accepted(self):
        with self.assertRaises(RuntimeError):
            self.check_gpu(gpu_rows="GPU-a, 0, 81920\nGPU-b, 0, 81920\n")

    def test_failed_hardware_query_is_not_treated_as_an_empty_gpu(self):
        with patch.object(native.platform, "system", return_value="Linux"), \
                patch.object(native.platform, "machine", return_value="x86_64"), \
                patch.object(native.subprocess, "run", side_effect=FileNotFoundError("nvidia-smi")):
            with self.assertRaises(RuntimeError):
                native.check_empty_gpu(2)

    def test_default_backend_rejects_multiple_physical_gpus_before_install(self):
        for value in ("0,1", "", "-1", "GPU-11111111-1111-1111-1111-111111111111"):
            with self.subTest(value=value), patch.object(native, "check_empty_gpu") as probe, \
                    patch.object(native.subprocess, "run") as command:
                with self.assertRaises(RuntimeError):
                    native.prepare_default_plan(options(gpu_indices=value), Path("unused"), Path("unused"), {})
                probe.assert_not_called()
                command.assert_not_called()

    def test_independent_reservation_is_required_before_install(self):
        with patch.object(native, "check_empty_gpu") as probe, \
                patch.object(native.subprocess, "run") as command:
            with self.assertRaises(RuntimeError):
                native.prepare_default_plan(options(exclusive_gpus=False), Path("unused"), Path("unused"), {})
            probe.assert_not_called()
            command.assert_not_called()

    def test_occupied_gpu_fails_before_cuda_bootstrap_checkpoint_download_and_writes(self):
        download = Mock()
        commands = [successful_query("GPU-11111111-1111-1111-1111-111111111111, 0, 81920\n"),
                    successful_query("GPU-11111111-1111-1111-1111-111111111111, 9876\n")]
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(native.platform, "system", return_value="Linux"), \
                patch.object(native.platform, "machine", return_value="x86_64"), \
                patch.object(native.subprocess, "run", side_effect=commands) as command, \
                patch.dict(sys.modules, {"huggingface_hub": SimpleNamespace(snapshot_download=download)}):
            root = Path(temporary)
            output = root / "output"
            with self.assertRaises(RuntimeError):
                native.prepare_default_plan(options(), root, output, {})
            download.assert_not_called()
            self.assertEqual(command.call_count, 2)
            self.assertTrue(all(call.args[0][0] == "nvidia-smi" for call in command.call_args_list))
            self.assertFalse(output.exists())
            self.assertFalse((root / ".engine-venv").exists())

    def test_insufficient_host_ram_fails_before_source_install_and_download(self):
        download = Mock()
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(native, "check_empty_gpu", return_value="GPU-11111111-1111-1111-1111-111111111111"), \
                patch.object(native.Path, "read_text", return_value="MemAvailable: 1024 kB\n"), \
                patch.object(native.subprocess, "run") as command, \
                patch.dict(sys.modules, {"huggingface_hub": SimpleNamespace(snapshot_download=download)}):
            with self.assertRaises(RuntimeError):
                native.prepare_default_plan(options(), Path(temporary), Path(temporary) / "output", {})
            command.assert_not_called()
            download.assert_not_called()


class NativeSourceAndLaunchTests(unittest.TestCase):
    def test_changed_source_cannot_reach_engine_installation_or_checkpoint_download(self):
        download = Mock()
        original_read = Path.read_text
        def read_text(path, *args, **kwargs):
            if path == Path("/proc/meminfo"):
                return "MemAvailable: " + str(160 * 1024 * 1024) + " kB\n"
            return original_read(path, *args, **kwargs)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = root / "vendor/native_engine"
            source = base / "python/sglang/module.py"
            source.parent.mkdir(parents=True)
            source.write_bytes(b"changed = True\n")
            (base / "SOURCE_MANIFEST.json").write_text(json.dumps({"files": [{
                "path": "python/sglang/module.py", "sha256": hashlib.sha256(b"original = True\n").hexdigest()}]}))
            with patch.object(native, "check_empty_gpu", return_value="GPU-11111111-1111-1111-1111-111111111111"), \
                    patch.object(native.Path, "read_text", read_text), \
                    patch.object(native.subprocess, "run") as command, \
                    patch.dict(sys.modules, {"huggingface_hub": SimpleNamespace(snapshot_download=download)}):
                with self.assertRaises(RuntimeError):
                    native.prepare_default_plan(options(), root, root / "output", {})
                command.assert_not_called()
                download.assert_not_called()
                self.assertFalse((root / "output").exists())

    def test_frozen_source_bytes_are_verified_before_install(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = root / "vendor/native_engine"
            source = base / "python/sglang/module.py"
            source.parent.mkdir(parents=True)
            source.write_bytes(b"answer = 42\n")
            manifest = base / "SOURCE_MANIFEST.json"
            manifest.write_text(json.dumps({"files": [{"path": "python/sglang/module.py",
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}]}))
            native.verify_frozen_sources(root)
            source.write_bytes(b"answer = 41\n")
            with self.assertRaises(RuntimeError):
                native.verify_frozen_sources(root)
            source.unlink()
            with self.assertRaises(RuntimeError):
                native.verify_frozen_sources(root)

    def test_missing_source_manifest_cannot_start_a_default_engine(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(RuntimeError):
                native.verify_frozen_sources(Path(temporary))

    def test_launch_preserves_verified_native_transfer_and_dense_model_flags(self):
        argv = native.engine_argv("/local/python", "/local/model", 31000, 31002)
        self.assertEqual(argv[:3], ["/local/python", "-m", "sglang.launch_server"])
        for flag, value in (("--model-path", "/local/model"), ("--port", "31000"),
                            ("--nccl-port", "31002"), ("--tp-size", "1"),
                            ("--hicache-size", "128"), ("--hicache-write-policy", "write_back"),
                            ("--hicache-io-backend", "direct"), ("--hicache-mem-layout", "layer_first"),
                            ("--attention-backend", "triton"), ("--dtype", "bfloat16"),
                            ("--context-length", "131072")):
            self.assertEqual(argv.count(flag), 1)
            self.assertEqual(argv[argv.index(flag) + 1], value)
        self.assertIn("--enable-hierarchical-cache", argv)
        rope = json.loads(argv[argv.index("--json-model-override-args") + 1])
        self.assertEqual(rope["rope_scaling"], {"rope_type": "yarn", "factor": 4.0,
                                              "original_max_position_embeddings": 32768})
        self.assertNotIn("--hicache-storage-backend", argv)
        self.assertNotIn("--speculative-algorithm", argv)


class TrackedStartupRecheckTests(unittest.TestCase):
    def fixture(self, root):
        selected = "GPU-11111111-1111-1111-1111-111111111111"
        args = SimpleNamespace(exclusive_gpus=True, gpu_indices="2",
            engine_url="http://127.0.0.1:31000", proxy_url="http://127.0.0.1:31003")
        plan = dict(gpu_uuid=selected, engine=dict(argv=["python", "-m", "sglang.launch_server"]),
                    proxy=dict(argv=["python", "proxy.py"]))
        session = SimpleNamespace(children=[], log_readers=[], output=root,
            phase=Mock(), redactor=SimpleNamespace(text=lambda value: value))
        return session, args, plan, {"TOOLSLACK_GPU_INDICES": "2"}, selected

    def test_gpu_occupied_during_download_cannot_reach_engine_popen(self):
        with tempfile.TemporaryDirectory() as temporary, \
                patch("artifact.runner.endpoint_responds", return_value=False), \
                patch("artifact.native.check_empty_gpu", side_effect=RuntimeError("GPU became occupied")), \
                patch("artifact.runner.subprocess.Popen") as spawn:
            session, args, plan, env, _ = self.fixture(Path(temporary))
            with self.assertRaises(RuntimeError):
                start_plan(session, args, plan, env, "a" * 64)
            spawn.assert_not_called()
            self.assertEqual(session.children, [])

    def test_physical_gpu_identity_change_cannot_reach_engine_popen(self):
        with tempfile.TemporaryDirectory() as temporary, \
                patch("artifact.runner.endpoint_responds", return_value=False), \
                patch("artifact.native.check_empty_gpu", return_value="GPU-22222222-2222-2222-2222-222222222222"), \
                patch("artifact.runner.subprocess.Popen") as spawn:
            session, args, plan, env, _ = self.fixture(Path(temporary))
            with self.assertRaises(PreflightError):
                start_plan(session, args, plan, env, "a" * 64)
            spawn.assert_not_called()

    def test_spawn_uses_verified_physical_uuid_instead_of_cuda_numeric_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            session, args, plan, env, selected = self.fixture(Path(temporary))
            processes = [SimpleNamespace(stdout=[], poll=lambda: None),
                         SimpleNamespace(stdout=[], poll=lambda: None)]
            with patch("artifact.runner.endpoint_responds", return_value=False), \
                    patch("artifact.native.check_empty_gpu", return_value=selected) as recheck, \
                    patch("artifact.runner.native_probe", return_value={}), \
                    patch("artifact.runner.get_json", return_value={"status": "tokenizer_ready",
                        "capacity": {"kv_service_profile_sha256": "a" * 64}}), \
                    patch("artifact.runner.subprocess.Popen", side_effect=processes) as spawn:
                start_plan(session, args, plan, env, "a" * 64)
                recheck.assert_called_once_with(2)
                self.assertEqual(spawn.call_count, 2)
                for call in spawn.call_args_list:
                    self.assertEqual(call.kwargs["env"]["CUDA_VISIBLE_DEVICES"], selected)
                    self.assertTrue(call.kwargs["start_new_session"])
            for reader in session.log_readers:
                reader.join(timeout=2)


class ModelWeightIntegrityTests(unittest.TestCase):
    def fixture(self, root):
        model = root / "checkpoint"
        model.mkdir()
        contents = {"model.safetensors.index.json": b'{"weight_map":{"a":"shard-a.safetensors","b":"shard-b.safetensors"}}\n',
                    "shard-a.safetensors": b"first test shard\n",
                    "shard-b.safetensors": b"second test shard\n"}
        rows = []
        for name, value in contents.items():
            (model / name).write_bytes(value)
            rows.append(dict(path=name, bytes=len(value), sha256=hashlib.sha256(value).hexdigest()))
        lock = dict(repository=native.MODEL_ID, revision=native.MODEL_REVISION,
                    index=rows[0], shards=rows[1:])
        (root / "resources").mkdir()
        (root / "resources/model-weights-lock.json").write_text(json.dumps(lock))
        return model, lock

    def test_complete_pinned_index_and_all_shards_pass(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model, _ = self.fixture(root)
            native.verify_model_weights(model, root)

    def test_same_size_corruption_of_index_or_any_shard_is_detected(self):
        for name in ("model.safetensors.index.json", "shard-a.safetensors", "shard-b.safetensors"):
            with self.subTest(file=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                model, _ = self.fixture(root)
                file = model / name
                original = file.read_bytes()
                file.write_bytes(bytes([original[0] ^ 1]) + original[1:])
                self.assertEqual(file.stat().st_size, len(original))
                with self.assertRaisesRegex(RuntimeError, name.replace(".", r"\.")):
                    native.verify_model_weights(model, root)

    def test_missing_or_truncated_shard_is_detected(self):
        for missing in (True, False):
            with self.subTest(missing=missing), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                model, _ = self.fixture(root)
                shard = model / "shard-b.safetensors"
                if missing:
                    shard.unlink()
                else:
                    shard.write_bytes(shard.read_bytes()[:-1])
                with self.assertRaisesRegex(RuntimeError, "shard-b"):
                    native.verify_model_weights(model, root)

    def test_wrong_repository_or_revision_fails_before_weight_hashing(self):
        for field in ("repository", "revision"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                model, lock = self.fixture(root)
                lock[field] = "different"
                (root / "resources/model-weights-lock.json").write_text(json.dumps(lock))
                with patch.object(native, "content_sha") as hashing:
                    with self.assertRaises(RuntimeError):
                        native.verify_model_weights(model, root)
                    hashing.assert_not_called()


class NativeInterpreterIntegrityTests(unittest.TestCase):
    def fixture(self, root):
        (root / "requirements").mkdir()
        versions = {"torch": "2.9.1", "sgl-kernel": "0.3.20", "transformers": "4.57.1"}
        (root / "requirements/engine.in").write_text("# Frozen runtime fixture\n" +
            "\n".join(name + "==" + version for name, version in versions.items()) + "\n")
        versions["sglang"] = "0.1.dev33+g46ef0661e"
        return dict(python="3.11.13", packages=versions,
            torch_build={"__version__": "2.9.1+cu128", "cuda": "12.8", "hip": None})

    def test_matching_frozen_runtime_passes_without_importing_torch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            receipt = self.fixture(root)
            with patch.object(native.subprocess, "run", return_value=successful_query(json.dumps(receipt))) as probe:
                actual = native.verify_native_interpreter("/local/native/python", root)
            self.assertEqual(actual, receipt)
            argv = probe.call_args.args[0]
            self.assertEqual(argv[:2], ["/local/native/python", "-c"])
            self.assertEqual(set(json.loads(argv[3])), set(receipt["packages"]))
            self.assertNotIn("import torch", argv[2])
            self.assertNotIn("from torch", argv[2])

    def test_actual_metadata_script_reads_annotated_torch_build_constants(self):
        # Execute the actual -c script against static local metadata stubs.
        # Merely mocking its final JSON cannot catch omissions in AST parsing.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            receipt = self.fixture(root)
            with patch.object(native.subprocess, "run", return_value=successful_query(json.dumps(receipt))) as probe:
                native.verify_native_interpreter("/local/native/python", root)
            script = probe.call_args.args[0][2]
            version_file = root / "torch-version.py"
            version_file.write_text("from typing import Optional\n"
                "__version__ = '2.9.1+cu128'\n"
                "cuda: Optional[str] = '12.8'\n"
                "hip: Optional[str] = None\n")
            locate = Mock(return_value=version_file)
            output = io.StringIO()
            with patch("importlib.metadata.distribution", return_value=SimpleNamespace(locate_file=locate)) as distribution, \
                    patch("importlib.metadata.version", side_effect=receipt["packages"].__getitem__), \
                    patch.object(sys, "argv", ["metadata-probe", json.dumps(sorted(receipt["packages"]))]), \
                    redirect_stdout(output):
                exec(compile(script, "<native-metadata-probe>", "exec"), {})
            actual = json.loads(output.getvalue())
            self.assertEqual(actual["torch_build"], receipt["torch_build"])
            self.assertEqual(actual["packages"], receipt["packages"])
            distribution.assert_called_once_with("torch")
            locate.assert_called_once_with("torch/version.py")

    def test_python_package_version_or_missing_distribution_mismatch_fails(self):
        for label in ("python", "sglang", "sgl-kernel", "missing"):
            with self.subTest(mismatch=label), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                receipt = self.fixture(root)
                if label == "python":
                    receipt["python"] = "3.12.14"
                elif label == "missing":
                    del receipt["packages"]["sgl-kernel"]
                else:
                    receipt["packages"][label] = "different"
                with patch.object(native.subprocess, "run", return_value=successful_query(json.dumps(receipt))):
                    with self.assertRaises(RuntimeError):
                        native.verify_native_interpreter("/local/native/python", root)

    def test_wrong_cuda_or_cpu_torch_build_cannot_pass_metadata_only_validation(self):
        builds = [{"__version__": "2.9.1+cpu", "cuda": None, "hip": None},
                  {"__version__": "2.9.1+cu130", "cuda": "13.0", "hip": None},
                  {"__version__": "2.9.1+rocm6.4", "cuda": None, "hip": "6.4"}]
        for build in builds:
            with self.subTest(build=build), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                receipt = self.fixture(root)
                receipt["packages"]["torch"] = build["__version__"]
                receipt["torch_build"] = build
                with patch.object(native.subprocess, "run", return_value=successful_query(json.dumps(receipt))):
                    with self.assertRaises(RuntimeError):
                        native.verify_native_interpreter("/local/native/python", root)

    def test_failed_or_malformed_probe_is_not_accepted_as_verified_runtime(self):
        failures = [FileNotFoundError("interpreter"),
                    subprocess.CalledProcessError(1, ["python", "-c", "metadata"]),
                    subprocess.TimeoutExpired(["python", "-c", "metadata"], 30)]
        for failure in failures:
            with self.subTest(failure=type(failure).__name__), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                self.fixture(root)
                with patch.object(native.subprocess, "run", side_effect=failure):
                    with self.assertRaises(RuntimeError):
                        native.verify_native_interpreter("/local/native/python", root)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.fixture(root)
            with patch.object(native.subprocess, "run", return_value=successful_query("not JSON")):
                with self.assertRaises(RuntimeError):
                    native.verify_native_interpreter("/local/native/python", root)


if __name__ == "__main__":
    unittest.main()
