"""Real CPU HTTP contract for controller-owned optional-memory scheduling.

Native install, weights and hardware are explicit doubles. The proxy request
uses an actual loopback server and the default generated worker configuration.
Nothing in these tests is GPU performance evidence.
"""
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from artifact import native


@dataclass
class FixtureProfile:
    model_path: str
    server_python: str
    server_cwd: str
    random_seed: int


def default_plan():
    """Exercise plan assembly after separately-tested hardware/source gates."""
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        output, checkpoint = root / "output", root / "checkpoint"
        output.mkdir(); checkpoint.mkdir()
        for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
            (checkpoint / name).write_text("{}\n")
        (checkpoint / "model.safetensors.index.json").write_text(json.dumps(
            {"weight_map": {"test": "fixture.safetensors"}}))
        (checkpoint / "fixture.safetensors").write_bytes(b"CPU fixture, no actual model")
        source = root / "vendor/native_engine"
        source.mkdir(parents=True)
        (source / "SOURCE_MANIFEST.json").write_text('{"files":[]}\n')
        args = SimpleNamespace(exclusive_gpus=True, gpu_indices="0", engine_python=Path(sys.executable),
            model_path=checkpoint, offline=True, engine_port=34200, proxy_port=34203, memory_workers=3)
        original_read = Path.read_text
        def read_text(path, *positional, **keyword):
            if path == Path("/proc/meminfo"):
                return "MemAvailable: " + str(160 * 1024 * 1024) + " kB\n"
            return original_read(path, *positional, **keyword)
        with patch.object(native, "check_empty_gpu", return_value="GPU-11111111-1111-1111-1111-111111111111"), \
                patch.object(native, "verify_frozen_sources"), \
                patch.object(native, "verify_native_interpreter", return_value={"cpu_test_double": True}), \
                patch.object(native, "verify_model_weights"), \
                patch.object(native.Path, "read_text", read_text), \
                patch.dict(sys.modules, {"server_profile": SimpleNamespace(ServerProfile=FixtureProfile)}):
            return native.prepare_default_plan(args, root, output,
                        {"TOOLSLACK_TOKENIZER": str(checkpoint)})


class DefaultProxyContractTests(unittest.IsolatedAsyncioTestCase):
    def test_generated_proxies_leave_optional_memory_admission_to_controller(self):
        plan = default_plan()
        for stage in ("calibration_proxy", "proxy"):
            argv = plan[stage]["argv"]
            self.assertEqual(argv.count("--optional-memory-workers"), 1)
            self.assertEqual(argv[argv.index("--optional-memory-workers") + 1], "0")

    async def test_default_proxy_accepts_controller_optional_memory_without_duplicate_budget_metadata(self):
        plan = default_plan()
        argv = plan["proxy"]["argv"]
        workers = int(argv[argv.index("--optional-memory-workers") + 1])
        backend = Path(__file__).resolve().parents[1] / "backend/proxy_v16"
        sys.path.insert(0, str(backend))
        import model_proxy as proxy
        from test_terminal_http import Harness
        original_serve = proxy.serve
        async def configured_serve(args):
            args.optional_memory_workers = workers
            await original_serve(args)
        with patch.object(proxy, "serve", configured_serve):
            harness = await Harness().start()
        self.addAsyncCleanup(harness.close)
        headers = {"x-toolslack-request-id": "optional-controller-owned",
                   "x-toolslack-request-kind": "memory"}
        # The real benchmark sends no x-toolslack-maintenance header because
        # its own controller already handles deadline/worker feasibility.
        async with harness.client.post(harness.url + "/v1/chat/completions",
                    json=harness.body(), headers=headers) as response:
            self.assertEqual(response.status, 200, await response.text())
        self.assertEqual(harness.chat_calls, 1)
        payload = next(row for row in harness.events() if row["event"] == "request_payload")
        self.assertEqual(payload["system_priority"], 0)
        self.assertFalse(any(row["event"] == "request_invalid" for row in harness.events()))


if __name__ == "__main__":
    unittest.main()
