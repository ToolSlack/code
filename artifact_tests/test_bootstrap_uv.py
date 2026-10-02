"""Contracts for the pip-free, fixed-wheel uv bootstrap; all fixtures are local."""
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import stat
import subprocess
import tempfile
import unittest
from unittest import mock
import warnings
import zipfile


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/bootstrap_uv.py"
SPEC = importlib.util.spec_from_file_location("toolslack_bootstrap_uv", SCRIPT)
uv = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(uv)


class DownloadResponse(io.BytesIO):
    status = 200

    def __init__(self, data, url):
        super().__init__(data)
        self.url = url

    def geturl(self):
        return self.url


class BootstrapUVTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name).resolve()
        self.binary = b"fixture executable bytes; never run without a mocked subprocess"
        self.filename = "uv-0.12.22-py3-none-test.whl"
        self.binary_path = "uv-0.12.22.data/scripts/uv"
        self.key = "Linux-x86_64-glibc"
        self.target = self.root / ".bootstrap/bin/uv"
        self.wheel = self.root / ".bootstrap/wheels" / self.filename
        self.version = subprocess.CompletedProcess([], 0, stdout="uv 0.12.22 (fixture build)\n", stderr="")

    def fixture(self, entries=None, cached=True):
        data = io.BytesIO()
        with warnings.catch_warnings(), zipfile.ZipFile(data, "w") as archive:
            warnings.simplefilter("ignore", UserWarning)
            for name, content in entries or [(self.binary_path, self.binary), ("../../outside", b"never extract")]:
                archive.writestr(name, content)
        self.wheel_bytes = data.getvalue()
        self.row = {
            "filename": self.filename,
            "url": "https://files.pythonhosted.org/packages/ab/cd/" + self.filename,
            "bytes": len(self.wheel_bytes),
            "sha256": hashlib.sha256(self.wheel_bytes).hexdigest(),
            "binary_path": self.binary_path,
        }
        self.manifest = {"schema": uv.SCHEMA, "version": uv.VERSION, "platforms": {self.key: self.row}}
        self.write_manifest()
        if cached:
            self.wheel.parent.mkdir(parents=True, exist_ok=True)
            self.wheel.write_bytes(self.wheel_bytes)

    def write_manifest(self):
        path = self.root / "requirements/bootstrap-wheels.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.manifest))

    def test_verified_cached_wheel_installs_only_binary_without_pip_or_network(self):
        self.fixture()
        with mock.patch.object(uv, "download_wheel") as download, mock.patch.object(uv.subprocess, "run", return_value=self.version) as run:
            result = uv.install(self.root, self.key)
        self.assertEqual(result, self.target)
        self.assertEqual(self.target.read_bytes(), self.binary)
        self.assertEqual(stat.S_IMODE(self.target.stat().st_mode), 0o755)
        self.assertFalse((self.root / "outside").exists())
        self.assertEqual(list(self.target.parent.iterdir()), [self.target])
        download.assert_not_called()
        argv = run.call_args.args[0]
        self.assertEqual(argv[1:], ["--version"])
        self.assertTrue(Path(argv[0]).name.startswith(".uv-binary-"))

    def test_matching_existing_binary_is_verified_before_execution_and_reused(self):
        self.fixture()
        self.target.parent.mkdir(parents=True)
        self.target.write_bytes(self.binary)
        self.target.chmod(0o755)
        before = self.target.stat().st_mtime_ns
        with mock.patch.object(uv.subprocess, "run", return_value=self.version) as run:
            uv.install(self.root, self.key)
        self.assertEqual(run.call_args.args[0], [str(self.target), "--version"])
        self.assertEqual(before, self.target.stat().st_mtime_ns)

    def test_unverified_existing_binary_is_never_executed(self):
        self.fixture()
        self.target.parent.mkdir(parents=True)
        self.target.write_bytes(b"untrusted existing executable")
        self.target.chmod(0o755)
        executed = []

        def check_command(argv, **kwargs):
            executed.append(argv[0])
            self.assertNotEqual(Path(argv[0]), self.target)
            self.assertEqual(Path(argv[0]).read_bytes(), self.binary)
            return self.version

        with mock.patch.object(uv.subprocess, "run", side_effect=check_command):
            uv.install(self.root, self.key)
        self.assertEqual(len(executed), 1)
        self.assertEqual(self.target.read_bytes(), self.binary)

    def test_corrupt_cached_wheel_is_rejected_before_any_executable(self):
        self.fixture()
        corrupt = bytearray(self.wheel_bytes)
        corrupt[-1] ^= 1
        self.wheel.write_bytes(corrupt)
        with mock.patch.object(uv.subprocess, "run") as run, mock.patch.object(uv, "download_wheel") as download:
            with self.assertRaisesRegex(uv.BootstrapError, "SHA256"):
                uv.install(self.root, self.key)
        run.assert_not_called()
        download.assert_not_called()
        self.assertFalse(self.target.exists())

    def test_missing_or_duplicate_expected_binary_rejected(self):
        for entries in [[("uv-other", self.binary)], [(self.binary_path, self.binary), (self.binary_path, self.binary)]]:
            with self.subTest(entries=entries):
                self.fixture(entries)
                with mock.patch.object(uv.subprocess, "run") as run:
                    with self.assertRaisesRegex(uv.BootstrapError, "exactly one"):
                        uv.install(self.root, self.key)
                run.assert_not_called()
                self.assertFalse(self.target.exists())

    def test_executable_member_must_be_regular_file(self):
        info = zipfile.ZipInfo(self.binary_path)
        info.create_system = 3
        info.external_attr = (stat.S_IFLNK | 0o755) << 16
        self.fixture([(info, b"../../untrusted")])
        with mock.patch.object(uv.subprocess, "run") as run:
            with self.assertRaisesRegex(uv.BootstrapError, "regular file"):
                uv.install(self.root, self.key)
        run.assert_not_called()

    def test_untrusted_download_urls_are_rejected_before_network(self):
        self.fixture(cached=False)
        urls = [
            "http://files.pythonhosted.org/packages/x/" + self.filename,
            "https://files.pythonhosted.org.attacker.example/packages/x/" + self.filename,
            "https://user@files.pythonhosted.org/packages/x/" + self.filename,
            "https://files.pythonhosted.org:8443/packages/x/" + self.filename,
            "https://files.pythonhosted.org/packages/x/other.whl",
            self.row["url"] + "?unlocked=1", self.row["url"] + "#fragment",
        ]
        with mock.patch.object(uv.urllib.request, "build_opener") as opener:
            for url in urls:
                with self.subTest(url=url):
                    self.row["url"] = url
                    self.write_manifest()
                    with self.assertRaises(uv.BootstrapError):
                        uv.install(self.root, self.key)
        opener.assert_not_called()
        self.assertFalse(self.wheel.exists())

    def test_redirect_cannot_leave_trusted_file_host(self):
        handler = uv.TrustedRedirect(self.filename)
        with self.assertRaises(uv.BootstrapError):
            handler.redirect_request(None, None, 302, "Found", {}, "https://attacker.example/" + self.filename)

    def test_unsupported_platform_fails_before_network_or_install(self):
        self.fixture(cached=False)
        with mock.patch.object(uv.platform, "system", return_value="Windows"), mock.patch.object(uv.platform, "machine", return_value="AMD64"), mock.patch.object(uv.urllib.request, "build_opener") as opener:
            with self.assertRaisesRegex(uv.BootstrapError, "No pinned"):
                uv.install(self.root)
        opener.assert_not_called()
        self.assertFalse((self.root / ".bootstrap").exists())

    def test_platform_selection_distinguishes_libc_and_architecture(self):
        cases = [("Linux", "x86_64", "glibc", "Linux-x86_64-glibc"), ("Linux", "AMD64", "musl", "Linux-x86_64-musl"), ("Linux", "aarch64", "musl", "Linux-aarch64"), ("Darwin", "arm64", "", "Darwin-arm64"), ("Darwin", "x86_64", "", "Darwin-x86_64")]
        for system, machine, libc, expected in cases:
            with self.subTest(expected=expected), mock.patch.object(uv.platform, "system", return_value=system), mock.patch.object(uv.platform, "machine", return_value=machine), mock.patch.object(uv.platform, "libc_ver", return_value=(libc, "")):
                self.assertEqual(uv.platform_key(), expected)

    def test_successful_streamed_download_is_atomically_cached(self):
        self.fixture(cached=False)
        opener = mock.Mock()
        opener.open.return_value = DownloadResponse(self.wheel_bytes, self.row["url"])
        with mock.patch.object(uv.urllib.request, "build_opener", return_value=opener), mock.patch.object(uv.subprocess, "run", return_value=self.version):
            uv.install(self.root, self.key)
        self.assertEqual(self.wheel.read_bytes(), self.wheel_bytes)
        self.assertEqual(list(self.wheel.parent.iterdir()), [self.wheel])
        self.assertEqual(self.target.read_bytes(), self.binary)

    def test_corrupt_truncated_or_oversized_download_never_installed(self):
        self.fixture(cached=False)
        corrupt = bytearray(self.wheel_bytes)
        corrupt[-1] ^= 1
        for data in [bytes(corrupt), self.wheel_bytes[:-1], self.wheel_bytes + b"oversized"]:
            with self.subTest(length=len(data)):
                opener = mock.Mock()
                opener.open.return_value = DownloadResponse(data, self.row["url"])
                with mock.patch.object(uv.urllib.request, "build_opener", return_value=opener), mock.patch.object(uv.subprocess, "run") as run:
                    with self.assertRaises(uv.BootstrapError):
                        uv.install(self.root, self.key)
                run.assert_not_called()
                self.assertFalse(self.wheel.exists())
                self.assertFalse(self.target.exists())
                self.assertEqual(list(self.wheel.parent.iterdir()), [])

    def test_unexpected_version_preserves_existing_binary_and_cleans_temporary(self):
        self.fixture()
        self.target.parent.mkdir(parents=True)
        self.target.write_bytes(b"old binary")
        old = subprocess.CompletedProcess([], 0, stdout="uv 0.8.13\n", stderr="")
        with mock.patch.object(uv.subprocess, "run", return_value=old):
            with self.assertRaisesRegex(uv.BootstrapError, "unexpected version"):
                uv.install(self.root, self.key)
        self.assertEqual(self.target.read_bytes(), b"old binary")
        self.assertEqual(list(self.target.parent.iterdir()), [self.target])

    def test_manifest_rejects_unpinned_version_paths_hash_and_size(self):
        self.fixture()
        for field, value in [("filename", "../uv-0.12.22-py3-none-test.whl"), ("binary_path", "../../bin/uv"), ("sha256", "0" * 63), ("bytes", True)]:
            original = self.row[field]
            with self.subTest(field=field):
                self.row[field] = value
                self.write_manifest()
                with self.assertRaises(uv.BootstrapError):
                    uv.load_row(self.root, self.key)
            self.row[field] = original
        self.manifest["version"] = "0.8.13"
        self.write_manifest()
        with self.assertRaisesRegex(uv.BootstrapError, "version"):
            uv.load_row(self.root, self.key)


if __name__ == "__main__":
    unittest.main()
