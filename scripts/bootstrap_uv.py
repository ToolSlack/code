#!/usr/bin/env python3
"""Install a hash-verified uv wheel's binary using Python's standard library.

No pip, global package installation, wheel extractall, or installer execution is
used. The sole executable is checked against the verified wheel before running.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import stat
import subprocess
import tempfile
import urllib.parse
import urllib.request
import zipfile

VERSION = "0.12.22"
SCHEMA = "toolslack.bootstrap-wheels.v1"
CHUNK = 1024 * 1024
TRUSTED_HOST = "files.pythonhosted.org"


class BootstrapError(RuntimeError):
    pass


def platform_key():
    system, machine = platform.system(), platform.machine().lower()
    machine = {"amd64": "x86_64", "aarch64": "arm64" if system == "Darwin" else "aarch64"}.get(machine, machine)
    if system == "Darwin" and machine in ("arm64", "x86_64"):
        return system + "-" + machine
    if system == "Linux" and machine == "aarch64":
        # This pinned wheel advertises both manylinux and musllinux tags.
        return "Linux-aarch64"
    if system == "Linux" and machine == "x86_64":
        libc = platform.libc_ver()[0].lower()
        if libc in ("glibc", "musl"):
            return "Linux-x86_64-" + libc
        if any(Path("/lib").glob("ld-musl-*.so.1")):
            return "Linux-x86_64-musl"
        try:
            if os.confstr("CS_GNU_LIBC_VERSION"):
                return "Linux-x86_64-glibc"
        except (AttributeError, ValueError, OSError):
            pass
        raise BootstrapError("Cannot identify the Linux C library for a pinned uv wheel.")
    raise BootstrapError("No pinned uv wheel for this operating system and architecture.")


def validate_url(url, filename):
    parsed = urllib.parse.urlparse(url)
    try:
        port = parsed.port
    except ValueError as error:
        raise BootstrapError("Invalid bootstrap download port.") from error
    if (parsed.scheme != "https" or parsed.hostname != TRUSTED_HOST or port not in (None, 443)
            or parsed.username or parsed.password or parsed.query or parsed.fragment
            or not parsed.path.startswith("/packages/")
            or Path(urllib.parse.unquote(parsed.path)).name != filename):
        raise BootstrapError("Bootstrap wheel URL must be the locked HTTPS Python package file-host URL.")
    return url


def load_row(root, key=None):
    try:
        manifest = json.loads((Path(root) / "requirements/bootstrap-wheels.json").read_text())
    except (OSError, ValueError) as error:
        raise BootstrapError("The pinned uv wheel manifest is missing or invalid.") from error
    if not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA or manifest.get("version") != VERSION:
        raise BootstrapError("The uv manifest schema or version differs from the pinned bootstrap.")
    key = key or platform_key()
    platforms = manifest.get("platforms")
    row = platforms.get(key) if isinstance(platforms, dict) else None
    if not isinstance(row, dict):
        raise BootstrapError("The uv manifest has no wheel for the selected platform.")
    filename = row.get("filename")
    if (not isinstance(filename, str) or Path(filename).name != filename
            or "/" in filename or "\\" in filename
            or not filename.startswith("uv-" + VERSION + "-") or not filename.endswith(".whl")):
        raise BootstrapError("Invalid pinned uv wheel filename.")
    if type(row.get("bytes")) is not int or row["bytes"] <= 0:
        raise BootstrapError("Invalid pinned uv wheel byte count.")
    if not isinstance(row.get("sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", row["sha256"]):
        raise BootstrapError("Invalid pinned uv wheel SHA256.")
    if row.get("binary_path") != "uv-" + VERSION + ".data/scripts/uv":
        raise BootstrapError("The uv binary member path differs from the pinned wheel layout.")
    if not isinstance(row.get("url"), str):
        raise BootstrapError("The uv wheel URL is missing.")
    validate_url(row["url"], filename)
    return row


def content_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_wheel(path, row):
    if Path(path).stat().st_size != row["bytes"] or content_sha(path) != row["sha256"]:
        raise BootstrapError("Cached or downloaded uv wheel differs from its pinned byte count/SHA256.")


class TrustedRedirect(urllib.request.HTTPRedirectHandler):
    def __init__(self, filename):
        self.filename = filename

    def redirect_request(self, request, fp, code, message, headers, newurl):
        validate_url(newurl, self.filename)
        return super().redirect_request(request, fp, code, message, headers, newurl)


def download_wheel(row, destination):
    validate_url(row["url"], row["filename"])
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        opener = urllib.request.build_opener(TrustedRedirect(row["filename"]))
        request = urllib.request.Request(row["url"], headers={"User-Agent": "ToolSlack-bootstrap/" + VERSION})
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".uv-wheel-", delete=False) as stream:
            temporary = Path(stream.name)
            digest, total = hashlib.sha256(), 0
            with opener.open(request, timeout=60) as response:
                if response.status != 200:
                    raise BootstrapError("The uv wheel download did not return HTTP 200.")
                validate_url(response.geturl(), row["filename"])
                while True:
                    block = response.read(CHUNK)
                    if not block:
                        break
                    total += len(block)
                    if total > row["bytes"]:
                        raise BootstrapError("The uv wheel download exceeds its pinned byte count.")
                    digest.update(block)
                    stream.write(block)
            if total != row["bytes"] or digest.hexdigest() != row["sha256"]:
                raise BootstrapError("Downloaded uv wheel differs from its pinned byte count/SHA256.")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(destination)
        temporary = None
    except (OSError, ValueError) as error:
        raise BootstrapError("Cannot download the pinned uv wheel: " + str(error)) from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def verify_version(binary):
    try:
        result = subprocess.run([str(binary), "--version"], capture_output=True, text=True,
                                check=True, timeout=15)
    except (OSError, subprocess.SubprocessError) as error:
        raise BootstrapError("The verified uv binary could not report its version.") from error
    if result.stdout.strip().split()[:2] != ["uv", VERSION]:
        raise BootstrapError("The verified uv binary reports an unexpected version.")


def install(root, key=None):
    root = Path(root).resolve()
    row = load_row(root, key)
    wheel = root / ".bootstrap/wheels" / row["filename"]
    if wheel.is_file():
        verify_wheel(wheel, row)
    else:
        download_wheel(row, wheel)
        verify_wheel(wheel, row)
    target = root / ".bootstrap/bin/uv"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with zipfile.ZipFile(wheel) as archive:
            matches = [entry for entry in archive.infolist() if entry.filename == row["binary_path"]]
            if len(matches) != 1:
                raise BootstrapError("Pinned uv wheel must contain exactly one expected executable member.")
            member = matches[0]
            file_type = stat.S_IFMT(member.external_attr >> 16)
            if (member.is_dir() or file_type not in (0, stat.S_IFREG) or member.flag_bits & 1
                    or not 0 < member.file_size <= 256 * 1024 * 1024):
                raise BootstrapError("The pinned uv executable member is not a bounded regular file.")
            expected = hashlib.sha256()
            with archive.open(member) as source:
                for block in iter(lambda: source.read(CHUNK), b""):
                    expected.update(block)
            # Never execute an existing binary before comparing its bytes with
            # the member inside the independently verified locked wheel.
            if (target.is_file() and not target.is_symlink() and os.access(target, os.X_OK)
                    and content_sha(target) == expected.hexdigest()):
                verify_version(target)
                return target
            with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".uv-binary-", delete=False) as stream:
                temporary = Path(stream.name)
                with archive.open(member) as source:
                    for block in iter(lambda: source.read(CHUNK), b""):
                        stream.write(block)
                stream.flush()
                os.fsync(stream.fileno())
            if content_sha(temporary) != expected.hexdigest():
                raise BootstrapError("Extracted uv binary differs from the verified executable member.")
            temporary.chmod(0o755)
            verify_version(temporary)
            temporary.replace(target)
            temporary = None
            return target
    except (OSError, ValueError, zipfile.BadZipFile, RuntimeError) as error:
        if isinstance(error, BootstrapError):
            raise
        raise BootstrapError("Cannot extract the pinned uv executable member: " + str(error)) from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(argv)
    try:
        install(args.root)
    except BootstrapError as error:
        parser.exit(1, "ToolSlack bootstrap error: " + str(error) + "\n")
    print("ToolSlack bootstrap ready: uv " + VERSION)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
