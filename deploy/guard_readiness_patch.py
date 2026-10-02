#!/usr/bin/env python3
"""Clone the restoring guard with one transient HTTP-readiness correction.

Only the startup probe's exception tuple changes. A malformed HTTP response
means not ready, never ready. All lease validation, timeout, process tracking,
GPU restoration, and telemetry source files are copied byte for byte.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil


OLD = """def healthy(profile):
    conn = http.client.HTTPConnection('127.0.0.1', profile['health']['port'], timeout=3)
    try:
        conn.request('GET', profile['health']['path']); response = conn.getresponse(); response.read()
        return response.status == 200
    except OSError: return False
    finally: conn.close()
"""
NEW = OLD.replace('except OSError:', 'except (OSError, http.client.HTTPException):')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(source: Path, target: Path):
    source, target = source.resolve(), target.resolve()
    if target.exists() or target == source or source in target.parents:
        raise ValueError('The guard clone must be a new separate directory')
    required = ('run_session.py', 'runtime_guard.py', 'guard_core.py', 'profile.py',
                'build_lease.py', 'verify_restored.py')
    for name in required:
        file = source / name
        if file.is_symlink() or not file.is_file():
            raise ValueError('Missing regular restoring guard source: ' + name)
    original = (source / 'runtime_guard.py').read_text()
    if original.count(OLD) != 1 or 'import http.client' not in original:
        raise ValueError('Readiness patch does not match the reviewed native guard')
    hashes = {name: sha(source / name) for name in required}
    shutil.copytree(source, target, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    patched = original.replace(OLD, NEW)
    compile(patched, str(target / 'runtime_guard.py'), 'exec')
    (target / 'runtime_guard.py').write_text(patched)
    if any(sha(target / name) != hashes[name] for name in required if name != 'runtime_guard.py'):
        raise ValueError('A non-readiness guard dependency changed during cloning')
    if any(sha(source / name) != hashes[name] for name in required):
        raise ValueError('The historical guard source changed during cloning')
    receipt = dict(schema='toolslack.guard-http-readiness-patch.v1',
                   source=str(source), target=str(target), GPU_operations=0,
                   change='HTTPException is transient not-ready within original startup timeout',
                   original_source_sha256=hashes,
                   target_source_sha256={name: sha(target / name) for name in required},
                   unchanged_restoration_validation=True)
    with (target / 'READINESS_PATCH.json').open('x') as stream:
        json.dump(receipt, stream, indent=2); stream.write('\n')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--target', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.target), indent=2))
