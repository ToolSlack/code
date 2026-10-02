import ast
import http.client
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest


FILE = Path(__file__).resolve().parents[1] / 'deploy/guard_readiness_patch.py'
SPEC = importlib.util.spec_from_file_location('guard_readiness_patch', FILE)
PATCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PATCH)


class Connection:
    def __init__(self, error=None, status=200):
        self.error, self.status, self.closed = error, status, False

    def request(self, method, path):
        assert (method, path) == ('GET', '/health')

    def getresponse(self):
        if self.error:
            raise self.error
        return SimpleNamespace(status=self.status, read=lambda: b'healthy')

    def close(self):
        self.closed = True


def probe(connection):
    namespace = {'http': SimpleNamespace(client=SimpleNamespace(
        HTTPConnection=lambda *a, **k: connection, HTTPException=http.client.HTTPException))}
    exec(compile(PATCH.NEW, '<readiness-only>', 'exec'), namespace)
    return namespace['healthy']({'health': {'port': 34200, 'path': '/health'}})


class ReadinessTests(unittest.TestCase):
    def test_bad_status_line_is_not_ready_and_connection_closes(self):
        conn = Connection(http.client.BadStatusLine('GET /health HTTP/1.1\r\n'))
        self.assertFalse(probe(conn)); self.assertTrue(conn.closed)

    def test_disconnect_and_os_error_are_not_ready(self):
        for error in (http.client.RemoteDisconnected('starting'), OSError('refused')):
            conn = Connection(error)
            self.assertFalse(probe(conn)); self.assertTrue(conn.closed)

    def test_only_http_200_is_ready(self):
        self.assertTrue(probe(Connection(status=200)))
        self.assertFalse(probe(Connection(status=503)))

    def test_programming_error_remains_fatal(self):
        conn = Connection(ValueError('programming error'))
        with self.assertRaises(ValueError): probe(conn)
        self.assertTrue(conn.closed)

    def test_clone_changes_only_reviewed_exception_tuple(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, target = Path(tmp) / 'original', Path(tmp) / 'clone'
            source.mkdir()
            text = 'import http.client\n\n' + PATCH.OLD + '\ndef restoring_guard():\n    return True\n'
            (source / 'runtime_guard.py').write_text(text)
            for name in ('run_session.py', 'guard_core.py', 'profile.py', 'build_lease.py', 'verify_restored.py'):
                (source / name).write_text('# unchanged public guard dependency\n')
            receipt = PATCH.build(source, target)
            self.assertEqual((source / 'runtime_guard.py').read_text(), text)
            self.assertEqual((target / 'runtime_guard.py').read_text(), text.replace(PATCH.OLD, PATCH.NEW))
            self.assertEqual(receipt['GPU_operations'], 0)
            for name in ('run_session.py', 'guard_core.py', 'profile.py', 'build_lease.py', 'verify_restored.py'):
                self.assertEqual((source / name).read_bytes(), (target / name).read_bytes())
            with self.assertRaises(ValueError): PATCH.build(source, target)


if __name__ == '__main__': unittest.main()
