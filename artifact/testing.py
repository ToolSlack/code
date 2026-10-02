"""Run unittest discovery and preserve machine-readable real test outcomes."""
import argparse
import json
from pathlib import Path
import sys
import unittest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", required=True)
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()
    directory = Path(args.directory).resolve()
    sys.path.insert(0, str(directory))
    suite = unittest.defaultTestLoader.discover(str(directory), pattern="test*.py")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    Path(args.summary).write_text(json.dumps(dict(
        tests_run=result.testsRun, failures=len(result.failures), errors=len(result.errors),
        skipped=len(result.skipped), expected_failures=len(result.expectedFailures),
        unexpected_successes=len(result.unexpectedSuccesses), successful=result.wasSuccessful(),
        failed_test_ids=[test.id() for test, _ in result.failures + result.errors],
        skipped_test_ids=[test.id() for test, _ in result.skipped]), indent=2) + "\n")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
