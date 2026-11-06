"""Run isolated suites whose local and Kaggle packages share module names."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SUITES = (
    "baseline_candidate/tests",
    "baseline_candidate/online_kaggle/tests",
    "baseline_candidate/online_kaggle/submission/tests",
    "finetuned_ensemble/tests",
)


def main():
    failed = []
    for suite in SUITES:
        print(f"\nTesting {suite}", flush=True)
        result = subprocess.run(
            [sys.executable, "-m", "pytest", suite, "-q", *sys.argv[1:]], cwd=ROOT,
        )
        if result.returncode:
            failed.append(suite)
    if failed:
        print("Failed suites: " + ", ".join(failed), file=sys.stderr)
    return bool(failed)


if __name__ == "__main__":
    raise SystemExit(main())
