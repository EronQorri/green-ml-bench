"""
run_all_test.py — smoke test for the full pipeline.

Limits Credit and HIGGS to TEST_NROWS rows and cuts Optuna to N_TRIALS
so the entire tune → train → eval cycle finishes in roughly 5-10 minutes.
Wine is too small to subsample and runs as-is.

Usage:
    python scripts/run_all_test.py
"""

import os
import sys
import subprocess
import time
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
TUNE_DIR = BASE_DIR / "models" / "tune"
MODELS_DIR = BASE_DIR / "models"

TEST_NROWS = 5_000
N_TRIALS = 5

ENV = {
    **os.environ,
    "TEST_NROWS": str(TEST_NROWS),
    "N_TRIALS": str(N_TRIALS),
    "NO_SHUTDOWN": "1",
    "NO_NOTIFY": "1",
}

TUNE_SCRIPTS = [
    ("tune_rfc.py", ["wine", "credit"]),
    ("tune_xgb.py", ["wine", "credit", "higgs"]),
    ("tune_mlp.py", ["wine", "credit", "higgs"]),
]

MODEL_SCRIPTS = [
    "log_regression.py",
    "random_forest.py",
    "xgboost_cpu.py",
    "xgboost_gpu.py",
    "mlp.py",
]

DATASETS = ["wine", "credit", "higgs"]


def run(script_path, dataset):
    label = f"{script_path.name} [{dataset}]"
    print(f"  >>> {label}", flush=True)
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, str(script_path), dataset],
        cwd=str(script_path.parent),
        env=ENV,
    )
    elapsed = time.time() - t0
    status = "OK" if result.returncode == 0 else f"FAILED (code {result.returncode})"
    print(f"      {status} — {elapsed:.0f}s", flush=True)
    return result.returncode == 0


overall_start = time.time()
failures = []

print("=" * 50)
print(f"TEST RUN  |  nrows={TEST_NROWS:,}  |  trials={N_TRIALS}")
print("=" * 50)

print("\n-- Phase 1: Tuning --")
for script_name, datasets in TUNE_SCRIPTS:
    for dataset in datasets:
        if not run(TUNE_DIR / script_name, dataset):
            failures.append(f"tune/{script_name} [{dataset}]")

print("\n-- Phase 2: Training --")
for dataset in DATASETS:
    for script_name in MODEL_SCRIPTS:
        if not run(MODELS_DIR / script_name, dataset):
            failures.append(f"{script_name} [{dataset}]")

total = time.time() - overall_start
print("\n" + "=" * 50)
print(f"Done in {total/60:.1f} min")
if failures:
    print(f"FAILURES ({len(failures)}):")
    for f in failures:
        print(f"  - {f}")
else:
    print("All passed.")
print("=" * 50)
