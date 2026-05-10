"""
run_scaling_all_models.py — All models on Higgs subsets for dataset-size scaling analysis.

Addresses RQ: "How does reducing the number of training instances on large datasets
affect model accuracy and energy consumption across different algorithms?"

Runs all models on Higgs subset sizes. RandomForest is capped at 500k rows.
Uses the same hyperparameters as the main run (from best_params.json).

Usage:
    python run_scaling_all_models.py

Must be run as Administrator for HardwareMonitor CPU power measurement.
"""

import os
import sys
import subprocess
import time
import requests
from pathlib import Path

BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
NTFY_CHANNEL = "eron_thesis_higgs_run_123"

NROWS_VALUES = [1_000, 10_000, 100_000, 500_000, 1_000_000, 5_000_000, 11_000_000]

MODEL_SCRIPTS = [
    "log_regression.py",
    "random_forest.py",
    "xgboost_cpu.py",
    "xgboost_gpu.py",
    "mlp.py",
]

# RF is too slow beyond 500k rows — skip it for larger subsets
RF_MAX_NROWS = 500_000


def _notify(title, body, priority="default"):
    try:
        requests.post(
            f"https://ntfy.sh/{NTFY_CHANNEL}",
            data=body.encode("utf-8"),
            headers={"Title": title, "Priority": priority, "Tags": "bell"},
            timeout=5,
        )
    except Exception:
        pass


def _run(script_path, nrows):
    env = {
        **os.environ,
        "TEST_NROWS": str(nrows),
    }
    label = f"{script_path.name} [higgs | nrows={nrows:,}]"
    print(f"\n>>> {label}")
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, str(script_path), "higgs"],
        cwd=str(script_path.parent),
        env=env,
    )
    elapsed = time.time() - t0
    status = "OK" if result.returncode == 0 else f"FAILED (code {result.returncode})"
    print(f"    {status} — {elapsed / 60:.1f} min")
    return result.returncode == 0, elapsed


# ── Main ──────────────────────────────────────────────────────────────────────

overall_start = time.time()
failures = []

print("=" * 60)
print("Scaling Subset Run — All Models on Higgs")
print(f"nrows: {NROWS_VALUES}")
print("=" * 60)

for nrows in NROWS_VALUES:
    print(f"\n{'─'*60}")
    print(f"nrows = {nrows:,}")
    print(f"{'─'*60}")
    scripts_this_round = [s for s in MODEL_SCRIPTS if not (s == "random_forest.py" and nrows > RF_MAX_NROWS)]
    for script_name in scripts_this_round:
        script_path = MODELS_DIR / script_name
        ok, elapsed = _run(script_path, nrows)
        if not ok:
            failures.append((script_name, nrows))
        _notify(
            f"Scaling: {script_name.replace('.py','')} [{nrows:,}] {'fertig' if ok else 'FAILED'}",
            f"Dauer: {elapsed/60:.1f} min",
        )

total_min = (time.time() - overall_start) / 60
print("\n" + "=" * 60)
print(f"DONE — {total_min:.1f} min total")
if failures:
    print(f"Failures: {failures}")
else:
    print("No failures.")
print("=" * 60)

_notify(
    "Thesis: Scaling Subset Run fertig!",
    f"Dauer: {total_min:.1f} min. Failures: {failures or 'keine'}",
    priority="high",
)
