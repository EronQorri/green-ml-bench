"""
run_all_test.py — smoke test: 1 trial per tuning script, all datasets.

Verifies the full pipeline (tune → train) works without committing to a long run.
Cancel at any time with Ctrl+C — partial results are still written.

Usage:
    python run_all_test.py
"""

import os
import sys
import subprocess
import time
import json
import csv
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results"
TUNE_DIR = BASE_DIR / "models" / "tune"
MODELS_DIR = BASE_DIR / "models"

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

TUNE_ENV = {
    **os.environ,
    "NO_SHUTDOWN": "1",
    "NO_NOTIFY": "1",
    "N_TRIALS": "1",
    "TEST_NROWS": "30000",  # higgs only; other datasets unaffected
}


def _save_tuning_result(model_label, dataset, best_f1, emissions_kg, duration_s):
    archive = RESULTS_DIR / "archive"
    archive.mkdir(parents=True, exist_ok=True)
    file_path = archive / "test_run_results.csv"
    file_exists = file_path.exists()
    with open(file_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "timestamp", "phase", "model", "dataset",
            "best_f1", "emissions_kg", "duration_s",
        ])
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "phase": "tuning",
            "model": model_label,
            "dataset": dataset,
            "best_f1": round(best_f1, 4) if best_f1 is not None else "",
            "emissions_kg": emissions_kg if emissions_kg is not None else "",
            "duration_s": round(duration_s, 2),
        })


def _run(script_path, dataset, extra_env=None):
    env = {**TUNE_ENV, **(extra_env or {})}
    label = f"{script_path.name} [{dataset}]"
    print(f"\n>>> {label}")
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, str(script_path), dataset],
        cwd=str(script_path.parent),
        env=env,
    )
    elapsed = time.time() - t0
    status = "OK" if result.returncode == 0 else f"FAILED (code {result.returncode})"
    print(f"    {status} — {elapsed:.1f}s")
    return result.returncode == 0, elapsed


overall_start = time.time()

try:
    print("=" * 60)
    print("PHASE 1 — Tuning (1 trial each, smoke test)")
    print("=" * 60)

    for script_name, datasets in TUNE_SCRIPTS:
        script_path = TUNE_DIR / script_name
        model_label = "tune_" + script_name.replace("tune_", "").replace(".py", "").upper()
        for dataset in datasets:
            ok, elapsed = _run(script_path, dataset)
            if not ok:
                print(f"  [WARN] {script_name} [{dataset}] failed — continuing")
                continue
            json_key = script_name.replace("tune_", "").replace(".py", "")
            params_file = MODELS_DIR / "best_params.json"
            best_f1, emissions_kg = None, None
            if params_file.exists():
                with open(params_file) as f:
                    entry = json.load(f).get(json_key, {}).get(dataset, {})
                best_f1 = entry.get("best_f1")
                emissions_kg = entry.get("emissions_kg")
            _save_tuning_result(model_label, dataset, best_f1, emissions_kg, elapsed)

    print("\n" + "=" * 60)
    print("PHASE 2 — Model Training (smoke test)")
    print("=" * 60)

    for dataset in DATASETS:
        for script_name in MODEL_SCRIPTS:
            script_path = MODELS_DIR / script_name
            _run(script_path, dataset)

except KeyboardInterrupt:
    elapsed = (time.time() - overall_start) / 60
    print(f"\n\nCancelled by user after {elapsed:.1f} min.")
    print("Partial results saved to results/archive/test_run_results.csv")
    sys.exit(0)

print("\n" + "=" * 60)
print(f"Smoke test complete — {(time.time() - overall_start)/60:.1f} min total")
print("Results saved to results/archive/test_run_results.csv")
print("=" * 60)
