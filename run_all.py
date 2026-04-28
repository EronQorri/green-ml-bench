"""
run_all.py — full pipeline: hyperparameter tuning → model training.

Runs every tuning script and every model script across all datasets.
Tuning results are written to results/results.csv alongside model results.
codecarbon tracks emissions for every step.

Usage:
    python run_all.py
"""

import os
import sys
import subprocess
import threading
import time
import json
import csv
from datetime import datetime
from pathlib import Path
import requests
sys.path.insert(0, str(Path(__file__).parent / "models"))
from power_monitor import CPUPowerMonitor, CARBON_KG_PER_KWH

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results"
TUNE_DIR = BASE_DIR / "models" / "tune"
MODELS_DIR = BASE_DIR / "models"
NTFY_CHANNEL = "eron_thesis_higgs_run_123"

# RFC skips higgs (too slow with CV on 11M rows)
TUNE_SCRIPTS = [
    ("tune_rfc.py", ["wine", "credit"]),
    ("tune_xgb.py", ["wine", "credit", "higgs"]),
    ("tune_mlp.py", ["wine", "credit", "higgs"]),
]

MODEL_SCRIPTS = [
    "log_regression.py",
    "random_forest.py",   # skips higgs internally
    "xgboost_cpu.py",
    "xgboost_gpu.py",
    "mlp.py",
]

DATASETS = ["wine", "credit", "higgs"]

# Env vars passed to tune subprocesses
TUNE_ENV = {
    **os.environ,
    "NO_SHUTDOWN": "1",  # prevent tune_mlp from shutting down mid-run
    "NO_NOTIFY": "1",    # suppress per-script ntfy; master sends one at the end
}

# --- helpers ---

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


def _save_tuning_result(model_label, dataset, best_f1, emissions_kg, duration_s, cpu_result=None):
    """Append a tuning row to results/results.csv in the same format as model rows."""
    RESULTS_DIR.mkdir(exist_ok=True)
    file_path = RESULTS_DIR / "results.csv"
    file_exists = file_path.exists()

    cpu_power = round(cpu_result["avg_watt"], 4) if cpu_result and cpu_result.get("avg_watt") else ""
    cpu_energy = round(cpu_result["energy_wh"], 6) if cpu_result and cpu_result.get("energy_wh") else ""
    # Tuning is CPU-only: corrected CO2 = HW CPU energy; CodeCarbon value kept for comparison
    if cpu_result and cpu_result.get("energy_wh"):
        co2eq_kg = round(cpu_result["energy_wh"] / 1000 * CARBON_KG_PER_KWH, 9)
    else:
        co2eq_kg = emissions_kg if emissions_kg is not None else ""

    with open(file_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "timestamp", "model", "dataset", "nrows",
            "accuracy", "f1", "co2eq_kg", "co2eq_codecarbon_kg",
            "cpu_power_hw_w", "cpu_energy_hw_wh", "training_time_s",
        ])
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model": model_label,
            "dataset": dataset,
            "nrows": "all",
            "accuracy": "",
            "f1": round(best_f1, 4) if best_f1 is not None else "",
            "co2eq_kg": co2eq_kg,
            "co2eq_codecarbon_kg": emissions_kg if emissions_kg is not None else "",
            "cpu_power_hw_w": cpu_power,
            "cpu_energy_hw_wh": cpu_energy,
            "training_time_s": round(duration_s, 2),
        })


def _tee(src, *dsts):
    for line in iter(src.readline, b""):
        for dst in dsts:
            dst.write(line)
            dst.flush()
    src.close()


def _run(script_path, dataset, extra_env=None):
    """Run script as subprocess; stream output to terminal + log file."""
    env = {**TUNE_ENV, **(extra_env or {})}
    label = f"{script_path.name} [{dataset}]"
    log_path = BASE_DIR / "logs" / f"{script_path.stem}_{dataset}.log"
    log_path.parent.mkdir(exist_ok=True)
    print(f"\n>>> {label}  (log: {log_path.relative_to(BASE_DIR)})")
    t0 = time.time()
    with open(log_path, "wb") as log_f:
        proc = subprocess.Popen(
            [sys.executable, str(script_path), dataset],
            cwd=str(script_path.parent),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        t = threading.Thread(target=_tee, args=(proc.stdout, sys.stdout.buffer, log_f))
        t.start()
        proc.wait()
        t.join()
    elapsed = time.time() - t0
    status = "OK" if proc.returncode == 0 else f"FAILED (code {proc.returncode})"
    print(f"    {status} — {elapsed / 60:.1f} min")
    return proc.returncode == 0, elapsed


# ── Phase 1: Hyperparameter tuning ───────────────────────────────────────────

print("=" * 60)
print("PHASE 1 — Hyperparameter Tuning")
print("=" * 60)

overall_start = time.time()
tune_failures = []

for script_name, datasets in TUNE_SCRIPTS:
    script_path = TUNE_DIR / script_name
    # Model label for results.csv  e.g. "tune_RFC", "tune_XGB", "tune_MLP"
    model_label = "tune_" + script_name.replace("tune_", "").replace(".py", "").upper()

    for dataset in datasets:
        cpu_monitor = CPUPowerMonitor()
        cpu_monitor.start()
        ok, elapsed = _run(script_path, dataset)
        cpu_result = cpu_monitor.stop()

        if not ok:
            tune_failures.append((script_name, dataset))
            continue

        json_key = script_name.replace("tune_", "").replace(".py", "")
        params_file = MODELS_DIR / "best_params.json"
        best_f1, emissions_kg = None, None
        if params_file.exists():
            with open(params_file) as f:
                entry = json.load(f).get(json_key, {}).get(dataset, {})
            best_f1 = entry.get("best_f1")
            emissions_kg = entry.get("emissions_kg")

        _save_tuning_result(model_label, dataset, best_f1, emissions_kg, elapsed, cpu_result)
        _notify(
            f"Thesis: {model_label} [{dataset}] fertig",
            f"Best F1: {f'{best_f1:.4f}' if best_f1 is not None else 'N/A'} | Dauer: {elapsed/60:.1f} min | Emissions: {f'{emissions_kg:.2e}' if emissions_kg is not None else 'N/A'} kg",
            priority="default",
        )

print("\nPhase 1 complete.")
if tune_failures:
    print(f"  Failures: {tune_failures}")

_notify(
    "Thesis: Tuning abgeschlossen",
    f"Alle Tuning-Skripte fertig. Failures: {tune_failures or 'keine'}",
    priority="high",
)

# ── Phase 2: Model training ───────────────────────────────────────────────────

print("\n" + "=" * 60)
print("PHASE 2 — Model Training")
print("=" * 60)

model_failures = []

for dataset in DATASETS:
    for script_name in MODEL_SCRIPTS:
        script_path = MODELS_DIR / script_name
        ok, elapsed = _run(script_path, dataset)
        if not ok:
            model_failures.append((script_name, dataset))
        _notify(
            f"Thesis: {script_name.replace('.py','')} [{dataset}] {'fertig' if ok else 'FAILED'}",
            f"Dauer: {elapsed/60:.1f} min",
            priority="default",
        )

print("\nPhase 2 complete.")
if model_failures:
    print(f"  Failures: {model_failures}")

# ── Summary ───────────────────────────────────────────────────────────────────

total_min = (time.time() - overall_start) / 60
all_failures = tune_failures + model_failures

print("\n" + "=" * 60)
print(f"ALL DONE — {total_min:.1f} min total")
if all_failures:
    print(f"Failures: {all_failures}")
else:
    print("No failures.")
print("=" * 60)

_notify(
    "Thesis: Master-Run beendet!",
    f"Alles fertig! Dauer: {total_min:.1f} min. Failures: {all_failures or 'keine'}",
    priority="urgent",
)
