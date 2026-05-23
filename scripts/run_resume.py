"""
run_resume.py — resume after tune_MLP [higgs] crash.

Skips all completed tuning steps (RFC wine/credit, XGB wine/credit/higgs,
MLP wine/credit). Runs tune_MLP higgs, then full Phase 2 model training.

Usage:
    python scripts/run_resume.py
"""

import os
import sys
import subprocess
import threading
import time
import json
from pathlib import Path
import requests

BASE_DIR = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / "results"
TUNE_DIR = BASE_DIR / "models" / "tune"
MODELS_DIR = BASE_DIR / "models"
NTFY_CHANNEL = "eron_thesis_higgs_run_123"

TUNE_SCRIPTS = [
    ("tune_mlp.py", ["higgs"]),
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
}


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


def _tee(src, *dsts):
    for line in iter(src.readline, b""):
        for dst in dsts:
            dst.write(line)
            dst.flush()
    src.close()


def _run(script_path, dataset, extra_env=None):
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


# ── Phase 1: tune_MLP higgs only ─────────────────────────────────────────────

print("=" * 60)
print("PHASE 1 — tune_MLP [higgs] (resume)")
print("=" * 60)

overall_start = time.time()
tune_failures = []

for script_name, datasets in TUNE_SCRIPTS:
    script_path = TUNE_DIR / script_name
    model_label = "tune_" + script_name.replace("tune_", "").replace(".py", "").upper()

    for dataset in datasets:
        ok, elapsed = _run(script_path, dataset)

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
    f"tune_MLP higgs fertig. Failures: {tune_failures or 'keine'}",
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
