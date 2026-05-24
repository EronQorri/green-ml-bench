"""
run_mlp_variation_and_scaling.py — MLP variation followed by full scaling run.

Runs both experiments back-to-back so all data points (including 11M HIGGS)
share the same session timestamp. Shuts down once at the very end.

Phase 1: MLP architecture variation (12 configs on full 11M HIGGS)
Phase 2: All models on HIGGS subsets [1k, 10k, 100k, 500k, 1M, 5M, 11M]

Must be run as Administrator for CPUPowerMonitor sensor access.
"""

import os
import sys
import subprocess
import time
import requests
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"
SCRIPTS_DIR = Path(__file__).parent

NTFY_CHANNEL = "eron_thesis_higgs_run_123"

NROWS_VALUES = [1_000, 5_000, 10_000, 50_000, 100_000, 500_000, 1_000_000, 5_000_000, 11_000_000]

MODEL_SCRIPTS = [
    "log_regression.py",
    "random_forest.py",
    "xgboost_cpu.py",
    "xgboost_gpu.py",
    "mlp.py",
]

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


def _run_model(script_path, nrows):
    env = {**os.environ, "TEST_NROWS": str(nrows)}
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


overall_start = time.time()

# ── Phase 1: MLP variation ────────────────────────────────────────────────────

print("=" * 60)
print("Phase 1: MLP Architecture Variation")
print("=" * 60)

mlp_var_script = SCRIPTS_DIR / "run_mlp_variation.py"
t0 = time.time()
result = subprocess.run([sys.executable, str(mlp_var_script)])
phase1_min = (time.time() - t0) / 60
print(f"\nPhase 1 done in {phase1_min:.1f} min.")

phase1_ok = result.returncode == 0
_notify(
    "Thesis: MLP Variation fertig",
    f"Dauer: {phase1_min:.1f} min. {'OK' if phase1_ok else 'FAILED'}",
)

# ── Phase 2: Scaling run ──────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("Phase 2: Scaling — All Models on Higgs Subsets")
print(f"nrows: {NROWS_VALUES}")
print("=" * 60)

failures = []

for nrows in NROWS_VALUES:
    print(f"\n{'─' * 60}")
    print(f"nrows = {nrows:,}")
    print(f"{'─' * 60}")
    scripts_this_round = [
        s for s in MODEL_SCRIPTS
        if not (s == "random_forest.py" and nrows > RF_MAX_NROWS)
    ]
    for script_name in scripts_this_round:
        ok, elapsed = _run_model(MODELS_DIR / script_name, nrows)
        if not ok:
            failures.append((script_name, nrows))
        _notify(
            f"Scaling: {script_name.replace('.py', '')} [{nrows:,}] {'fertig' if ok else 'FAILED'}",
            f"Dauer: {elapsed / 60:.1f} min",
        )

# ── Summary ───────────────────────────────────────────────────────────────────

total_min = (time.time() - overall_start) / 60
print("\n" + "=" * 60)
print(f"DONE — {total_min:.1f} min total")
print("No failures." if not failures else f"Failures: {failures}")
print("=" * 60)

_notify(
    "Thesis: MLP Variation + Scaling fertig!",
    f"Gesamtdauer: {total_min:.1f} min. Failures: {failures or 'keine'}",
    priority="high",
)