"""
run_xgb_breakeven.py — XGBoost CPU vs GPU on Higgs subsets for break-even analysis.

Runs XGBoost (CPU) and XGBoost_GPU on Higgs with increasing nrows.
Same hyperparameters (from Higgs tuning) across all sizes — only dataset size varies.
Results are appended to results/results.csv with the actual nrows value recorded.

Usage:
    python run_xgb_breakeven.py

Must be run as Administrator for HardwareMonitor CPU power measurement.
"""

import os
import sys
import time
import requests
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.append(str(_ROOT))
sys.path.append(str(_ROOT / "models"))

from utils import load_data, save_results, save_inference_time, load_best_params
from power_monitor import CPUPowerMonitor, compute_corrected_co2, print_cpu_summary
from xgboost import XGBClassifier
from sklearn.model_selection import KFold, cross_validate
from sklearn.metrics import f1_score, make_scorer
from codecarbon import EmissionsTracker
from config import BASE_DIR, RANDOM_STATE, CV_FOLDS

NROWS_VALUES = [1_000, 5_000, 10_000, 50_000, 100_000, 500_000, 1_000_000, 5_000_000, 11_000_000]

XGB_BASE = {"objective": "binary:logistic", "eval_metric": "logloss"}
NTFY_CHANNEL = "eron_thesis_higgs_run_123"

cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
_tuned = load_best_params("xgb", "higgs")["best_params"]


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


def run_xgb(nrows, device="cpu"):
    model_name = "XGBoost" if device == "cpu" else "XGBoost_GPU"
    project_name = f"xgb_subset_{device}_{nrows}"

    os.environ["TEST_NROWS"] = str(nrows)
    X, y = load_data("higgs")
    del os.environ["TEST_NROWS"]

    model = XGBClassifier(
        **XGB_BASE, **_tuned,
        random_state=RANDOM_STATE,
        **({"device": "cuda"} if device == "gpu" else {}),
    )

    tracker = EmissionsTracker(
        output_dir=str(BASE_DIR / "emissions"),
        project_name=project_name,
    )
    cpu_monitor = CPUPowerMonitor()

    cpu_monitor.start()
    tracker.start()

    start = time.time()
    cv_results = cross_validate(
        model, X, y, cv=cv,
        scoring={"accuracy": "accuracy", "f1": make_scorer(f1_score, average="weighted")},
        return_estimator=True,
    )
    training_time = time.time() - start
    emissions_cc = tracker.stop()
    cpu_result = cpu_monitor.stop()

    co2_corrected = compute_corrected_co2(tracker, cpu_result)
    print_cpu_summary(cpu_result, tracker.final_emissions_data.cpu_energy)

    trained_model = cv_results["estimator"][0]
    single_row = X[:1]
    start_inf = time.perf_counter()
    _ = trained_model.predict(single_row)
    inference_time = time.perf_counter() - start_inf

    save_results(
        model_name, "higgs",
        cv_results["test_accuracy"].mean(),
        cv_results["test_f1"].mean(),
        co2_corrected, emissions_cc, cpu_result,
        training_time, nrows,
    )
    save_inference_time(model_name, "higgs", co2_corrected, nrows, inference_time)

    return co2_corrected, training_time


# ── Main ──────────────────────────────────────────────────────────────────────

overall_start = time.time()
failures = []

print("=" * 60)
print("Higgs Subset Run — XGBoost CPU vs GPU")
print(f"nrows values: {NROWS_VALUES}")
print("=" * 60)

for nrows in NROWS_VALUES:
    for device in ["cpu", "gpu"]:
        model_name = "XGBoost" if device == "cpu" else "XGBoost_GPU"
        print(f"\n>>> {model_name} | higgs | nrows={nrows:,}")
        try:
            co2, t = run_xgb(nrows, device)
            print(f"    OK — {t/60:.1f} min | CO₂: {co2:.2e} kg")
            _notify(
                f"Subset: {model_name} [{nrows:,}] fertig",
                f"CO₂: {co2:.2e} kg | {t/60:.1f} min",
            )
        except Exception as e:
            print(f"    FAILED: {e}")
            failures.append((model_name, nrows))

total_min = (time.time() - overall_start) / 60
print("\n" + "=" * 60)
print(f"DONE — {total_min:.1f} min total")
if failures:
    print(f"Failures: {failures}")
print("=" * 60)

_notify(
    "Thesis: Higgs Subset Run fertig!",
    f"Dauer: {total_min:.1f} min. Failures: {failures or 'keine'}",
    priority="high",
)
