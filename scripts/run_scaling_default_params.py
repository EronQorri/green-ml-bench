"""
run_scaling_default_params.py — Scaling run with default (untuned) hyperparameters.

Mirrors run_scaling_all_models.py but skips best_params.json entirely.
Results are saved with model names like RF_default, XGB_default, etc. so
they can be overlaid against the tuned-HP run in the scaling plots.

Usage:
    python run_scaling_default_params.py

Must be run as Administrator for HardwareMonitor CPU power measurement.


NOT USED ANYMORE WAS JUST A TEST TO CHECK IF THE SCALING RUNS WORK WITH DEFAULT HPs. FINAL RUNS ARE WITH BEST PARAMS.
"""

import os
import sys
import time
import numpy as np
import requests
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score
from skorch import NeuralNetClassifier
from skorch.callbacks import EarlyStopping
from xgboost import XGBClassifier
from codecarbon import EmissionsTracker

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / "models"))

from utils import load_data_split, save_results, get_nrows
from power_monitor import CPUPowerMonitor, compute_corrected_co2, print_cpu_summary
from config import RANDOM_STATE

NROWS_VALUES = [1_000, 5_000, 10_000, 50_000, 100_000, 500_000, 1_000_000, 5_000_000, 11_000_000]
RF_MAX_NROWS = 500_000
DATASET = "higgs"
NTFY_CHANNEL = "eron_thesis_higgs_run_123"

XGB_BASE = {"objective": "binary:logistic", "eval_metric": "logloss"}

# XGBoost defaults: n_estimators=100, max_depth=6, learning_rate=0.3
XGB_DEFAULTS = {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.3}

# MLP defaults: two layers of 100 units, Adam lr=0.001, no dropout
MLP_LAYER_SIZES = [100, 100]
MLP_LR = 0.001
MLP_BATCH = 256
MLP_DROPOUT = 0.0


class MLPModule(nn.Module):
    def __init__(self, input_dim, num_classes, layer_sizes=None, dropout_rate=0.0):
        super().__init__()
        if layer_sizes is None:
            layer_sizes = [100, 100]
        layers = []
        cur = input_dim
        for out in layer_sizes:
            layers += [nn.Linear(cur, out), nn.BatchNorm1d(out), nn.ReLU(), nn.Dropout(dropout_rate)]
            cur = out
        layers.append(nn.Linear(cur, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, X):
        return self.network(X)


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


def run_model(model_name, model, X_train, X_test, y_train, y_test, nrows):
    tracker = EmissionsTracker(
        output_dir=str(BASE_DIR / "emissions"),
        project_name=f"{model_name}_{DATASET}",
    )
    cpu_monitor = CPUPowerMonitor()
    cpu_monitor.start()
    tracker.start()

    t0 = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - t0

    emissions_cc = tracker.stop()
    cpu_result = cpu_monitor.stop()
    co2 = compute_corrected_co2(tracker, cpu_result)
    print_cpu_summary(cpu_result, tracker.final_emissions_data.cpu_energy)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    print(f"    acc={acc:.4f}  f1={f1:.4f}  time={training_time:.1f}s")

    save_results(model_name, DATASET, acc, f1, co2, emissions_cc, cpu_result, training_time, nrows, tracker)
    return f1, training_time


# ── Main ──────────────────────────────────────────────────────────────────────

import random
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.cuda.manual_seed_all(RANDOM_STATE)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

overall_start = time.time()
device = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 60)
print("Scaling Default-Params Run — All Models on Higgs")
print(f"nrows: {NROWS_VALUES}")
print(f"MLP device: {device}")
print("=" * 60)

for nrows in NROWS_VALUES:
    print(f"\n{'─'*60}")
    print(f"nrows = {nrows:,}")
    print(f"{'─'*60}")

    os.environ["TEST_NROWS"] = str(nrows)
    X_train_df, X_test_df, y_train, y_test = load_data_split(DATASET)

    # ── Logistic Regression ───────────────────────────────────────────────────
    print("\n>>> LR_default")
    lr = make_pipeline(
        StandardScaler(),
        LogisticRegression(solver="sag", random_state=RANDOM_STATE, max_iter=1000),
    )
    f1, t = run_model("LR_default", lr, X_train_df, X_test_df, y_train, y_test, nrows)
    _notify(f"Default LR [{nrows:,}] done", f"f1={f1:.4f}  {t/60:.1f} min")

    # ── Random Forest (capped at 500k) ────────────────────────────────────────
    if nrows <= RF_MAX_NROWS:
        print("\n>>> RF_default")
        rf = RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1
        )
        f1, t = run_model("RF_default", rf, X_train_df, X_test_df, y_train, y_test, nrows)
        _notify(f"Default RF [{nrows:,}] done", f"f1={f1:.4f}  {t/60:.1f} min")

    # ── XGBoost CPU ───────────────────────────────────────────────────────────
    print("\n>>> XGB_default")
    xgb = XGBClassifier(**XGB_BASE, **XGB_DEFAULTS, random_state=RANDOM_STATE)
    f1, t = run_model("XGB_default", xgb, X_train_df, X_test_df, y_train, y_test, nrows)
    _notify(f"Default XGB [{nrows:,}] done", f"f1={f1:.4f}  {t/60:.1f} min")

    # ── XGBoost GPU ───────────────────────────────────────────────────────────
    print("\n>>> XGB_GPU_default")
    xgb_gpu = XGBClassifier(**XGB_BASE, **XGB_DEFAULTS, random_state=RANDOM_STATE, device="cuda")
    f1, t = run_model("XGB_GPU_default", xgb_gpu, X_train_df, X_test_df, y_train, y_test, nrows)
    _notify(f"Default XGB_GPU [{nrows:,}] done", f"f1={f1:.4f}  {t/60:.1f} min")

    # ── MLP ───────────────────────────────────────────────────────────────────
    print("\n>>> MLP_default")
    torch.manual_seed(RANDOM_STATE)
    X_train_np = X_train_df.to_numpy().astype(np.float32)
    X_test_np = X_test_df.to_numpy().astype(np.float32)
    y_train_np = y_train.to_numpy().astype(np.int64)
    y_test_np = y_test.to_numpy().astype(np.int64)
    input_dim = X_train_np.shape[1]

    net = NeuralNetClassifier(
        module=MLPModule,
        module__input_dim=input_dim,
        module__num_classes=2,
        module__layer_sizes=MLP_LAYER_SIZES,
        module__dropout_rate=MLP_DROPOUT,
        max_epochs=200,
        lr=MLP_LR,
        iterator_train__batch_size=MLP_BATCH,
        iterator_valid__batch_size=MLP_BATCH,
        criterion=nn.CrossEntropyLoss,
        optimizer=torch.optim.Adam,
        iterator_train__shuffle=True,
        device=device,
        verbose=0,
        callbacks=[EarlyStopping(patience=10)],
    )
    mlp = make_pipeline(StandardScaler(), net)
    f1, t = run_model("MLP_default", mlp, X_train_np, X_test_np, y_train_np, y_test_np, nrows)
    _notify(f"Default MLP [{nrows:,}] done", f"f1={f1:.4f}  {t/60:.1f} min")

total_min = (time.time() - overall_start) / 60
print("\n" + "=" * 60)
print(f"DONE — {total_min:.1f} min total")
print("=" * 60)

_notify(
    "Thesis: Default-Params Scaling Run fertig!",
    f"Dauer: {total_min:.1f} min",
    priority="high",
)
