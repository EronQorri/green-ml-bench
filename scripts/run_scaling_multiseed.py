"""
run_scaling_multiseed.py — Scaling run with fixed test set and multiple seeds.

Methodology:
  1. HIGGS is split ONCE (FIXED_SPLIT_SEED) into a train pool (80%) and a
     fixed test set (20%, ~2.2M rows). The test set never changes across runs.
  2. For each nrows in NROWS, a stratified subsample is drawn from the train
     pool using the current seed. None = use the entire train pool (~8.8M rows).
  3. Three seeds control which subsample is drawn, averaging out lucky splits.

The CSV column `nrows` always reflects the actual number of training instances
used (e.g. ~8.8M for the full-train-pool case, not 11M).

Results saved to results/results_scaling_multiseed.csv with a `seed` column.

Usage:
    python scripts/run_scaling_multiseed.py

Must be run as Administrator for HardwareMonitor CPU power measurement.
"""

import sys
import csv
import time
import random
import requests
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score
from skorch import NeuralNetClassifier
from skorch.callbacks import EarlyStopping
from xgboost import XGBClassifier
from codecarbon import EmissionsTracker
import pandas as pd

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / "models"))

from power_monitor import CPUPowerMonitor, compute_corrected_co2, print_cpu_summary
from utils import load_best_params

# Seed used for the one-time train/test split — never changes
FIXED_SPLIT_SEED = 42

# Seeds that control which subsample is drawn from the train pool
SEEDS = [42, 123, 999]

# Training sizes to evaluate. None = entire train pool (~8.8M rows for HIGGS).
NROWS = [1_000, 10_000, 50_000, 100_000, 200_000, 500_000, 1_000_000, 5_000_000, None]

DATASET  = "higgs"
PARQUET  = BASE_DIR / "csv_files" / "higgs" / "higgs.parquet"
TARGET   = "label"
TEST_SIZE = 0.2
NTFY     = "eron_thesis_higgs_run_123"
OUT_CSV  = BASE_DIR / "results" / "results_scaling_multiseed.csv"

EPOCHS = 200
XGB_BASE = {"objective": "binary:logistic", "eval_metric": "logloss"}
RFC_MAX_ROWS = 500_000


class MLPModule(nn.Module):
    def __init__(self, input_dim, num_classes, layer_sizes=None, dropout_rate=0.0):
        super().__init__()
        if layer_sizes is None:
            layer_sizes = [256, 128]
        layers, cur = [], input_dim
        for out in layer_sizes:
            layers += [nn.Linear(cur, out), nn.BatchNorm1d(out), nn.ReLU(), nn.Dropout(dropout_rate)]
            cur = out
        layers.append(nn.Linear(cur, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, X):
        return self.network(X)


def _notify(title, body, priority="default"):
    try:
        requests.post(f"https://ntfy.sh/{NTFY}", data=body.encode(),
                      headers={"Title": title, "Priority": priority, "Tags": "bell"}, timeout=5)
    except Exception:
        pass


def _save(model_name, seed, nrows, acc, f1, co2, co2_cc, cpu_result, training_time, tracker):
    OUT_CSV.parent.mkdir(exist_ok=True)
    exists = OUT_CSV.exists()
    edata = tracker.final_emissions_data if tracker else None
    with open(OUT_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "timestamp", "model", "dataset", "nrows", "seed",
            "accuracy", "f1", "co2eq_kg", "co2eq_codecarbon_kg",
            "cpu_power_hw_w", "cpu_energy_hw_wh", "gpu_energy_wh",
            "ram_energy_wh", "training_time_s",
        ])
        if not exists:
            w.writeheader()
        w.writerow({
            "timestamp":           datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model":               model_name,
            "dataset":             DATASET,
            "nrows":               nrows,
            "seed":                seed,
            "accuracy":            round(acc, 4),
            "f1":                  round(f1, 4),
            "co2eq_kg":            co2,
            "co2eq_codecarbon_kg": co2_cc,
            "cpu_power_hw_w":      round(cpu_result["avg_watt"], 4),
            "cpu_energy_hw_wh":    round(cpu_result["energy_wh"], 6),
            "gpu_energy_wh":       round(edata.gpu_energy * 1000, 6) if edata else "",
            "ram_energy_wh":       round(edata.ram_energy * 1000, 6) if edata else "",
            "training_time_s":     round(training_time, 2),
        })


def _run(model_name, model, X_train, X_test, y_train, y_test, seed, actual_nrows):
    tracker = EmissionsTracker(
        output_dir=str(BASE_DIR / "emissions"),
        project_name=f"{model_name}_{DATASET}_s{seed}_{actual_nrows}",
    )
    cpu_monitor = CPUPowerMonitor()
    cpu_monitor.start()
    tracker.start()

    t0 = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - t0

    co2_cc = tracker.stop()
    cpu_result = cpu_monitor.stop()
    co2 = compute_corrected_co2(tracker, cpu_result)
    print_cpu_summary(cpu_result, tracker.final_emissions_data.cpu_energy)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average="weighted")
    print(f"    acc={acc:.4f}  f1={f1:.4f}  time={training_time:.1f}s")

    _save(model_name, seed, actual_nrows, acc, f1, co2, co2_cc, cpu_result, training_time, tracker)
    return f1, training_time


# ── Setup ─────────────────────────────────────────────────────────────────────

print("Loading HIGGS parquet...")
df_full = pd.read_parquet(PARQUET)

# One-time fixed split: train pool (80%) + fixed test set (20%)
print(f"Creating fixed train/test split (seed={FIXED_SPLIT_SEED}, test_size={TEST_SIZE})...")
df_train_pool, df_test = train_test_split(
    df_full, test_size=TEST_SIZE, stratify=df_full[TARGET], random_state=FIXED_SPLIT_SEED
)
X_test  = df_test.drop(TARGET, axis=1)
y_test  = df_test[TARGET]
X_te_np = X_test.to_numpy().astype(np.float32)
y_te_np = y_test.to_numpy().astype(np.int64)

TRAIN_POOL_SIZE = len(df_train_pool)
print(f"  Train pool: {TRAIN_POOL_SIZE:,} rows  |  Fixed test set: {len(df_test):,} rows")

overall_start = time.time()
device = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 60)
print(f"Multi-seed scaling run — seeds={SEEDS}  nrows={NROWS}")
print(f"MLP device: {device}")
print("=" * 60)

# ── Main loop ─────────────────────────────────────────────────────────────────

for seed in SEEDS:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"\n{'='*60}\nSEED {seed}\n{'='*60}")

    rf_p  = load_best_params("rfc", DATASET)["best_params"]
    xgb_p = load_best_params("xgb", DATASET)["best_params"]
    mlp_p = load_best_params("mlp", DATASET)["best_params"]
    mlp_layers = [mlp_p[f"layer_{i}"] for i in range(mlp_p["n_layers"])]

    for nrows in NROWS:
        # None means: use entire train pool
        use_full_pool = nrows is None or nrows >= TRAIN_POOL_SIZE
        actual_nrows  = TRAIN_POOL_SIZE if use_full_pool else nrows

        print(f"\n{'-'*60}\nnrows={actual_nrows:,}{'  (full train pool)' if use_full_pool else ''}  seed={seed}\n{'-'*60}")

        # Draw subsample from train pool (subsample seed varies; test set is fixed)
        if use_full_pool:
            X_train = df_train_pool.drop(TARGET, axis=1)
            y_train = df_train_pool[TARGET]
        else:
            df_sub, _ = train_test_split(
                df_train_pool, train_size=nrows,
                stratify=df_train_pool[TARGET], random_state=seed,
            )
            X_train = df_sub.drop(TARGET, axis=1)
            y_train = df_sub[TARGET]

        # ── Logistic Regression ───────────────────────────────────────────────
        print("\n>>> LogisticRegression")
        lr = make_pipeline(StandardScaler(), LogisticRegression(solver="sag", random_state=seed, max_iter=1000))
        f1, t = _run("LogisticRegression", lr, X_train, X_test, y_train, y_test, seed, actual_nrows)
        _notify(f"Multiseed LR s{seed} [{actual_nrows:,}]", f"f1={f1:.4f} {t:.0f}s")

        # ── Random Forest (capped at RFC_MAX_ROWS) ────────────────────────────
        if actual_nrows <= RFC_MAX_ROWS:
            print("\n>>> RandomForest")
            rf = RandomForestClassifier(**rf_p, random_state=seed, n_jobs=-1)
            f1, t = _run("RandomForest", rf, X_train, X_test, y_train, y_test, seed, actual_nrows)
            _notify(f"Multiseed RF s{seed} [{actual_nrows:,}]", f"f1={f1:.4f} {t:.0f}s")
        else:
            print(f"\n>>> RandomForest  SKIPPED (nrows={actual_nrows:,} > RFC_MAX_ROWS={RFC_MAX_ROWS:,})")

        # ── XGBoost CPU ───────────────────────────────────────────────────────
        print("\n>>> XGBoost_CPU")
        xgb_cpu = XGBClassifier(**XGB_BASE, **xgb_p, random_state=seed)
        f1, t = _run("XGBoost_CPU", xgb_cpu, X_train, X_test, y_train, y_test, seed, actual_nrows)
        _notify(f"Multiseed XGB_CPU s{seed} [{actual_nrows:,}]", f"f1={f1:.4f} {t:.0f}s")

        # ── XGBoost GPU ───────────────────────────────────────────────────────
        print("\n>>> XGBoost_GPU")
        xgb_gpu = XGBClassifier(**XGB_BASE, **xgb_p, random_state=seed, device="cuda")
        f1, t = _run("XGBoost_GPU", xgb_gpu, X_train, X_test, y_train, y_test, seed, actual_nrows)
        _notify(f"Multiseed XGB_GPU s{seed} [{actual_nrows:,}]", f"f1={f1:.4f} {t:.0f}s")

        # ── MLP ───────────────────────────────────────────────────────────────
        print("\n>>> MLP_PyTorch")
        X_tr_np = X_train.to_numpy().astype(np.float32)
        y_tr_np = y_train.to_numpy().astype(np.int64)

        net = NeuralNetClassifier(
            module=MLPModule,
            module__input_dim=X_tr_np.shape[1],
            module__num_classes=2,
            module__layer_sizes=mlp_layers,
            module__dropout_rate=mlp_p["dropout_rate"],
            max_epochs=EPOCHS,
            lr=mlp_p["lr"],
            iterator_train__batch_size=mlp_p["batch_size"],
            iterator_valid__batch_size=mlp_p["batch_size"],
            criterion=nn.CrossEntropyLoss,
            optimizer=torch.optim.Adam,
            iterator_train__shuffle=True,
            device=device,
            verbose=0,
            callbacks=[EarlyStopping(patience=10)],
        )
        mlp = make_pipeline(StandardScaler(), net)
        f1, t = _run("MLP_PyTorch", mlp, X_tr_np, X_te_np, y_tr_np, y_te_np, seed, actual_nrows)
        _notify(f"Multiseed MLP s{seed} [{actual_nrows:,}]", f"f1={f1:.4f} {t/60:.1f}min")


total_min = (time.time() - overall_start) / 60
print(f"\n{'='*60}\nDONE — {total_min:.1f} min total\n{'='*60}")
_notify("Thesis: Multi-seed scaling run fertig!", f"{total_min:.1f} min", priority="high")

import subprocess

print("\nCommitting and pushing results...")
subprocess.run(["git", "add", "."], cwd=BASE_DIR, check=True)
subprocess.run(["git", "commit", "-m", "scaling rework"], cwd=BASE_DIR, check=True)
subprocess.run(["git", "push", "all"], cwd=BASE_DIR, check=True)
print("Git done. Shutting down...")

import os
os.system("shutdown /s /t 60")
