"""
run_inference_power.py — Re-measure inference_time and cpu_power_inference_w
for all nrows=all rows in inference_time.csv.

Fits each model on a 10,000-row stratified subset (fast), then runs 100
single-row predictions with a warmup. Records median latency (inference_time)
and average CPU package power (cpu_power_inference_w). Overwrites existing
values in the nrows=all rows so stale single-shot timings are replaced.

Usage:
    python scripts/run_inference_power.py
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from skorch import NeuralNetClassifier
from skorch.callbacks import EarlyStopping

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / "models"))

from utils import load_data, load_best_params
from power_monitor import CPUPowerMonitor
from config import RANDOM_STATE

INFERENCE_CSV = BASE_DIR / "results" / "inference_time.csv"
FIT_NROWS = 10_000  # rows used for quick-fit; only affects speed, not inference power

xgb_base = {
    "wine":   {"objective": "multi:softmax", "num_class": 3, "eval_metric": "mlogloss"},
    "credit": {"objective": "binary:logistic", "eval_metric": "logloss"},
    "higgs":  {"objective": "binary:logistic", "eval_metric": "logloss"},
}
num_classes = {"wine": 3, "credit": 2, "higgs": 2}


class MLPModule(nn.Module):
    def __init__(self, input_dim, num_classes, layer_sizes=(256, 128), dropout_rate=0.2):
        super().__init__()
        layers = []
        cur = input_dim
        for out in layer_sizes:
            layers += [nn.Linear(cur, out), nn.BatchNorm1d(out), nn.ReLU(), nn.Dropout(dropout_rate)]
            cur = out
        layers.append(nn.Linear(cur, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, X):
        return self.network(X)


def build_model(model_key, dataset, input_dim):
    if model_key == "lr":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(solver="sag", random_state=RANDOM_STATE, max_iter=200),
        )
    if model_key == "rf":
        p = load_best_params("rfc", dataset)["best_params"]
        return RandomForestClassifier(**p, random_state=RANDOM_STATE, n_jobs=-1)
    if model_key in ("xgb_cpu", "xgb_gpu"):
        p = load_best_params("xgb", dataset)["best_params"]
        device = "cuda" if model_key == "xgb_gpu" else None
        kwargs = {**xgb_base[dataset], **p, "random_state": RANDOM_STATE}
        if device:
            kwargs["device"] = device
        return XGBClassifier(**kwargs)
    if model_key == "mlp":
        p = load_best_params("mlp", dataset)["best_params"]
        layer_sizes = [p[f"layer_{i}"] for i in range(p["n_layers"])]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.manual_seed(RANDOM_STATE)
        net = NeuralNetClassifier(
            module=MLPModule,
            module__input_dim=input_dim,
            module__num_classes=num_classes[dataset],
            module__layer_sizes=layer_sizes,
            module__dropout_rate=p["dropout_rate"],
            max_epochs=50,
            lr=p["lr"],
            iterator_train__batch_size=min(p["batch_size"], FIT_NROWS // 2),
            iterator_valid__batch_size=min(p["batch_size"], FIT_NROWS // 2),
            criterion=nn.CrossEntropyLoss,
            optimizer=torch.optim.Adam,
            iterator_train__shuffle=True,
            device=device,
            verbose=0,
            callbacks=[EarlyStopping(patience=5)],
        )
        return make_pipeline(StandardScaler(), net)
    raise ValueError(f"Unknown model_key: {model_key}")


CONFIGS = [
    ("LogisticRegression", "wine",   "lr"),
    ("LogisticRegression", "credit", "lr"),
    ("LogisticRegression", "higgs",  "lr"),
    ("RandomForest",       "wine",   "rf"),
    ("RandomForest",       "credit", "rf"),
    ("XGBoost",            "wine",   "xgb_cpu"),
    ("XGBoost",            "credit", "xgb_cpu"),
    ("XGBoost",            "higgs",  "xgb_cpu"),
    ("XGBoost_GPU",        "wine",   "xgb_gpu"),
    ("XGBoost_GPU",        "credit", "xgb_gpu"),
    ("XGBoost_GPU",        "higgs",  "xgb_gpu"),
    ("MLP_PyTorch",        "wine",   "mlp"),
    ("MLP_PyTorch",        "credit", "mlp"),
    ("MLP_PyTorch",        "higgs",  "mlp"),
]


_INFERENCE_BUDGET = 30.0

def measure_inference(model, single_row):
    """Runs predictions for 30 s, returns (median_latency_s, avg_cpu_watt, energy_per_inference_wh, n)."""
    model.predict(single_row)  # warmup before monitor starts
    monitor = CPUPowerMonitor()
    monitor.start()
    t_start = time.perf_counter()
    times = []
    while time.perf_counter() - t_start < _INFERENCE_BUDGET:
        t0 = time.perf_counter()
        model.predict(single_row)
        times.append(time.perf_counter() - t0)
    result = monitor.stop()
    n = len(times)
    energy_per_inference_wh = (result["energy_wh"] / n) if result.get("energy_wh") is not None else None
    return float(np.median(times)), result.get("avg_watt"), energy_per_inference_wh, n


def load_subset(dataset, nrows):
    os.environ["TEST_NROWS"] = str(nrows)
    X, y = load_data(dataset)
    del os.environ["TEST_NROWS"]
    if len(X) > nrows:
        X, _, y, _ = train_test_split(
            X, y, train_size=nrows, stratify=y, random_state=RANDOM_STATE
        )
    return X, y


def main():
    if not INFERENCE_CSV.exists():
        print(f"inference_time.csv not found at {INFERENCE_CSV}. Run the main benchmark first.")
        sys.exit(1)

    df = pd.read_csv(INFERENCE_CSV)
    df.columns = df.columns.str.strip()
    if "cpu_power_inference_w" not in df.columns:
        df["cpu_power_inference_w"] = pd.NA

    cached_data = {}

    for model_name, dataset, model_key in CONFIGS:
        mask = (
            (df["model"].str.strip() == model_name) &
            (df["dataset"].str.strip() == dataset) &
            (df["nrows"].astype(str).str.strip() == "all")
        )
        if not mask.any():
            print(f"  [skip] {model_name}/{dataset} — no row in inference_time.csv")
            continue

        print(f"  [{model_name}/{dataset}] loading data and fitting...", flush=True)

        if dataset not in cached_data:
            X, y = load_subset(dataset, FIT_NROWS)
            cached_data[dataset] = (X, y)
        X, y = cached_data[dataset]

        is_mlp = model_key == "mlp"
        X_fit = X.to_numpy().astype(np.float32) if is_mlp else X
        y_fit = y.to_numpy().astype(np.int64) if is_mlp else y
        single_row = X_fit[:1]

        model = build_model(model_key, dataset, X_fit.shape[1])
        model.fit(X_fit, y_fit)

        print(f"  [{model_name}/{dataset}] measuring (30 s)...", flush=True)
        median_latency, avg_watt, energy_per_inference_wh, n_reps = measure_inference(model, single_row)

        if "energy_per_inference_wh" not in df.columns:
            df["energy_per_inference_wh"] = pd.NA
        if "n_inference_reps" not in df.columns:
            df["n_inference_reps"] = pd.NA

        df.loc[mask, "inference_time"] = median_latency
        df.loc[mask, "cpu_power_inference_w"] = round(avg_watt, 4) if avg_watt is not None else pd.NA
        df.loc[mask, "energy_per_inference_wh"] = f"{energy_per_inference_wh:.6e}" if energy_per_inference_wh is not None else pd.NA
        df.loc[mask, "n_inference_reps"] = n_reps
        watt_str = f"{avg_watt:.2f} W" if avg_watt is not None else "sensor unavailable"
        print(f"  [{model_name}/{dataset}] {median_latency*1e6:.1f} µs | {watt_str} | n={n_reps:,}")

    df.to_csv(INFERENCE_CSV, index=False)
    print(f"\nUpdated: {INFERENCE_CSV}")


if __name__ == "__main__":
    main()
