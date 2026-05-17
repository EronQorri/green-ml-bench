"""
run_inference_power.py — Backfill cpu_power_inference_w in inference_time.csv.

Inference power depends only on model architecture, not training set size.
This script reconstructs each model with its tuned hyperparameters, fits it on
a 10,000-row stratified subset of the relevant dataset (fast, a few minutes
total), and measures average CPU power during 100 single-row predictions via
CPUPowerMonitor. The result is written into inference_time.csv as a new column.

Run this once after the main benchmark — no full retraining required.
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


def measure_inference_power(model, single_row):
    monitor = CPUPowerMonitor()
    monitor.start()
    model.predict(single_row)  # warmup
    for _ in range(100):
        model.predict(single_row)
    result = monitor.stop()
    return result.get("avg_watt")


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

        already = df.loc[mask, "cpu_power_inference_w"]
        if already.notna().all() and (already != "").all():
            print(f"  [skip] {model_name}/{dataset} — already measured")
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

        print(f"  [{model_name}/{dataset}] measuring inference power...", flush=True)
        avg_watt = measure_inference_power(model, single_row)

        df.loc[mask, "cpu_power_inference_w"] = round(avg_watt, 4) if avg_watt is not None else pd.NA
        print(f"  [{model_name}/{dataset}] {avg_watt:.2f} W" if avg_watt else f"  [{model_name}/{dataset}] sensor unavailable")

    df.to_csv(INFERENCE_CSV, index=False)
    print(f"\nUpdated: {INFERENCE_CSV}")


if __name__ == "__main__":
    main()
