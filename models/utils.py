import pandas as pd
import os
import sys
import csv
from datetime import datetime
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config


def load_data(dataset):
    cfg = config[dataset]
    if dataset == "higgs":
        df = pd.read_parquet(cfg["path"])
        if cfg.get("nrows"):
            df = df.head(cfg["nrows"])
    else:
        df = pd.read_csv(cfg["path"], names=cfg["names"], skiprows=cfg["skiprows"], delimiter=cfg["delimiter"])
    if cfg.get("drop_cols"):
        df = df.drop(cfg["drop_cols"], axis=1)
    X = df.drop(cfg["target"], axis=1)
    y = df[cfg["target"]]
    if cfg.get("label_offset"):
        y = y + cfg["label_offset"]
    return X, y


def save_results(model, dataset, accuracy, f1, emissions, training_time, nrows):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    file_path = os.path.join(results_dir, "results.csv")
    file_exists = os.path.isfile(file_path)
    with open(file_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "model", "dataset", "nrows", "accuracy", "f1", "emissions_kg", "training_time_s", "carbon_optimal_score"])
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model": model,
            "dataset": dataset,
            "nrows": nrows if nrows else "all",
            "accuracy": round(accuracy, 4),
            "f1": round(f1, 4),
            "emissions_kg": emissions,
            "training_time_s": round(training_time, 2),
            "carbon_optimal_score": f1 / (emissions * training_time) ** 0.5
        })


def minimal_preprocess(X, y):
    X = X.copy()
    num_cols = X.select_dtypes(include='number').columns
    cat_cols = X.select_dtypes(exclude='number').columns
    if len(num_cols) > 0:
        X[num_cols] = SimpleImputer(strategy='median').fit_transform(X[num_cols])
    if len(cat_cols) > 0:
        X[cat_cols] = SimpleImputer(strategy='most_frequent').fit_transform(X[cat_cols])
        for col in cat_cols:
            X[col] = LabelEncoder().fit_transform(X[col])
    return X, y