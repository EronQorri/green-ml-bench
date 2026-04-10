import pandas as pd
import os
import sys
import csv
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RANDOM_STATE, config


def load_data(dataset):
    cfg = config[dataset]
    if dataset == "higgs":
        df = pd.read_parquet(cfg["path"])
        if cfg.get("nrows"):
            df = df.sample(n=cfg["nrows"], random_state=RANDOM_STATE)
    else:
        df = pd.read_csv(
            cfg["path"],
            names=cfg["names"],
            skiprows=cfg["skiprows"],
            delimiter=cfg["delimiter"],
        )
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
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp",
                "model",
                "dataset",
                "nrows",
                "accuracy",
                "f1",
                "emissions_kg",
                "training_time_s",
            ],
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow(
            {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "model": model,
                "dataset": dataset,
                "nrows": nrows if nrows else "all",
                "accuracy": round(accuracy, 4),
                "f1": round(f1, 4),
                "emissions_kg": emissions,
                "training_time_s": round(training_time, 2),
            }
        )


def save_inference_time(model, dataset, emissions, nrows, inference_time):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    file_path = os.path.join(results_dir, "inference_time.csv")
    file_exists = os.path.isfile(file_path)
    with open(file_path, "a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp",
                "model",
                "dataset",
                "nrows",
                "inference_time",
                "emissions_kg",
            ],
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow(
            {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "model": model,
                "dataset": dataset,
                "nrows": nrows if nrows else "all",
                "inference_time": inference_time,
                "emissions_kg": emissions,
            }
        )
