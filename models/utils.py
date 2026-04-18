import pandas as pd
import os
import sys
import csv
import json
from datetime import datetime
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RANDOM_STATE, config, BASE_DIR

PARAMS_FILE = BASE_DIR / "models" / "best_params.json"


def load_data(dataset):
    cfg = config[dataset]
    if dataset == "higgs":
        df = pd.read_parquet(cfg["path"])
        test_nrows = os.environ.get("TEST_NROWS")
        nrows = int(test_nrows) if test_nrows else cfg.get("nrows")
        if nrows:
            df = df.sample(n=nrows, random_state=RANDOM_STATE)
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


def save_best_params(model_key, dataset, data):
    """Upsert tuning results into the shared best_params.json."""
    params = {}
    if PARAMS_FILE.exists():
        with open(PARAMS_FILE) as f:
            params = json.load(f)
    params.setdefault(model_key, {})[dataset] = data
    with open(PARAMS_FILE, "w") as f:
        json.dump(params, f, indent=4)


def load_best_params(model_key, dataset):
    """Load best_params entry for a given model/dataset from shared JSON."""
    if not PARAMS_FILE.exists():
        raise FileNotFoundError(
            f"No tuning results at {PARAMS_FILE}. Run tuning first."
        )
    with open(PARAMS_FILE) as f:
        params = json.load(f)
    if model_key not in params or dataset not in params[model_key]:
        raise KeyError(
            f"No tuning result for {model_key}/{dataset} in {PARAMS_FILE}."
        )
    return params[model_key][dataset]


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
                "co2eq_kg",
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
                "co2eq_kg": emissions,
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
                "co2eq_kg",
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
                "co2eq_kg": emissions,
            }
        )
