import pandas as pd
import os
import sys
import csv
import json
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RANDOM_STATE, config, BASE_DIR, TEST_SIZE

PARAMS_FILE = BASE_DIR / "models" / "best_params.json"


def get_nrows(dataset):
    """Returns actual nrows used — respects TEST_NROWS env var."""
    test_nrows = os.environ.get("TEST_NROWS")
    return int(test_nrows) if test_nrows else config[dataset].get("nrows")


def load_data(dataset):
    cfg = config[dataset]
    if dataset == "higgs":
        df = pd.read_parquet(cfg["path"])
    else:
        df = pd.read_csv(
            cfg["path"],
            names=cfg["names"],
            skiprows=cfg["skiprows"],
            delimiter=cfg["delimiter"],
        )
    if cfg.get("drop_cols"):
        df = df.drop(cfg["drop_cols"], axis=1)
    test_nrows = os.environ.get("TEST_NROWS")
    nrows = int(test_nrows) if test_nrows else cfg.get("nrows")
    if nrows and nrows < len(df):
        df, _ = train_test_split(df, train_size=nrows, stratify=df[cfg["target"]], random_state=RANDOM_STATE)
    X = df.drop(cfg["target"], axis=1)
    y = df[cfg["target"]]
    if cfg.get("label_offset"):
        y = y + cfg["label_offset"]
    return X, y


def load_data_split(dataset):
    """Stratified 80/20 train/test split. Consistent across tune and eval scripts."""
    X, y = load_data(dataset)
    return train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)


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


def save_results(model, dataset, accuracy, f1, co2_corrected, co2_codecarbon, cpu_result, training_time, nrows, tracker=None):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    file_path = os.path.join(results_dir, "results.csv")
    file_exists = os.path.isfile(file_path)
    edata = tracker.final_emissions_data if tracker is not None else None
    gpu_energy_wh = round(edata.gpu_energy * 1000, 6) if edata else None
    ram_energy_wh = round(edata.ram_energy * 1000, 6) if edata else None
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
                "co2eq_codecarbon_kg",
                "cpu_power_hw_w",
                "cpu_energy_hw_wh",
                "gpu_energy_wh",
                "ram_energy_wh",
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
                "accuracy": round(accuracy, 4) if accuracy is not None else "",
                "f1": round(f1, 4),
                "co2eq_kg": co2_corrected,
                "co2eq_codecarbon_kg": co2_codecarbon,
                "cpu_power_hw_w": round(cpu_result["avg_watt"], 4),
                "cpu_energy_hw_wh": round(cpu_result["energy_wh"], 6),
                "gpu_energy_wh": gpu_energy_wh,
                "ram_energy_wh": ram_energy_wh,
                "training_time_s": round(training_time, 2),
            }
        )


def save_inference_time(model, dataset, emissions, nrows, inference_time,
                        cpu_power_inference_w=None, energy_per_inference_wh=None,
                        n_inference_reps=None):
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
                "cpu_power_inference_w",
                "energy_per_inference_wh",
                "n_inference_reps",
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
                "cpu_power_inference_w": round(cpu_power_inference_w, 4) if cpu_power_inference_w is not None else "",
                "energy_per_inference_wh": f"{energy_per_inference_wh:.6e}" if energy_per_inference_wh is not None else "",
                "n_inference_reps": n_inference_reps if n_inference_reps is not None else "",
            }
        )
