import pandas as pd
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config


from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import csv
from datetime import datetime

def load_data(dataset):
    cfg = config[dataset]
    if dataset == "higgs":
        df = pd.read_parquet(cfg["path"])
    else:
        df = pd.read_csv(cfg["path"], names=cfg["names"], skiprows=cfg["skiprows"], delimiter=cfg["delimiter"])
    if cfg.get("drop_cols"):
        df = df.drop(cfg["drop_cols"], axis=1)
    X = df.drop(cfg["target"], axis=1)
    y = df[cfg["target"]]
    if cfg.get("label_offset"):
        y = y + cfg["label_offset"]
    return X, y

def save_results(model, dataset, accuracy, f1, emissions, training_time):
    os.makedirs("results", exist_ok=True)
    file_path = "results/results.csv"
    file_exists = os.path.isfile(file_path)
    
    with open(file_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "model", "dataset", "accuracy", "f1", "emissions_kg", "training_time_s"])
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model": model,
            "dataset": dataset,
            "accuracy": round(accuracy, 4),
            "f1": round(f1, 4),
            "emissions_kg": emissions,
            "training_time_s": round(training_time, 2)
        })


def minimal_preprocess(X, y):
    # 1. Missing values (median für numerisch, most_frequent für kategorisch)
    num_cols = X.select_dtypes(include='number').columns
    cat_cols = X.select_dtypes(exclude='number').columns
    
    if len(num_cols) > 0:
        X[num_cols] = SimpleImputer(strategy='median').fit_transform(X[num_cols])
    if len(cat_cols) > 0:
        X[cat_cols] = SimpleImputer(strategy='most_frequent').fit_transform(X[cat_cols])
        for col in cat_cols:
            X[col] = LabelEncoder().fit_transform(X[col])
    
    return X, y