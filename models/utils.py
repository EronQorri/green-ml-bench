import pandas as pd
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config

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