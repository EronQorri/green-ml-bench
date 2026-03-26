import pandas as pd
import numpy as np
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(base_dir, "results.csv"))

df = df.drop_duplicates(subset=["dataset", "nrows", "model"], keep="last")

def normalize_scores(group):
    baseline = group[group["model"] == "LogisticRegression"]
    
    if baseline.empty:
        ref_emissions = group["emissions_kg"].mean()
        ref_time = group["training_time_s"].mean()
    else:
        ref_emissions = baseline.iloc[0]["emissions_kg"]
        ref_time = baseline.iloc[0]["training_time_s"]
    
    group["emissions_norm"] = group["emissions_kg"] / ref_emissions
    group["time_norm"] = group["training_time_s"] / ref_time
    group["score_norm"] = group["f1"] / (group["emissions_norm"] * group["time_norm"]) ** 0.5
    
    return group

df = df.groupby(["dataset", "nrows"], group_keys=False).apply(normalize_scores)

print(df[["model", "dataset", "nrows", "f1", "score_norm"]]
      .sort_values(["dataset", "nrows", "score_norm"], ascending=[True, True, False])
      .to_string(index=False))