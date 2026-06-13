import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

mpl.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["Palatino Linotype", "Palatino", "Book Antiqua", "DejaVu Serif"],
    "axes.titlesize":    16,
    "axes.labelsize":    15,
    "xtick.labelsize":   14,
    "ytick.labelsize":   14,
    "legend.fontsize":   14,
})

BASE_DIR = Path(__file__).parent.parent.parent
RESULTS_DIR = BASE_DIR / "results"
PLOTS_DIR = Path(__file__).parent

MODEL_PALETTE = {
    "LogisticRegression": (123/255, 167/255, 188/255),
    "XGBoost":            (130/255, 185/255, 154/255),
    "XGBoost_GPU":        (192/255, 112/255, 112/255),
    "MLP_PyTorch":        (155/255, 135/255, 181/255),
}

MODEL_LABELS = {
    "LogisticRegression": "Logistic Regression",
    "XGBoost":            "XGBoost (CPU)",
    "XGBoost_GPU":        "XGBoost (GPU)",
    "MLP_PyTorch":        "MLP (PyTorch)",
}

MODELS = ["LogisticRegression", "XGBoost", "XGBoost_GPU", "MLP_PyTorch"]

# --- Load data ---

results = pd.read_csv(RESULTS_DIR / "results.csv")
infer   = pd.read_csv(RESULTS_DIR / "inference_time.csv")

def get_train_row(model):
    df = results[
        (results["model"] == model) &
        (results["dataset"] == "higgs") &
        (results["nrows"] == "all")
    ]
    return df.iloc[-1]

def get_infer_row(model):
    df = infer[
        (infer["model"] == model) &
        (infer["dataset"] == "higgs") &
        (infer["nrows"] == "all")
    ]
    return df.iloc[-1]

def carbon_intensity(row):
    total_wh = row["cpu_energy_hw_wh"] + row["gpu_energy_wh"] + row["ram_energy_wh"]
    return row["co2eq_kg"] / total_wh

# Derive a single carbon intensity from all models and average
ci_values = [carbon_intensity(get_train_row(m)) for m in MODELS]
ci = float(np.mean(ci_values))

train_co2 = {}
infer_co2  = {}
for m in MODELS:
    tr = get_train_row(m)
    inf = get_infer_row(m)
    train_co2[m] = tr["co2eq_kg"]
    infer_co2[m] = inf["energy_per_inference_wh"] * ci

print(f"Carbon intensity: {ci*1000:.4f} g CO2/Wh")
for m in MODELS:
    print(f"  {MODEL_LABELS[m]:<25}  train={train_co2[m]*1e3:.4f} g  "
          f"infer={infer_co2[m]*1e9:.2f} ng/pred")

# x-axis: extend past the last crossover (XGBoost CPU vs MLP at ~3.4 M)
N_max = 5_000_000
N = np.linspace(0, N_max, 5000)

fig, ax = plt.subplots(figsize=(9, 5))

for m in MODELS:
    total = (train_co2[m] + N * infer_co2[m]) * 1e3   # grams
    ax.plot(N / 1e6, total, color=MODEL_PALETTE[m], linewidth=1.5, label=MODEL_LABELS[m])

ax.set_xlabel("Number of predictions (millions)")
ax.set_ylabel("Cumulative emissions (gCO₂eq)")
ax.legend()
ax.set_xlim(left=0)
ax.set_ylim(bottom=0)
ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.7)

plt.tight_layout()
out_path = PLOTS_DIR / "lifecycle_xgb_gpu_vs_mlp_higgs.pdf"
plt.savefig(out_path, bbox_inches="tight")
print(f"Saved → {out_path}")
