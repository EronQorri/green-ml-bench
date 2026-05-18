import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
from matplotlib.transforms import blended_transform_factory
from pathlib import Path

mpl.rcParams.update({
    "font.family":     "serif",
    "font.serif":      ["Palatino Linotype", "Palatino", "Book Antiqua", "DejaVu Serif"],
    "axes.titlesize":  11,
    "axes.labelsize":  10,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "legend.fontsize":  9,
})

BASE_DIR  = Path(__file__).parent.parent.parent
PLOTS_DIR = Path(__file__).parent

MODEL_PALETTE = {
    "tune_RFC": (212/255, 149/255, 106/255),
    "tune_XGB": (130/255, 185/255, 154/255),
    "tune_MLP": (155/255, 135/255, 181/255),
}
MODEL_LABELS = {
    "tune_RFC": "Random Forest",
    "tune_XGB": "XGBoost",
    "tune_MLP": "MLP",
}
DATASET_LABELS = {"wine": "Wine", "credit": "Credit", "higgs": "HIGGS"}
SHADE_COLORS   = {"wine": "#f8f5f2", "credit": "#f2f6f2", "higgs": "#f2f2f8"}

def fmt_time(s):
    if s >= 3600: return f"{s/3600:.1f} h"
    if s >= 60:   return f"{s/60:.0f} min"
    return f"{s:.0f} s"

def fmt_co2(g):
    if g >= 1000: return f"{g/1000:.2f} kg"
    if g >= 1:    return f"{g:.1f} g"
    return f"{g:.2f} g"

df = pd.read_csv(BASE_DIR / "results" / "results.csv")
df.columns = df.columns.str.strip()
df = df[df["model"].str.startswith("tune_")].copy()
df["co2_g"] = df["co2eq_kg"].astype(float) * 1000
df["time_s"] = df["training_time_s"].astype(float)

# HIGGS → Credit → Wine, within each group descending by CO2 (largest at top)
rows = []
for ds in ["higgs", "credit", "wine"]:
    subset = df[df["dataset"] == ds].sort_values("co2_g", ascending=False)
    for _, r in subset.iterrows():
        rows.append({
            "model":   r["model"],
            "dataset": r["dataset"],
            "co2_g":   r["co2_g"],
            "time_s":  r["time_s"],
        })

n     = len(rows)
y_pos = np.arange(n)

fig, ax = plt.subplots(figsize=(8, 4.0))

# Shaded dataset bands
ds_groups = {}
for i, r in enumerate(rows):
    ds_groups.setdefault(r["dataset"], []).append(i)

for ds, indices in ds_groups.items():
    ylo, yhi = min(indices) - 0.45, max(indices) + 0.45
    ax.axhspan(ylo, yhi, color=SHADE_COLORS[ds], zorder=0, lw=0)

# Horizontal bars
x_left = 0.15
for i, r in enumerate(rows):
    ax.barh(i, r["co2_g"], left=x_left, height=0.55,
            color=MODEL_PALETTE[r["model"]], zorder=2,
            edgecolor="white", linewidth=0.4)

# CO2 value inside bar (near left edge), duration to the right with gap
for i, r in enumerate(rows):
    ax.text(x_left * 1.3, i, fmt_time(r["time_s"]),
            ha="left", va="center", fontsize=7.5, color="white", fontweight="bold")
    ax.text(r["co2_g"] * 1.5, i, fmt_co2(r["co2_g"]),
            ha="left", va="center", fontsize=8.0, color="#333333")

# y-axis: model names
ax.set_yticks(y_pos)
ax.set_yticklabels([MODEL_LABELS[r["model"]] for r in rows], fontsize=9)

# Dataset group labels on the right margin
trans = blended_transform_factory(ax.transAxes, ax.transData)
for ds, indices in ds_groups.items():
    mid = np.mean(indices)
    ax.text(1.015, mid, DATASET_LABELS[ds], transform=trans,
            ha="left", va="center", fontsize=8.5,
            style="italic", color="#666666")

# x-axis
ax.set_xscale("log")
ax.set_xlim(x_left * 0.5, max(r["co2_g"] for r in rows) * 14)
ax.set_xlabel(r"Tuning CO$_2$ (g, log scale)", fontsize=10)
ax.xaxis.grid(False)
ax.yaxis.grid(False)
ax.set_ylim(-0.6, n - 0.4)

# Spines
for spine in ["top", "right", "left"]:
    ax.spines[spine].set_visible(False)

# Legend
patches = [mpatches.Patch(color=MODEL_PALETTE[m], label=MODEL_LABELS[m])
           for m in ["tune_RFC", "tune_XGB", "tune_MLP"]]
ax.legend(handles=patches, loc="upper right", frameon=True,
          fontsize=8.5, framealpha=0.9)

# Footnote
fig.text(0.5, -0.04,
         "Random Forest was not tuned on HIGGS; literature defaults were used instead.",
         ha="center", fontsize=7.5, color="#666666", style="italic")

plt.tight_layout()
plt.savefig(PLOTS_DIR / "tuning_costs.pdf", bbox_inches="tight")
print("Saved: analysis/plots/tuning_costs.pdf")
plt.close()
