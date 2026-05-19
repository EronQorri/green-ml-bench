import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from pathlib import Path

mpl.rcParams.update({
    "font.family":    "serif",
    "font.serif":     ["Palatino Linotype", "Palatino", "Book Antiqua", "DejaVu Serif"],
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})

BASE_DIR  = Path(__file__).parent.parent.parent
PLOTS_DIR = Path(__file__).parent

lam = 1.0

df = pd.read_csv(BASE_DIR / "results" / "results.csv")
df.columns = df.columns.str.strip()
df["model"]   = df["model"].astype(str).str.strip()
df["dataset"] = df["dataset"].astype(str).str.strip()

mlp_var = df[(df["dataset"] == "higgs") & df["model"].str.match(r"^MLP_\d+x\d+$", na=False)].copy()

parsed = mlp_var["model"].str.extract(r"^MLP_(\d+)x(\d+)$")
mlp_var["depth"]    = pd.to_numeric(parsed[0])
mlp_var["width"]    = pd.to_numeric(parsed[1])
mlp_var["co2eq_kg"] = pd.to_numeric(mlp_var["co2eq_kg"], errors="coerce")
mlp_var["f1"]       = pd.to_numeric(mlp_var["f1"],       errors="coerce")
mlp_var = mlp_var.dropna(subset=["depth", "width", "f1", "co2eq_kg"])

mn = mlp_var["co2eq_kg"].min()
mx = mlp_var["co2eq_kg"].max()
mlp_var["co2eq_kg_scaled"] = (mlp_var["co2eq_kg"] - mn) / max(mx - mn, 1e-9)
mlp_var["ewf1"] = mlp_var["f1"] / (1 + lam * mlp_var["co2eq_kg_scaled"])

piv = mlp_var.pivot_table(index="depth", columns="width", values="ewf1", aggfunc="mean")

# find best cell for annotation
best_idx = mlp_var["ewf1"].idxmax()
best_depth = mlp_var.loc[best_idx, "depth"]
best_width = mlp_var.loc[best_idx, "width"]

fig, ax = plt.subplots(figsize=(6, 4))

sns.heatmap(
    piv, ax=ax,
    annot=True, fmt=".3f",
    cmap="YlGn",
    linewidths=0.5, linecolor="white",
    annot_kws={"size": 9},
    cbar_kws={"label": "EWF1"},
)

# mark the best cell with a border
row_pos = list(piv.index).index(best_depth)
col_pos = list(piv.columns).index(best_width)
ax.add_patch(plt.Rectangle(
    (col_pos, row_pos), 1, 1,
    fill=False, edgecolor="black", linewidth=2.0, clip_on=False,
))

ax.set_title("MLP Architecture Variation — EWF1 (HIGGS)", pad=10)
ax.set_xlabel("Width (units per layer)")
ax.set_ylabel("Depth (hidden layers)")

plt.tight_layout()
plt.savefig(PLOTS_DIR / "mlp_variation_ewf1.pdf", bbox_inches="tight")
print("Saved: analysis/plots/mlp_variation_ewf1.pdf")
plt.close()
