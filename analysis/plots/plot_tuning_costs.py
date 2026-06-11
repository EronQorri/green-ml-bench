import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path

mpl.rcParams.update({
    "font.family":     "serif",
    "font.serif":      ["Palatino Linotype", "Palatino", "Book Antiqua", "DejaVu Serif"],
    "axes.titlesize":  19,
    "axes.labelsize":  17,
    "xtick.labelsize":  17,
    "ytick.labelsize":  17,
    "legend.fontsize":  16,
})

BASE_DIR  = Path(__file__).parent.parent.parent
PLOTS_DIR = Path(__file__).parent

TUNE_ORDER = ["tune_RFC", "tune_XGB", "tune_MLP"]
MODEL_PALETTE = {
    "tune_RFC": (212/255, 149/255, 106/255),
    "tune_XGB": (130/255, 185/255, 154/255),
    "tune_MLP": (155/255, 135/255, 181/255),
}
MODEL_LABELS = {
    "tune_RFC": "RF",
    "tune_XGB": "XGB",
    "tune_MLP": "MLP",
}
DATASET_LABELS = {"wine": "Wine", "credit": "Credit", "higgs": "HIGGS"}

def fmt_time(s):
    if s >= 3600: return f"{s/3600:.1f} h"
    if s >= 60:   return f"{s/60:.0f} m"
    return f"{s:.0f} s"

def fmt_co2(g):
    if g >= 1:    return f"{g:.1f}"
    return f"{g:.2f}"

df = pd.read_csv(BASE_DIR / "results" / "results.csv")
df.columns = df.columns.str.strip()
df["model"]   = df["model"].astype(str).str.strip()
df["dataset"] = df["dataset"].astype(str).str.strip()
df["nrows"]   = df["nrows"].astype(str).str.strip()
df = df[df["model"].str.startswith("tune_") & (df["nrows"] == "all")].copy()
df["co2_g"]  = df["co2eq_kg"].astype(float) * 1000
df["time_s"] = df["training_time_s"].astype(float)
df["model_label"]   = df["model"].map(MODEL_LABELS)
df["dataset_label"] = df["dataset"].map(DATASET_LABELS)

LABEL_ORDER        = [MODEL_LABELS[m] for m in TUNE_ORDER]
DATASET_LABEL_ORDER = [DATASET_LABELS[d] for d in ["wine", "credit", "higgs"]]
palette = {MODEL_LABELS[m]: MODEL_PALETTE[m] for m in TUNE_ORDER}

fig, (ax_co2, ax_time) = plt.subplots(1, 2, figsize=(12, 5))

bp_kwargs = dict(
    x="dataset_label", hue="model_label",
    hue_order=LABEL_ORDER, order=DATASET_LABEL_ORDER,
    palette=palette, errorbar=None, dodge=True,
)

# CO2 subplot
sns.barplot(data=df, y="co2_g", ax=ax_co2, **bp_kwargs)
ax_co2.set_yscale("log")
ax_co2.set_xlabel("")
ax_co2.set_ylabel(r"Tuning Emissions (gCO₂eq, log scale)")
ax_co2.set_title("Emissions")
for container in ax_co2.containers:
    labels = [fmt_co2(v) if not np.isnan(v) else "" for v in container.datavalues]
    ax_co2.bar_label(container, labels=labels, padding=3, fontsize=15)
if ax_co2.get_legend():
    ax_co2.get_legend().remove()
ylo, yhi = ax_co2.get_ylim()
ax_co2.set_ylim(bottom=ylo, top=yhi * 4)

# Duration subplot
sns.barplot(data=df, y="time_s", ax=ax_time, **bp_kwargs)
ax_time.set_yscale("log")
ax_time.set_xlabel("")
ax_time.set_ylabel("Tuning Duration (s, log scale)")
ax_time.set_title("Tuning Duration")
for container in ax_time.containers:
    labels = [fmt_time(v) if not np.isnan(v) else "" for v in container.datavalues]
    ax_time.bar_label(container, labels=labels, padding=3, fontsize=15)
if ax_time.get_legend():
    ax_time.get_legend().remove()
ylo, yhi = ax_time.get_ylim()
ax_time.set_ylim(bottom=ylo, top=yhi * 4)

# Shared legend above
patches = [mpatches.Patch(color=MODEL_PALETTE[m], label=MODEL_LABELS[m]) for m in TUNE_ORDER]
fig.legend(handles=patches, loc="upper center", bbox_to_anchor=(0.5, 1.04),
           ncol=3, fontsize=16, frameon=True, framealpha=0.9)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(PLOTS_DIR / "tuning_costs.pdf", bbox_inches="tight")
print("Saved: analysis/plots/tuning_costs.pdf")
plt.close()
