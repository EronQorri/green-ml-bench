import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches

mpl.rcParams.update({
    "font.family":     "serif",
    "font.serif":      ["Palatino Linotype", "Palatino", "Book Antiqua", "DejaVu Serif"],
    "axes.titlesize":  11,
    "axes.labelsize":  10,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "legend.fontsize":  9,
})

from pathlib import Path

BASE_DIR  = Path(__file__).parent.parent.parent
PLOTS_DIR = Path(__file__).parent

DATASET_PALETTE = {
    "wine":   (123/255, 167/255, 188/255),
    "credit": (212/255, 149/255, 106/255),
    "higgs":  (130/255, 185/255, 154/255),
}
DATASET_LABELS = {"wine": "Wine", "credit": "Credit", "higgs": "HIGGS"}
MODEL_ORDER  = ["tune_RFC", "tune_XGB", "tune_MLP"]
MODEL_LABELS = {"tune_RFC": "Random Forest", "tune_XGB": "XGBoost", "tune_MLP": "MLP"}
DATASETS     = ["wine", "credit", "higgs"]

df = pd.read_csv(BASE_DIR / "results" / "results.csv")
df.columns = df.columns.str.strip()
df = df[df["model"].str.startswith("tune_")].copy()
df["co2_g"]    = df["co2eq_kg"].astype(float) * 1000
df["time_s"]   = df["training_time_s"].astype(float)

co2  = {m: {} for m in MODEL_ORDER}
time = {m: {} for m in MODEL_ORDER}
for _, row in df.iterrows():
    m, d = row["model"], row["dataset"]
    if m in co2:
        co2[m][d]  = row["co2_g"]
        time[m][d] = row["time_s"]


def fmt_co2(g):
    if g >= 1000:
        return f"{g/1000:.2f} kg"
    return f"{g:.1f} g"


def fmt_time(s):
    if s >= 3600:
        return f"{s/3600:.1f} h"
    if s >= 60:
        return f"{s/60:.0f} min"
    return f"{s:.0f} s"


x      = np.arange(len(MODEL_ORDER))
width  = 0.23
offsets = np.array([-1, 0, 1]) * width

fig, (ax_co2, ax_time) = plt.subplots(
    2, 1, figsize=(9, 7), sharex=True,
    gridspec_kw={"hspace": 0.12},
)
fig.suptitle("Hyperparameter Tuning Overhead", fontsize=13, fontweight="bold", y=0.97)

for ax, data_dict, ylabel in [
    (ax_co2,  co2,  r"CO$_2$ (g)"),
    (ax_time, time, "Duration (s)"),
]:
    for i, ds in enumerate(DATASETS):
        off = offsets[i]
        for j, model in enumerate(MODEL_ORDER):
            val = data_dict[model].get(ds)
            if val is None:
                continue
            ax.bar(x[j] + off, val, width,
                   color=DATASET_PALETTE[ds], zorder=3,
                   edgecolor="white", linewidth=0.4)
            label = fmt_co2(val) if ax is ax_co2 else fmt_time(val)
            ax.text(x[j] + off, val * 1.25, label,
                    ha="center", va="bottom", fontsize=6.5,
                    rotation=45, color="#333333")

    ax.set_yscale("log")
    ax.set_ylabel(ylabel, fontsize=10)
    ax.yaxis.grid(True, which="both", linestyle="--", alpha=0.35, zorder=0)
    ax.set_axisbelow(True)
    ylo, yhi = ax.get_ylim()
    ax.set_ylim(bottom=ylo, top=yhi * 8)  # headroom for labels + annotation

# ── annotate the MLP-HIGGS outlier on both panels ─────────────────────────────
for ax, data_dict in [(ax_co2, co2), (ax_time, time)]:
    val = data_dict["tune_MLP"]["higgs"]
    bx  = x[2] + offsets[2]          # MLP group, higgs bar (rightmost offset)
    ax.annotate(
        "MLP × HIGGS\n≈ 95 h tuning",
        xy=(bx, val),
        xytext=(bx + 0.52, val * 0.35),
        fontsize=7.5,
        color="#222222",
        arrowprops=dict(arrowstyle="->", color="#555555", lw=0.8),
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#aaaaaa", lw=0.7),
    )

# ── x-axis labels (bottom panel only, shared) ─────────────────────────────────
ax_time.set_xticks(x)
ax_time.set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER], fontsize=10)

# ── legend ────────────────────────────────────────────────────────────────────
patches = [mpatches.Patch(color=DATASET_PALETTE[d], label=DATASET_LABELS[d])
           for d in DATASETS]
fig.legend(handles=patches, loc="upper center", bbox_to_anchor=(0.5, 0.935),
           ncol=3, frameon=True, fontsize=9,
           title="Dataset", title_fontsize=9)

fig.text(
    0.5, 0.005,
    "* Random Forest was not tuned on HIGGS; "
    "RFC hyperparameters were set from literature defaults.",
    ha="center", fontsize=7.5, color="#555555", style="italic",
)

plt.savefig(PLOTS_DIR / "tuning_costs.pdf", bbox_inches="tight")
print("Saved: analysis/plots/tuning_costs.pdf")
plt.close()
