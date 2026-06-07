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
    "axes.titlesize": 19,
    "axes.labelsize": 18,
    "xtick.labelsize": 17,
    "ytick.labelsize": 17,
    "legend.fontsize": 17,
})

BASE_DIR   = Path(__file__).parent.parent.parent
RESULTS_DIR = BASE_DIR / "results"
PLOTS_DIR  = Path(__file__).parent

MODEL_ORDER = [
    "LogisticRegression", "RandomForest",
    "XGBoost", "XGBoost_GPU", "MLP_PyTorch",
]
MODEL_SHORT = {
    "LogisticRegression": "LR",
    "RandomForest":       "RF",
    "XGBoost":            "XGB",
    "XGBoost_GPU":        "XGB-GPU",
    "MLP_PyTorch":        "MLP",
}
MODEL_PALETTE = {
    "LogisticRegression": (123/255, 167/255, 188/255),
    "RandomForest":       (212/255, 149/255, 106/255),
    "XGBoost":            (130/255, 185/255, 154/255),
    "XGBoost_GPU":        (192/255, 112/255, 112/255),
    "MLP_PyTorch":        (155/255, 135/255, 181/255),
}

# ── load & clean ──────────────────────────────────────────────────────────────
df = pd.read_csv(RESULTS_DIR / "results.csv")
df.columns = df.columns.str.strip()
df["dataset"] = df["dataset"].astype(str).str.strip()
df["model"]   = df["model"].astype(str).str.strip()
df["nrows"]   = df["nrows"].astype(str).str.strip()

df = df[df["accuracy"].notna() & (df["accuracy"] != "")]
df = df[df["nrows"] == "all"]
df = df[df["model"].isin(MODEL_ORDER)]
df = df.drop_duplicates(subset=["dataset", "model"], keep="last")

# merge inference times (for cpu_power_inference_w if needed)
df_inf = pd.read_csv(RESULTS_DIR / "inference_time.csv")
df_inf.columns = df_inf.columns.str.strip()
df_inf["dataset"] = df_inf["dataset"].astype(str).str.strip()
df_inf["model"]   = df_inf["model"].astype(str).str.strip()
df_inf["nrows"]   = df_inf["nrows"].astype(str).str.strip()
df_inf = df_inf[df_inf["nrows"] == "all"]
df_inf = df_inf.drop_duplicates(subset=["dataset", "model"], keep="last")

df = pd.merge(df, df_inf[["model", "dataset", "nrows", "inference_time"]],
              on=["model", "dataset", "nrows"], how="left")

df["cf1"]      = (df["co2eq_kg"] * 1e6) / (df["f1"] * 100)
df["co2eq_g"]  = df["co2eq_kg"] * 1000
df["f1_pct"]   = df["f1"] * 100


def fmt(val):
    if pd.isna(val):
        return ""
    a = abs(val)
    if a == 0:
        return "0"
    elif a < 0.01:
        return f"{val:.2e}"
    elif a < 100:
        return f"{val:.3g}"
    else:
        return f"{val:,.0f}"


# ── per-dataset settings ──────────────────────────────────────────────────────
# co2_log:  use log scale for CO2?
# time_log: use log scale for time?
# wf1_pad, cf1_pad: fractional headroom above max bar for labels
# co2_top_mult, time_top_mult: multiply max value for upper ylim
DATASET_CFG = {
    "wine": {
        "co2_log":        False,
        "time_log":       False,
        "co2_top_mult":   1.45,   # linear: tight ceiling
        "time_top_mult":  1.40,
        "wf1_pad":        0.10,   # absolute headroom above max bar
        "cf1_pad":       0.10,
    },
    "credit": {
        "co2_log":        True,
        "time_log":       True,
        "co2_top_mult":   6.0,    # log: more headroom so labels clear
        "time_top_mult":  6.0,
        "wf1_pad":        0.10,
        "cf1_pad":       0.10,
    },
    "higgs": {
        "co2_log":        True,
        "time_log":       True,
        "co2_top_mult":   3.0,    # fine as-is per user
        "time_top_mult":  3.0,
        "wf1_pad":        0.10,
        "cf1_pad":       0.06,
    },
}

DATASET_LABELS = {"wine": "Wine", "credit": "Credit", "higgs": "HIGGS"}

for ds, cfg in DATASET_CFG.items():
    sub = df[df["dataset"] == ds].copy()
    if sub.empty:
        continue

    models_here  = [m for m in MODEL_ORDER if m in sub["model"].values]
    short_labels = [MODEL_SHORT[m] for m in models_here]
    palette      = [MODEL_PALETTE[m] for m in models_here]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle(DATASET_LABELS[ds], fontsize=22, fontweight="bold")
    ax_wf1, ax_co2, ax_time, ax_cf1 = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    def bar(ax, col):
        vals   = [sub.loc[sub["model"] == m, col].values[0]
                  if m in sub["model"].values else np.nan
                  for m in models_here]
        colors = palette
        bars   = ax.bar(short_labels, vals, color=colors, width=0.55)
        for bar_, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar_.get_x() + bar_.get_width() / 2,
                        bar_.get_height(),
                        fmt(v),
                        ha="center", va="bottom", fontsize=16,
                        transform=ax.transData)
        return vals

    # ── WF1 ──────────────────────────────────────────────────────────────────
    wf1_vals = bar(ax_wf1, "f1_pct")
    ax_wf1.set_title("WF1")
    ax_wf1.set_ylabel("WF1 (%)")
    valid_wf1 = [v for v in wf1_vals if not np.isnan(v)]
    ax_wf1.set_ylim(0, max(valid_wf1) / 0.85)

    # ── Emissions ────────────────────────────────────────────────────────────
    co2_vals = bar(ax_co2, "co2eq_g")
    ax_co2.set_title(r"Emissions (gCO$_2$eq" + (", log)" if cfg["co2_log"] else ", linear)"))
    ax_co2.set_ylabel(r"Emissions (gCO$_2$eq)")
    valid_co2 = [v for v in co2_vals if not np.isnan(v) and v > 0]
    if cfg["co2_log"]:
        ax_co2.set_yscale("log")
        lo = min(valid_co2) * 0.3
        hi = lo * (max(valid_co2) / lo) ** (1 / 0.85)
        ax_co2.set_ylim(lo, hi)
    else:
        ax_co2.set_ylim(0, max(valid_co2) / 0.85)

    # ── Training time ─────────────────────────────────────────────────────────
    time_vals = bar(ax_time, "training_time_s")
    ax_time.set_title("Training Time (s" + (", log)" if cfg["time_log"] else ", linear)"))
    ax_time.set_ylabel("Training Time (s)")
    valid_time = [v for v in time_vals if not np.isnan(v) and v > 0]
    if cfg["time_log"]:
        ax_time.set_yscale("log")
        lo = min(valid_time) * 0.3
        hi = lo * (max(valid_time) / lo) ** (1 / 0.85)
        ax_time.set_ylim(lo, hi)
    else:
        ax_time.set_ylim(0, max(valid_time) / 0.85)

    # ── CF1 ──────────────────────────────────────────────────────────────────
    cf1_vals = bar(ax_cf1, "cf1")
    ax_cf1.set_title(r"CF1 (mgCO$_2$eq/WF1)")
    ax_cf1.set_ylabel(r"CF1 (mgCO$_2$eq/WF1)")
    valid_cf1 = [v for v in cf1_vals if not np.isnan(v)]
    ax_cf1.set_ylim(0, max(valid_cf1) / 0.85)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = PLOTS_DIR / f"{ds}_benchmark.pdf"
    plt.savefig(out, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()
