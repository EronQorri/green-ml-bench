import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
from scipy.interpolate import make_interp_spline
from scipy.optimize import brentq

mpl.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["Palatino Linotype", "Palatino", "Book Antiqua", "DejaVu Serif"],
    "axes.titlesize":    11,
    "axes.labelsize":    10,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "legend.fontsize":    9,
})

BASE_DIR = Path(__file__).parent.parent.parent
RESULTS_DIR = BASE_DIR / "results"
PLOTS_DIR = Path(__file__).parent

MODEL_ORDER = ["LogisticRegression", "RandomForest", "XGBoost", "XGBoost_GPU", "MLP_PyTorch"]
MODEL_PALETTE = {
    "LogisticRegression": (123/255, 167/255, 188/255),
    "RandomForest":       (212/255, 149/255, 106/255),
    "XGBoost":            (130/255, 185/255, 154/255),
    "XGBoost_GPU":        (192/255, 112/255, 112/255),
    "MLP_PyTorch":        (155/255, 135/255, 181/255),
}

df = pd.read_csv(RESULTS_DIR / "results_scaling.csv")
df.columns = df.columns.str.strip()
df["dataset"] = df["dataset"].astype(str).str.strip()
df["model"] = df["model"].astype(str).str.strip()
df["nrows"] = df["nrows"].astype(str).str.strip()
df["nrows_int"] = pd.to_numeric(df["nrows"], errors="coerce")
df["co2eq_kg"] = pd.to_numeric(df["co2eq_kg"], errors="coerce")
df["training_time_s"] = pd.to_numeric(df["training_time_s"], errors="coerce")
df["f1"] = pd.to_numeric(df["f1"], errors="coerce")

df = df.dropna(subset=["nrows_int"])
df = df.drop_duplicates(subset=["dataset", "nrows", "model"], keep="last")


# ── Plot 1: XGBoost CPU vs GPU — Break-Even ───────────────────────────────────

df_be = df[
    (df["dataset"] == "higgs") &
    (df["model"].isin(["XGBoost", "XGBoost_GPU"]))
].dropna(subset=["co2eq_kg"]).copy()

cpu_sub = df_be[df_be["model"] == "XGBoost"].set_index("nrows_int")["co2eq_kg"]
gpu_sub = df_be[df_be["model"] == "XGBoost_GPU"].set_index("nrows_int")["co2eq_kg"]
common = sorted(set(cpu_sub.index) & set(gpu_sub.index))

if len(common) >= 3:
    XGB_PALETTE_BE = {"XGBoost": MODEL_PALETTE["XGBoost"], "XGBoost_GPU": MODEL_PALETTE["XGBoost_GPU"]}
    x = np.array(common, dtype=float)
    y_cpu = np.array([cpu_sub[n] for n in common])
    y_gpu = np.array([gpu_sub[n] for n in common])

    log_x = np.log10(x)
    k = min(2, len(common) - 1)
    spl_cpu = make_interp_spline(log_x, np.log10(y_cpu), k=k)
    spl_gpu = make_interp_spline(log_x, np.log10(y_gpu), k=k)

    n_be = None
    log_x_search = np.linspace(log_x[0], log_x[-1], 2000)
    diff_vals = spl_cpu(log_x_search) - spl_gpu(log_x_search)
    sign_changes = np.where(np.diff(np.sign(diff_vals)))[0]
    if len(sign_changes) > 0:
        idx = sign_changes[0]
        log_be = brentq(lambda lx: spl_cpu(lx) - spl_gpu(lx), log_x_search[idx], log_x_search[idx + 1])
        n_be = 10 ** log_be

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle("XGBoost CPU vs GPU — Break-Even (Higgs Subsets)", fontsize=13, fontweight="bold")

    x_line = np.logspace(log_x[0], log_x[-1], 400)
    ax.plot(x_line, 10 ** spl_cpu(np.log10(x_line)),
            color=XGB_PALETTE_BE["XGBoost"], label="XGBoost CPU (spline)")
    ax.plot(x_line, 10 ** spl_gpu(np.log10(x_line)),
            color=XGB_PALETTE_BE["XGBoost_GPU"], label="XGBoost GPU (spline)")
    ax.scatter(x, y_cpu, color=XGB_PALETTE_BE["XGBoost"], zorder=5, s=60)
    ax.scatter(x, y_gpu, color=XGB_PALETTE_BE["XGBoost_GPU"], zorder=5, s=60)

    if n_be:
        co2_be = 10 ** spl_cpu(np.log10(n_be))
        ax.axvline(n_be, color="gray", linestyle="--", linewidth=1.2)
        ax.annotate(
            f"Break-Even\n~{n_be:,.0f} rows",
            xy=(n_be, co2_be), xytext=(n_be * 1.5, co2_be * 5),
            fontsize=9, arrowprops=dict(arrowstyle="->", color="gray"),
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Dataset size (rows)")
    ax.set_ylabel("CO₂ (kg)")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "xgb_breakeven_higgs.pdf", bbox_inches="tight")
    print("Saved: analysis/plots/xgb_breakeven_higgs.pdf")
    plt.close()
else:
    print(f"Skipping break-even plot -- need >=3 common nrows, got {len(common)}.")


# ── Plot 2: Dataset-size scaling — all models on Higgs subsets ────────────────

df_scaling = df[df["dataset"] == "higgs"].copy()
scaling_models = [m for m in MODEL_ORDER if m in df_scaling["model"].unique()]

if len(scaling_models) >= 2 and df_scaling["nrows_int"].nunique() >= 3:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Scaling Analysis: Higgs Subsets — All Models", fontsize=13, fontweight="bold")

    for ax, (col, title, log) in zip(axes, [
        ("co2eq_kg",        "CO₂ (kg)",         True),
        ("training_time_s", "Training Time (s)", True),
        ("f1",              "F1-Score",          False),
    ]):
        for model in scaling_models:
            sub = df_scaling[df_scaling["model"] == model].sort_values("nrows_int")
            if sub.empty:
                continue
            ax.plot(sub["nrows_int"], sub[col],
                    marker="o", label=model, color=MODEL_PALETTE[model])
        ax.set_xscale("log")
        if log:
            ax.set_yscale("log")
        ax.set_title(title)
        ax.set_xlabel("Dataset size (rows)")
        ax.set_ylabel(title)

    handles = [plt.Line2D([0], [0], color=MODEL_PALETTE[m], marker="o", label=m)
               for m in scaling_models]
    fig.legend(handles=handles, loc="lower center", ncol=len(scaling_models),
               bbox_to_anchor=(0.5, -0.05), fontsize=9)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "scaling_higgs.pdf", bbox_inches="tight")
    print("Saved: analysis/plots/scaling_higgs.pdf")
    plt.close()
else:
    print(f"Skipping scaling plot -- need >=2 models and >=3 nrows, got "
          f"{len(scaling_models)} models / {df_scaling['nrows_int'].nunique()} nrows.")
