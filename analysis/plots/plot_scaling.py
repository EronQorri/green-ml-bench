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
    "font.family":    "serif",
    "font.serif":     ["Palatino Linotype", "Palatino", "Book Antiqua", "DejaVu Serif"],
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
})

BASE_DIR   = Path(__file__).parent.parent.parent
RESULTS_DIR = BASE_DIR / "results"
PLOTS_DIR  = Path(__file__).parent

MODEL_ORDER = ["LogisticRegression", "RandomForest", "XGBoost", "XGBoost_GPU", "MLP_PyTorch"]
MODEL_PALETTE = {
    "LogisticRegression": (123/255, 167/255, 188/255),
    "RandomForest":       (212/255, 149/255, 106/255),
    "XGBoost":            (130/255, 185/255, 154/255),
    "XGBoost_GPU":        (192/255, 112/255, 112/255),
    "MLP_PyTorch":        (155/255, 135/255, 181/255),
}
MODEL_LABEL = {
    "LogisticRegression": "Logistic Regression",
    "RandomForest":       "Random Forest",
    "XGBoost":            "XGBoost",
    "XGBoost_GPU":        "XGBoost GPU",
    "MLP_PyTorch":        "MLP",
}

# ── Load multiseed data (1k–1M, seeds 42 / 123 / 999) ─────────────────────────
ms = pd.read_csv(RESULTS_DIR / "results_scaling_multiseed.csv")
ms["nrows_int"] = pd.to_numeric(ms["nrows"], errors="coerce")
ms["co2eq_kg"]  = pd.to_numeric(ms["co2eq_kg"], errors="coerce")
ms["f1"]        = pd.to_numeric(ms["f1"], errors="coerce")

ms_mean = ms.groupby(["model", "nrows_int"])[["f1", "co2eq_kg"]].mean().reset_index()
ms_lo   = ms.groupby(["model", "nrows_int"])[["f1", "co2eq_kg"]].min().reset_index()
ms_hi   = ms.groupby(["model", "nrows_int"])[["f1", "co2eq_kg"]].max().reset_index()

# ── All data comes from multiseed CSV (MLP at 5M/11M pasted in with empty seed)
df_all = ms_mean[["model", "nrows_int", "f1", "co2eq_kg"]].copy()


# ── Plot 1: XGBoost CPU vs GPU — Break-Even ───────────────────────────────────

df_be = df_all[df_all["model"].isin(["XGBoost", "XGBoost_GPU"])].dropna(subset=["co2eq_kg"]).copy()
cpu_sub = df_be[df_be["model"] == "XGBoost"].set_index("nrows_int")["co2eq_kg"]
gpu_sub = df_be[df_be["model"] == "XGBoost_GPU"].set_index("nrows_int")["co2eq_kg"]
common  = sorted(set(cpu_sub.index) & set(gpu_sub.index))

if len(common) >= 3:
    x     = np.array(common, dtype=float)
    y_cpu = np.array([cpu_sub[n] for n in common])
    y_gpu = np.array([gpu_sub[n] for n in common])

    log_x = np.log10(x)
    k = min(2, len(common) - 1)
    spl_cpu = make_interp_spline(log_x, np.log10(y_cpu), k=k)
    spl_gpu = make_interp_spline(log_x, np.log10(y_gpu), k=k)

    n_be = None
    log_x_search = np.linspace(log_x[0], log_x[-1], 2000)
    diff_vals    = spl_cpu(log_x_search) - spl_gpu(log_x_search)
    sign_changes = np.where(np.diff(np.sign(diff_vals)))[0]
    if len(sign_changes) > 0:
        idx   = sign_changes[0]
        log_be = brentq(lambda lx: spl_cpu(lx) - spl_gpu(lx),
                        log_x_search[idx], log_x_search[idx + 1])
        n_be = 10 ** log_be

    fig, ax = plt.subplots(figsize=(9, 5))

    x_line = np.logspace(log_x[0], log_x[-1], 400)
    ax.plot(x_line, 10 ** spl_cpu(np.log10(x_line)),
            color=MODEL_PALETTE["XGBoost"], label="XGBoost CPU (spline)")
    ax.plot(x_line, 10 ** spl_gpu(np.log10(x_line)),
            color=MODEL_PALETTE["XGBoost_GPU"], label="XGBoost GPU (spline)")
    ax.scatter(x, y_cpu, color=MODEL_PALETTE["XGBoost"], zorder=5, s=60)
    ax.scatter(x, y_gpu, color=MODEL_PALETTE["XGBoost_GPU"], zorder=5, s=60)

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
    if n_be:
        print(f"  Break-even: ~{n_be:,.0f} rows")
    plt.close()
else:
    print(f"Skipping break-even plot — need >=3 common nrows, got {len(common)}.")


# ── Plot 2: Scaling — CO₂ and F1 across all models ────────────────────────────

scaling_models = [m for m in MODEL_ORDER if m in df_all["model"].unique()]

fig, axes = plt.subplots(1, 2, figsize=(11, 5))

for ax, (col, ylabel, use_log) in zip(axes, [
    ("co2eq_kg", "CO₂ (kg)",    True),
    ("f1",       "Weighted F1", False),
]):
    for model in scaling_models:
        color = MODEL_PALETTE[model]
        label = MODEL_LABEL.get(model, model)

        sub_all  = df_all[df_all["model"] == model].sort_values("nrows_int")
        if sub_all.empty:
            continue

        ax.plot(sub_all["nrows_int"], sub_all[col],
                marker="o", color=color, label=label,
                linewidth=1.5, markersize=4)

        # shaded seed range only for the multiseed portion (1k–1M)
        sub_lo = ms_lo[ms_lo["model"] == model].sort_values("nrows_int")
        sub_hi = ms_hi[ms_hi["model"] == model].sort_values("nrows_int")
        if not sub_lo.empty:
            ax.fill_between(sub_lo["nrows_int"], sub_lo[col], sub_hi[col],
                            alpha=0.15, color=color)

    ax.set_xscale("log")
    if use_log:
        ax.set_yscale("log")
    ax.set_xlabel("Dataset size (rows)")
    ax.set_ylabel(ylabel)

    # readable x-axis tick labels
    ticks = [1_000, 10_000, 50_000, 100_000, 500_000, 1_000_000, 5_000_000, 11_000_000]
    labels = ["1K", "10K", "50K", "100K", "500K", "1M", "5M", "11M"]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=30, ha="right")

handles = [
    plt.Line2D([0], [0], color=MODEL_PALETTE[m], marker="o",
               label=MODEL_LABEL.get(m, m))
    for m in scaling_models
]
fig.legend(handles=handles, loc="lower center", ncol=min(len(scaling_models), 5),
           bbox_to_anchor=(0.5, -0.08), fontsize=9)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "scaling_higgs.pdf", bbox_inches="tight")
print("Saved: analysis/plots/scaling_higgs.pdf")
plt.close()
