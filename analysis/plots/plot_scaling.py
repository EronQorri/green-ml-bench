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

MODEL_ORDER = ["LogisticRegression", "RandomForest", "XGBoost_CPU", "XGBoost_GPU", "MLP_PyTorch"]
MODEL_PALETTE = {
    "LogisticRegression": (123/255, 167/255, 188/255),
    "RandomForest":       (212/255, 149/255, 106/255),
    "XGBoost_CPU":        (130/255, 185/255, 154/255),
    "XGBoost_GPU":        (192/255, 112/255, 112/255),
    "MLP_PyTorch":        (155/255, 135/255, 181/255),
}
MODEL_LABEL = {
    "LogisticRegression": "Logistic Regression",
    "RandomForest":       "Random Forest",
    "XGBoost_CPU":        "XGBoost CPU",
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

df_be = df_all[df_all["model"].isin(["XGBoost_CPU", "XGBoost_GPU"])].dropna(subset=["co2eq_kg"]).copy()
cpu_sub = df_be[df_be["model"] == "XGBoost_CPU"].set_index("nrows_int")["co2eq_kg"]
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
    ax.plot(x_line, 10 ** spl_cpu(np.log10(x_line)) * 1000,
            color=MODEL_PALETTE["XGBoost_CPU"], label="CPU")
    ax.plot(x_line, 10 ** spl_gpu(np.log10(x_line)) * 1000,
            color=MODEL_PALETTE["XGBoost_GPU"], label="GPU")
    ax.scatter(x, y_cpu * 1000, color=MODEL_PALETTE["XGBoost_CPU"], zorder=5, s=80)
    ax.scatter(x, y_gpu * 1000, color=MODEL_PALETTE["XGBoost_GPU"], zorder=5, s=80)

    if n_be:
        co2_be = 10 ** spl_cpu(np.log10(n_be)) * 1000
        ax.axvline(n_be, color="gray", linestyle="--", linewidth=1.2)
        ax.annotate(
            f"Break-Even\n~{n_be:,.0f} rows",
            xy=(n_be, co2_be), xytext=(n_be * 1.5, co2_be * 5),
            fontsize=13, arrowprops=dict(arrowstyle="->", color="gray"),
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Dataset size (rows, log scale)", fontsize=15)
    ax.set_ylabel("Emissions (gCO₂eq, log scale)", fontsize=15)
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.tick_params(axis="both", labelsize=13)
    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.legend(fontsize=13)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "xgb_breakeven_higgs.pdf", bbox_inches="tight")
    print("Saved: analysis/plots/xgb_breakeven_higgs.pdf")
    if n_be:
        print(f"  Break-even: ~{n_be:,.0f} rows")
    plt.close()
else:
    print(f"Skipping break-even plot — need >=3 common nrows, got {len(common)}.")


# ── Plot 2a / 2b / 2c: Scaling — two panel PDFs + shared legend PDF ──────────

scaling_models = [m for m in MODEL_ORDER if m in df_all["model"].unique()]

LABEL_SIZE  = 22
TICK_SIZE   = 19
LEGEND_SIZE = 17

ACTUAL_TICKS  = [1_000, 10_000, 50_000, 100_000, 200_000, 500_000,
                 1_000_000, 5_000_000, 8_800_000]
TICK_LABELS   = ["1k", "10k", "50k", "100k", "200k", "500k", "1M", "5M", "8.8M"]

SCALING_LABEL = {
    "LogisticRegression": "LR",
    "RandomForest":       "RF",
    "XGBoost_CPU":        "XGB CPU",
    "XGBoost_GPU":        "XGB GPU",
    "MLP_PyTorch":        "MLP",
}

handles = [
    plt.Line2D([0], [0], color=MODEL_PALETTE[m], marker="o", linewidth=2,
               markersize=7, label=SCALING_LABEL.get(m, m))
    for m in scaling_models
]

def _draw_scaling_panel(ax, col, ylabel, use_log):
    for model in scaling_models:
        color = MODEL_PALETTE[model]
        sub_all = df_all[df_all["model"] == model].sort_values("nrows_int")
        if sub_all.empty:
            continue
        y = sub_all[col] * 1000 if col == "co2eq_kg" else (sub_all[col] * 100 if col == "f1" else sub_all[col])
        ax.plot(sub_all["nrows_int"], y,
                marker="o", color=color, linewidth=2.5, markersize=7)
        sub_lo = ms_lo[ms_lo["model"] == model].sort_values("nrows_int")
        sub_hi = ms_hi[ms_hi["model"] == model].sort_values("nrows_int")
        if not sub_lo.empty:
            ylo = sub_lo[col] * 1000 if col == "co2eq_kg" else (sub_lo[col] * 100 if col == "f1" else sub_lo[col])
            yhi = sub_hi[col] * 1000 if col == "co2eq_kg" else (sub_hi[col] * 100 if col == "f1" else sub_hi[col])
            ax.fill_between(sub_lo["nrows_int"], ylo, yhi, alpha=0.15, color=color)
    ax.set_xscale("log")
    if use_log:
        ax.set_yscale("log")
    ax.set_xlabel("Dataset size (rows, log scale)", fontsize=LABEL_SIZE)
    ax.set_ylabel(ylabel, fontsize=LABEL_SIZE)
    ax.set_xticks(ACTUAL_TICKS)
    ax.set_xticklabels(TICK_LABELS, rotation=35, ha="right")
    if use_log:
        ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.tick_params(axis="both", labelsize=TICK_SIZE)

# ── 2a: Emissions ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 6))
_draw_scaling_panel(ax, "co2eq_kg", "Emissions (gCO₂eq, log scale)", True)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "scaling_higgs_emissions.pdf", bbox_inches="tight")
print("Saved: analysis/plots/scaling_higgs_emissions.pdf")
plt.close()

# ── 2b: Weighted F1 ───────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 6))
_draw_scaling_panel(ax, "f1", "WF1 (%)", False)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "scaling_higgs_f1.pdf", bbox_inches="tight")
print("Saved: analysis/plots/scaling_higgs_f1.pdf")
plt.close()

# ── 2c: Shared legend strip ───────────────────────────────────────────────────
fig_leg, ax_leg = plt.subplots(figsize=(10, 0.6))
ax_leg.axis("off")
ax_leg.legend(handles=handles, loc="center", ncol=len(scaling_models),
              fontsize=LEGEND_SIZE, frameon=False)
plt.tight_layout(pad=0.1)
plt.savefig(PLOTS_DIR / "scaling_higgs_legend.pdf", bbox_inches="tight")
print("Saved: analysis/plots/scaling_higgs_legend.pdf")
plt.close()


# ── Plot 3: CF1 Heatmap across models × dataset sizes ────────────────────────

import seaborn as sns

df_cf1 = df_all.copy()
df_cf1["cf1"] = (df_cf1["co2eq_kg"] * 1e6) / (df_cf1["f1"] * 100)
df_cf1["model_label"] = df_cf1["model"].map(SCALING_LABEL)
df_cf1["size_label"]  = df_cf1["nrows_int"].map(
    dict(zip(ACTUAL_TICKS, TICK_LABELS))
)

# pivot: rows = models (fixed order), columns = dataset sizes
row_order = [SCALING_LABEL[m] for m in MODEL_ORDER if m in df_cf1["model"].unique()]
col_order  = [l for l in TICK_LABELS if l in df_cf1["size_label"].unique()]

pivot = (df_cf1
         .dropna(subset=["cf1"])
         .pivot_table(index="model_label", columns="size_label",
                      values="cf1", aggfunc="mean")
         .reindex(index=row_order, columns=col_order))

# log-scale the values for color mapping so large RF outliers don't wash out detail
log_pivot = np.log10(pivot.where(pivot > 0))

HM_LABEL  = 25
HM_TICK   = 22
HM_ANNOT  = 20

fig, ax = plt.subplots(figsize=(13, 5))

sns.heatmap(
    log_pivot,
    ax=ax,
    cmap="YlOrRd",
    annot=pivot.round(1),       # show raw CF1 values as annotation
    fmt=".1f",
    annot_kws={"size": HM_ANNOT},
    linewidths=0.4,
    linecolor="white",
    cbar_kws={"label": "CF1 (mgCO₂eq / WF1), log scale"},
)

# grey out missing cells (RF above 500k)
for (r, c), val in np.ndenumerate(pivot.values):
    if np.isnan(val):
        ax.add_patch(plt.Rectangle((c, r), 1, 1,
                     fill=True, color="#cccccc", zorder=3))
        ax.text(c + 0.5, r + 0.5, "—", ha="center", va="center",
                fontsize=HM_ANNOT, color="#666666", zorder=4)

ax.set_xlabel("Dataset size (rows)", fontsize=HM_LABEL)
ax.set_ylabel("")
ax.tick_params(axis="x", labelsize=HM_TICK, rotation=0)
ax.tick_params(axis="y", labelsize=HM_TICK, rotation=0)
ax.collections[0].colorbar.ax.tick_params(labelsize=HM_TICK - 2)
ax.collections[0].colorbar.set_label("CF1 (mgCO₂eq / WF1), log scale",
                                      fontsize=HM_LABEL - 2)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "scaling_higgs_cf1_heatmap.pdf", bbox_inches="tight")
print("Saved: analysis/plots/scaling_higgs_cf1_heatmap.pdf")
plt.close()
