import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.interpolate import make_interp_spline
from scipy.optimize import brentq

BASE_DIR = Path(__file__).parent.parent.parent
RESULTS_DIR = BASE_DIR / "results"
PLOTS_DIR = Path(__file__).parent

MODEL_ORDER = ["LogisticRegression", "RandomForest", "XGBoost", "XGBoost_GPU", "MLP_PyTorch"]
MODEL_PALETTE = dict(zip(MODEL_ORDER, sns.color_palette("tab10", len(MODEL_ORDER))))

df_results = pd.read_csv(RESULTS_DIR / "results.csv")
df_inf = pd.read_csv(RESULTS_DIR / "inference_time.csv")
df_results.columns = df_results.columns.str.strip()
df_inf.columns = df_inf.columns.str.strip()

df_results["dataset"] = df_results["dataset"].astype(str).str.strip()
df_results["model"] = df_results["model"].astype(str).str.strip()
df_inf["dataset"] = df_inf["dataset"].astype(str).str.strip()
df_inf["model"] = df_inf["model"].astype(str).str.strip()

# keep only training rows (tuning rows have no accuracy)
df_results = df_results[df_results["accuracy"].notna() & (df_results["accuracy"] != "")]
df_results = df_results.drop_duplicates(subset=["dataset", "nrows", "model"], keep="last")
df_inf = df_inf.drop_duplicates(subset=["dataset", "nrows", "model"], keep="last")

# Split: main runs (nrows="all") vs. subset runs (nrows=integer)
df_results_main = df_results[df_results["nrows"].astype(str) == "all"]
df_results_sub  = df_results[df_results["nrows"].astype(str) != "all"]

# df = main runs only — used for Plots 1, 2, 3
df = pd.merge(
    df_results_main,
    df_inf[df_inf["nrows"].astype(str) == "all"][["model", "dataset", "nrows", "inference_time"]],
    on=["model", "dataset", "nrows"],
    how="left",
)

# carbon-optimal score (F1 penalized by scaled time + scaled CO2)
l_1, l_2 = 0.5, 1.0
for col in ["training_time_s", "co2eq_kg"]:
    mn = df.groupby("dataset")[col].transform("min")
    mx = df.groupby("dataset")[col].transform("max")
    df[f"{col}_scaled"] = (df[col] - mn) / (mx - mn).clip(lower=1e-9)

df["carbon_optimal_score"] = df["f1"] / (
    1 + l_1 * df["training_time_s_scaled"] + l_2 * df["co2eq_kg_scaled"]
)


def custom_format(val):
    if pd.isna(val):
        return ""
    abs_val = abs(val)
    if abs_val == 0:
        return "0"
    elif abs_val < 0.01:
        return f"{val:.2e}"
    elif abs_val < 10:
        return f"{val:.3f}".rstrip("0").rstrip(".")
    else:
        return f"{val:.1f}".rstrip("0").rstrip(".")


# ── Plot 1: main comparison dashboard ───────────────────���────────────────────

layout = [
    ["emissions", "emissions", "time",     "time",     "inf_time",   "inf_time"],
    ["acc",       "acc",       "acc",      "f1",       "f1",         "f1"],
    ["carb_wine", "carb_wine", "carb_credit", "carb_credit", "carb_higgs", "carb_higgs"],
]

fig, axes = plt.subplot_mosaic(layout, figsize=(18, 12))
fig.suptitle("Model Comparison: Efficiency vs. Accuracy", fontsize=15, fontweight="bold", y=0.98)

bp_kwargs = dict(hue="model", hue_order=MODEL_ORDER, palette=MODEL_PALETTE)

sns.barplot(data=df, x="dataset", y="co2eq_kg", ax=axes["emissions"], **bp_kwargs)
axes["emissions"].set_title("CO₂ Emissions — corrected (kg, log)")
axes["emissions"].set_yscale("log")
axes["emissions"].set_xlabel("")

sns.barplot(data=df, x="dataset", y="training_time_s", ax=axes["time"], **bp_kwargs)
axes["time"].set_title("Training Time (s, log)")
axes["time"].set_yscale("log")
axes["time"].set_xlabel("")

sns.barplot(data=df, x="dataset", y="inference_time", ax=axes["inf_time"], **bp_kwargs)
axes["inf_time"].set_title("Inference Time per Sample (s, log)")
axes["inf_time"].set_yscale("log")
axes["inf_time"].set_xlabel("")

sns.barplot(data=df, x="dataset", y="accuracy", ax=axes["acc"], **bp_kwargs)
axes["acc"].set_title("Accuracy")
axes["acc"].set_ylim(0, 1.25)
axes["acc"].set_xlabel("")

sns.barplot(data=df, x="dataset", y="f1", ax=axes["f1"], **bp_kwargs)
axes["f1"].set_title("F1-Score")
axes["f1"].set_ylim(0, 1.25)
axes["f1"].set_xlabel("")

for ds, key in zip(["wine", "credit", "higgs"], ["carb_wine", "carb_credit", "carb_higgs"]):
    subset = df[df["dataset"] == ds]
    if subset.empty:
        axes[key].set_visible(False)
        continue
    models_in_ds = [m for m in MODEL_ORDER if m in subset["model"].values]
    sns.barplot(data=subset, x="model", y="carbon_optimal_score",
                order=models_in_ds, hue="model", hue_order=models_in_ds,
                palette=MODEL_PALETTE, ax=axes[key], dodge=False)
    axes[key].set_title(f"Carbon-Score: {ds}")
    axes[key].set_xlabel("")
    axes[key].set_ylim(0, 1.25)
    axes[key].tick_params(axis="x", rotation=45)

for name, ax in axes.items():
    for container in ax.containers:
        labels = [custom_format(v) for v in container.datavalues]
        ax.bar_label(container, labels=labels, padding=6, fontsize=8)
    if ax.get_legend():
        ax.get_legend().remove()

handles, labels = axes["emissions"].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.94),
           ncol=5, fontsize=11)
plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.savefig(PLOTS_DIR / "vergleich.png", dpi=150, bbox_inches="tight")
print("Saved: results/vergleich.png")
plt.close()


# ── Plot 2: CodeCarbon vs. HardwareMonitor (neue RQ) ─────────────────────────

cc_cols = ["co2eq_codecarbon_kg", "cpu_power_hw_w"]
if all(c in df.columns for c in cc_cols):
    df_melt = df[["model", "dataset", "co2eq_kg", "co2eq_codecarbon_kg"]].copy()
    df_melt = df_melt.rename(columns={
        "co2eq_kg": "HardwareMonitor (corrected)",
        "co2eq_codecarbon_kg": "CodeCarbon",
    })
    df_melt = df_melt.melt(
        id_vars=["model", "dataset"],
        var_name="method",
        value_name="co2_kg",
    )

    datasets = ["wine", "credit", "higgs"]
    fig, axes = plt.subplots(1, len(datasets), figsize=(16, 5), sharey=False)
    fig.suptitle("CO₂ Estimate: CodeCarbon vs. HardwareMonitor-corrected", fontsize=13, fontweight="bold")

    for ax, ds in zip(axes, datasets):
        subset = df_melt[df_melt["dataset"] == ds]
        if subset.empty:
            ax.set_visible(False)
            continue
        models_in_ds = [m for m in MODEL_ORDER if m in subset["model"].values]
        sns.barplot(data=subset, x="model", y="co2_kg", hue="method",
                    order=models_in_ds, errorbar=None, ax=ax)
        ax.set_title(ds.capitalize())
        ax.set_xlabel("")
        ax.set_ylabel("CO₂ (kg)" if ds == "wine" else "")
        ax.tick_params(axis="x", rotation=30)
        if ax.get_legend():
            ax.get_legend().remove()

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", fontsize=10)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "codecarbon_vs_hw.png", dpi=150, bbox_inches="tight")
    print("Saved: results/codecarbon_vs_hw.png")
    plt.close()
else:
    print("Skipping CodeCarbon vs. HardwareMonitor plot — columns not found.")


# ── Plot 3: XGBoost CPU vs GPU ────────────────────────────────────────────────

df_xgb = df[df["model"].isin(["XGBoost", "XGBoost_GPU"])].copy()

if not df_xgb.empty:
    XGB_PALETTE = {"XGBoost": MODEL_PALETTE["XGBoost"], "XGBoost_GPU": MODEL_PALETTE["XGBoost_GPU"]}
    xgb_kwargs = dict(hue="model", hue_order=["XGBoost", "XGBoost_GPU"],
                      palette=XGB_PALETTE, errorbar=None)

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    axes = dict(zip(["co2", "time", "acc", "inf"], axes))
    fig.suptitle("XGBoost: CPU vs. GPU", fontsize=13, fontweight="bold")

    metrics = [
        ("co2eq_kg",        "co2", "CO₂ (kg, log)",          True),
        ("training_time_s", "time","Training Time (s, log)",  True),
        ("accuracy",        "acc", "Accuracy",                False),
        ("inference_time",  "inf", "Inference Time (s, log)", True),
    ]

    for col, key, ylabel, log in metrics:
        ax = axes[key]
        sns.barplot(data=df_xgb, x="dataset", y=col, ax=ax, **xgb_kwargs)
        ax.set_title(ylabel.split(" (")[0])
        ax.set_xlabel("")
        ax.set_ylabel(ylabel)
        if log:
            ax.set_yscale("log")
        for container in ax.containers:
            labels = [custom_format(v) for v in container.datavalues]
            ax.bar_label(container, labels=labels, padding=4, fontsize=8)
        if ax.get_legend():
            ax.get_legend().remove()

    handles, labels = axes["co2"].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.98),
               ncol=2, fontsize=10)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "xgb_cpu_vs_gpu.png", dpi=150, bbox_inches="tight")
    print("Saved: results/xgb_cpu_vs_gpu.png")
    plt.close()
else:
    print("Skipping XGBoost CPU vs GPU plot — no data found.")


# ── Plot 4: Higgs Subset Break-Even (from run_higgs_subsets.py data) ──────────

df_higgs_sub = df_results_sub[
    (df_results_sub["dataset"] == "higgs") &
    (df_results_sub["model"].isin(["XGBoost", "XGBoost_GPU"]))
].copy()

df_higgs_sub["nrows_int"] = pd.to_numeric(df_higgs_sub["nrows"], errors="coerce")
df_higgs_sub = df_higgs_sub.dropna(subset=["nrows_int", "co2eq_kg"])
df_higgs_sub["co2eq_kg"] = pd.to_numeric(df_higgs_sub["co2eq_kg"], errors="coerce")
df_higgs_sub = df_higgs_sub.dropna(subset=["co2eq_kg"])

cpu_sub = df_higgs_sub[df_higgs_sub["model"] == "XGBoost"].set_index("nrows_int")["co2eq_kg"]
gpu_sub = df_higgs_sub[df_higgs_sub["model"] == "XGBoost_GPU"].set_index("nrows_int")["co2eq_kg"]
common_sub = sorted(set(cpu_sub.index) & set(gpu_sub.index))

if len(common_sub) >= 3:
    XGB_PALETTE_BE = {"XGBoost": MODEL_PALETTE["XGBoost"], "XGBoost_GPU": MODEL_PALETTE["XGBoost_GPU"]}
    x = np.array(common_sub, dtype=float)
    y_cpu = np.array([cpu_sub[n] for n in common_sub])
    y_gpu = np.array([gpu_sub[n] for n in common_sub])

    log_x = np.log10(x)
    k = min(2, len(common_sub) - 1)
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
    plt.savefig(PLOTS_DIR / "xgb_breakeven_higgs.png", dpi=150, bbox_inches="tight")
    print("Saved: analysis/plots/xgb_breakeven_higgs.png")
    plt.close()
else:
    print(f"Skipping Higgs break-even plot — need ≥3 common nrows, got {len(common_sub)}.")
    print("  Run run_higgs_subsets.py first to collect subset data.")


# ── Plot 5: Dataset-size scaling — all models on Higgs subsets ────────────────

df_scaling = df_results_sub[df_results_sub["dataset"] == "higgs"].copy()
df_scaling["nrows_int"] = pd.to_numeric(df_scaling["nrows"], errors="coerce")
df_scaling = df_scaling.dropna(subset=["nrows_int"])
df_scaling["co2eq_kg"] = pd.to_numeric(df_scaling["co2eq_kg"], errors="coerce")
df_scaling["training_time_s"] = pd.to_numeric(df_scaling["training_time_s"], errors="coerce")
df_scaling["f1"] = pd.to_numeric(df_scaling["f1"], errors="coerce")

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
    plt.savefig(PLOTS_DIR / "scaling_higgs.png", dpi=150, bbox_inches="tight")
    print("Saved: analysis/plots/scaling_higgs.png")
    plt.close()
else:
    print(f"Skipping scaling plot — need ≥2 models and ≥3 nrows, got {len(scaling_models)} models / {df_scaling['nrows_int'].nunique() if not df_scaling.empty else 0} nrows.")
    print("  Run run_scaling_subsets.py first.")
