import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

mpl.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["Palatino Linotype", "Palatino", "Book Antiqua", "DejaVu Serif"],
    "axes.titlesize":    11,
    "axes.labelsize":    10,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "legend.fontsize":    9,
})
from pathlib import Path

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
df_results = pd.read_csv(RESULTS_DIR / "results.csv")
df_inf = pd.read_csv(RESULTS_DIR / "inference_time.csv")
df_results.columns = df_results.columns.str.strip()
df_inf.columns = df_inf.columns.str.strip()

df_results["dataset"] = df_results["dataset"].astype(str).str.strip()
df_results["model"]   = df_results["model"].astype(str).str.strip()
df_results["nrows"]   = df_results["nrows"].astype(str).str.strip()
df_inf["dataset"]     = df_inf["dataset"].astype(str).str.strip()
df_inf["model"]       = df_inf["model"].astype(str).str.strip()
df_inf["nrows"]       = df_inf["nrows"].astype(str).str.strip()

df_results = df_results[df_results["accuracy"].notna() & (df_results["accuracy"] != "")]
df_results = df_results.drop_duplicates(subset=["dataset", "nrows", "model"], keep="last")
df_inf     = df_inf.drop_duplicates(subset=["dataset", "nrows", "model"], keep="last")

df_results_main = df_results[df_results["nrows"].astype(str) == "all"]

df = pd.merge(
    df_results_main,
    df_inf[df_inf["nrows"].astype(str) == "all"][["model", "dataset", "nrows", "inference_time", "cpu_power_inference_w"]],
    on=["model", "dataset", "nrows"],
    how="left",
)

lam = 1.0
C0  = 1e-3
df["ewf1"] = df["f1"] / (1 + lam * np.log1p(df["co2eq_kg"] / C0))


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


DATASET_LABELS = {"wine": "Wine", "credit": "Credit", "higgs": "HIGGS"}

# ── Plot 1b: Predictive Performance (Accuracy + Weighted F1) ─────────────────

bp_perf = dict(hue="model", hue_order=MODEL_ORDER, palette=MODEL_PALETTE, errorbar=None)

fig, axes_perf = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Predictive Performance across Datasets", fontsize=13, fontweight="bold")

sns.barplot(data=df, x="dataset", y="accuracy", ax=axes_perf[0], **bp_perf)
axes_perf[0].set_title("Accuracy")
axes_perf[0].set_ylim(0, 1.15)
axes_perf[0].set_xlabel("")
axes_perf[0].set_ylabel("Accuracy")

sns.barplot(data=df, x="dataset", y="f1", ax=axes_perf[1], **bp_perf)
axes_perf[1].set_title("Weighted F1-Score")
axes_perf[1].set_ylim(0, 1.15)
axes_perf[1].set_xlabel("")
axes_perf[1].set_ylabel("Weighted F1")

for ax in axes_perf:
    for container in ax.containers:
        labels = [custom_format(v) for v in container.datavalues]
        ax.bar_label(container, labels=labels, padding=4, fontsize=8)
    if ax.get_legend():
        ax.get_legend().remove()

handles, labels = axes_perf[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.93),
           ncol=5, fontsize=10)
plt.tight_layout(rect=[0, 0, 1, 0.90])
plt.savefig(PLOTS_DIR / "vergleich_performance.pdf", bbox_inches="tight")
print("Saved: analysis/plots/vergleich_performance.pdf")
plt.close()


# ── Plot 1c: Ecological Cost (CO₂ Emissions + Training Time) ─────────────────

bp_eff = dict(hue="model", hue_order=MODEL_ORDER, palette=MODEL_PALETTE, errorbar=None)

fig, axes_eff = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Ecological Cost across Datasets", fontsize=13, fontweight="bold")

sns.barplot(data=df, x="dataset", y="co2eq_kg", ax=axes_eff[0], **bp_eff)
axes_eff[0].set_title("CO₂ Emissions — corrected (kg, log)")
axes_eff[0].set_yscale("log")
axes_eff[0].set_xlabel("")
axes_eff[0].set_ylabel("CO₂ (kg)")

sns.barplot(data=df, x="dataset", y="training_time_s", ax=axes_eff[1], **bp_eff)
axes_eff[1].set_title("Training Time (s, log)")
axes_eff[1].set_yscale("log")
axes_eff[1].set_xlabel("")
axes_eff[1].set_ylabel("Training Time (s)")

for ax in axes_eff:
    for container in ax.containers:
        labels = [custom_format(v) for v in container.datavalues]
        ax.bar_label(container, labels=labels, padding=4, fontsize=8)
    if ax.get_legend():
        ax.get_legend().remove()
    ylo, yhi = ax.get_ylim()
    ax.set_ylim(bottom=ylo, top=yhi * 2)

handles, labels = axes_eff[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.93),
           ncol=5, fontsize=10)
plt.tight_layout(rect=[0, 0, 1, 0.90])
plt.savefig(PLOTS_DIR / "vergleich_efficiency.pdf", bbox_inches="tight")
print("Saved: analysis/plots/vergleich_efficiency.pdf")
plt.close()


# ── Plot 1d: Carbon-Penalised F1 (EWF1) per Dataset ─────────────────────────

fig, axes_ewf1 = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle("EWF1 per Dataset", fontsize=13, fontweight="bold")

for ax, ds in zip(axes_ewf1, ["wine", "credit", "higgs"]):
    subset = df[df["dataset"] == ds]
    if subset.empty:
        ax.set_visible(False)
        continue
    models_in_ds = [m for m in MODEL_ORDER if m in subset["model"].values]
    sns.barplot(data=subset, x="model", y="ewf1",
                order=models_in_ds, hue="model", hue_order=models_in_ds,
                palette=MODEL_PALETTE, ax=ax, dodge=False, errorbar=None)
    ax.set_title(DATASET_LABELS.get(ds, ds))
    ax.set_xlabel("")
    ax.set_ylabel("EWF1" if ds == "wine" else "")
    ax.set_ylim(0, 1.1)
    ax.tick_params(axis="x", rotation=45)
    for container in ax.containers:
        labels = [custom_format(v) for v in container.datavalues]
        ax.bar_label(container, labels=labels, padding=4, fontsize=8)
    if ax.get_legend():
        ax.get_legend().remove()

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig(PLOTS_DIR / "vergleich_ewf1.pdf", bbox_inches="tight")
print("Saved: analysis/plots/vergleich_ewf1.pdf")
plt.close()


# ── Full dashboard (vergleich.pdf, kept for reference) ────────────────────────

layout = [
    ["emissions", "emissions", "time",        "time",        "inf_time",   "inf_time"],
    ["acc",       "acc",       "acc",          "f1",          "f1",         "f1"],
    ["carb_wine", "carb_wine", "carb_credit",  "carb_credit", "carb_higgs", "carb_higgs"],
]

fig, axes = plt.subplot_mosaic(layout, figsize=(18, 12))
fig.suptitle("Model Comparison: Efficiency vs. Accuracy", fontsize=15, fontweight="bold", y=0.98)

bp_kwargs = dict(hue="model", hue_order=MODEL_ORDER, palette=MODEL_PALETTE)

sns.barplot(data=df, x="dataset", y="co2eq_kg",       ax=axes["emissions"], **bp_kwargs)
axes["emissions"].set_title("CO₂ Emissions — corrected (kg, log)")
axes["emissions"].set_yscale("log")
axes["emissions"].set_xlabel("")

sns.barplot(data=df, x="dataset", y="training_time_s", ax=axes["time"], **bp_kwargs)
axes["time"].set_title("Training Time (s, log)")
axes["time"].set_yscale("log")
axes["time"].set_xlabel("")

sns.barplot(data=df, x="dataset", y="inference_time",  ax=axes["inf_time"], **bp_kwargs)
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
    sns.barplot(data=subset, x="model", y="ewf1",
                order=models_in_ds, hue="model", hue_order=models_in_ds,
                palette=MODEL_PALETTE, ax=axes[key], dodge=False)
    axes[key].set_title(f"EWF1: {ds}")
    axes[key].set_xlabel("")
    axes[key].set_ylim(0, None)
    axes[key].tick_params(axis="x", rotation=45)

for name, ax in axes.items():
    for container in ax.containers:
        labels = [custom_format(v) for v in container.datavalues]
        ax.bar_label(container, labels=labels, padding=6, fontsize=8)
    if ax.get_legend():
        ax.get_legend().remove()
    ylo, yhi = ax.get_ylim()
    if ax.get_yscale() == "log":
        ax.set_ylim(bottom=ylo, top=yhi * 2)

handles, labels = axes["emissions"].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.94),
           ncol=5, fontsize=11)
plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.savefig(PLOTS_DIR / "vergleich.pdf", bbox_inches="tight")
print("Saved: analysis/plots/vergleich.pdf")
plt.close()


# ── CodeCarbon vs. HardwareMonitor underestimation factor ────────────────────

cc_cols = ["co2eq_codecarbon_kg", "cpu_power_hw_w"]
if all(c in df.columns for c in cc_cols):
    MODEL_LABELS = {
        "LogisticRegression": "LR",
        "RandomForest":       "RF",
        "XGBoost":            "XGB",
        "XGBoost_GPU":        "XGB-GPU",
        "MLP_PyTorch":        "MLP",
    }

    df_ratio = df[["model", "dataset", "co2eq_kg", "co2eq_codecarbon_kg"]].copy()
    df_ratio = df_ratio[df_ratio["co2eq_codecarbon_kg"] > 0].copy()
    df_ratio["factor"] = df_ratio["co2eq_kg"] / df_ratio["co2eq_codecarbon_kg"]

    datasets_order  = [d for d in ["wine", "credit", "higgs"] if d in df_ratio["dataset"].values]
    models_in_data  = [m for m in MODEL_ORDER if m in df_ratio["model"].values]

    x       = np.arange(len(datasets_order))
    n       = len(models_in_data)
    width   = 0.15
    offsets = np.linspace(-(n - 1) / 2, (n - 1) / 2, n) * width

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle("CodeCarbon Underestimation Factor (HW-corrected / CodeCarbon)",
                 fontsize=13, fontweight="bold")

    for offset, model in zip(offsets, models_in_data):
        subset = df_ratio[df_ratio["model"] == model].set_index("dataset")
        vals = [subset.loc[ds, "factor"] if ds in subset.index else np.nan
                for ds in datasets_order]
        ax.bar(x + offset, vals, width=width, label=MODEL_LABELS[model],
               color=MODEL_PALETTE[model])

    ax.set_xticks(x)
    ax.set_xticklabels([d.capitalize() for d in datasets_order], fontsize=11)
    ax.set_ylabel("Underestimation Factor (HW / CC)", fontsize=10)
    ax.set_ylim(1.0, 2.85)
    ax.axhline(1.0, color="red", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9, ncol=2)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "codecarbon_vs_hw.pdf", bbox_inches="tight")
    print("Saved: analysis/plots/codecarbon_vs_hw.pdf")
    plt.close()
else:
    print("Skipping CodeCarbon vs. HardwareMonitor plot — columns not found.")


# ── XGBoost CPU vs GPU ────────────────────────────────────────────────────────

df_xgb = df[df["model"].isin(["XGBoost", "XGBoost_GPU"])].copy()

if not df_xgb.empty:
    XGB_PALETTE = {"XGBoost": MODEL_PALETTE["XGBoost"], "XGBoost_GPU": MODEL_PALETTE["XGBoost_GPU"]}
    xgb_kwargs  = dict(hue="model", hue_order=["XGBoost", "XGBoost_GPU"],
                       palette=XGB_PALETTE, errorbar=None)

    fig, axes_arr = plt.subplots(2, 2, figsize=(10, 8))
    axes = {"co2": axes_arr[0, 0], "time": axes_arr[0, 1],
            "acc": axes_arr[1, 0], "inf":  axes_arr[1, 1]}
    fig.suptitle("XGBoost: CPU vs. GPU", fontweight="bold")

    metrics = [
        ("co2eq_kg",        "co2",  r"CO$_2$ (kg)",       True),
        ("training_time_s", "time", "Training Time (s)",   True),
        ("accuracy",        "acc",  "Accuracy",            False),
        ("inference_time",  "inf",  "Inference Time (s)",  False),
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
        ylo, yhi = ax.get_ylim()
        if ax.get_yscale() == "log":
            ax.set_ylim(bottom=ylo, top=yhi * 2)
        elif key == "acc":
            ax.set_ylim(top=1.15)
        elif key == "inf":
            ax.set_ylim(top=yhi * 1.1)

    handles, labels = axes["co2"].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.93),
               ncol=2, frameon=True)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(PLOTS_DIR / "xgb_cpu_vs_gpu.pdf", bbox_inches="tight")
    print("Saved: analysis/plots/xgb_cpu_vs_gpu.pdf")
    plt.close()
else:
    print("Skipping XGBoost CPU vs GPU plot — no data found.")


# ── MLP architecture variation (depth × width grid on HIGGS) ─────────────────

mlp_var = df_results[df_results["dataset"] == "higgs"].copy()
mlp_var = mlp_var[mlp_var["model"].str.match(r"^MLP_\d+x\d+$", na=False)]

if not mlp_var.empty:
    parsed = mlp_var["model"].str.extract(r"^MLP_(\d+)x(\d+)$")
    mlp_var["depth"]    = pd.to_numeric(parsed[0])
    mlp_var["width"]    = pd.to_numeric(parsed[1])
    mlp_var["co2eq_kg"] = pd.to_numeric(mlp_var["co2eq_kg"], errors="coerce")
    mlp_var["f1"]       = pd.to_numeric(mlp_var["f1"], errors="coerce")
    mlp_var = mlp_var.dropna(subset=["depth", "width", "f1", "co2eq_kg"])

    piv_f1  = mlp_var.pivot_table(index="depth", columns="width", values="f1",       aggfunc="mean")
    piv_co2 = mlp_var.pivot_table(index="depth", columns="width", values="co2eq_kg", aggfunc="mean")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("MLP Architecture Variation — HIGGS", fontsize=14, fontweight="bold")

    sns.heatmap(piv_f1, ax=axes[0], annot=True, fmt=".3f", cmap="Blues",
                linewidths=0.5, linecolor="white", annot_kws={"size": 9},
                cbar_kws={"label": "Weighted F1"})
    axes[0].set_title("Weighted F1")
    axes[0].set_xlabel("Width (units per layer)")
    axes[0].set_ylabel("Depth (hidden layers)")

    sns.heatmap(piv_co2, ax=axes[1], annot=True, fmt=".3f", cmap="Oranges",
                linewidths=0.5, linecolor="white", annot_kws={"size": 9},
                cbar_kws={"label": "CO₂ (kg)"})
    axes[1].set_title("CO₂ Emissions (kg)")
    axes[1].set_xlabel("Width (units per layer)")
    axes[1].set_ylabel("Depth (hidden layers)")

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(PLOTS_DIR / "mlp_variation.pdf", bbox_inches="tight")
    print("Saved: analysis/plots/mlp_variation.pdf")
    plt.close()
else:
    print("Skipping MLP variation plot -- no MLP_DxW rows in results.csv.")
    print("  Run run_mlp_variation.py first.")
