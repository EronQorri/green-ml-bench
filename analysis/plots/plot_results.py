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
    "LogisticRegression": (123/255, 167/255, 188/255),  # clrLR
    "RandomForest":       (212/255, 149/255, 106/255),  # clrRF
    "XGBoost":            (130/255, 185/255, 154/255),  # clrXC
    "XGBoost_GPU":        (192/255, 112/255, 112/255),  # clrXG
    "MLP_PyTorch":        (155/255, 135/255, 181/255),  # clrML
}
METHOD_PALETTE = {
    "HardwareMonitor (corrected)": ( 90/255, 140/255, 170/255),
    "CodeCarbon":                  (210/255, 175/255, 130/255),
}

df_results = pd.read_csv(RESULTS_DIR / "results.csv")
df_inf = pd.read_csv(RESULTS_DIR / "inference_time.csv")
df_results.columns = df_results.columns.str.strip()
df_inf.columns = df_inf.columns.str.strip()

df_results["dataset"] = df_results["dataset"].astype(str).str.strip()
df_results["model"] = df_results["model"].astype(str).str.strip()
df_results["nrows"] = df_results["nrows"].astype(str).str.strip()
df_inf["dataset"] = df_inf["dataset"].astype(str).str.strip()
df_inf["model"] = df_inf["model"].astype(str).str.strip()
df_inf["nrows"] = df_inf["nrows"].astype(str).str.strip()

# keep only training rows (tuning rows have no accuracy)
df_results = df_results[df_results["accuracy"].notna() & (df_results["accuracy"] != "")]
df_results = df_results.drop_duplicates(subset=["dataset", "nrows", "model"], keep="last")
df_inf = df_inf.drop_duplicates(subset=["dataset", "nrows", "model"], keep="last")

df_results_main = df_results[df_results["nrows"].astype(str) == "all"]

# df = main runs only — used for Plots 1, 2, 3
df = pd.merge(
    df_results_main,
    df_inf[df_inf["nrows"].astype(str) == "all"][["model", "dataset", "nrows", "inference_time"]],
    on=["model", "dataset", "nrows"],
    how="left",
)

# EWF1: portable log-ratio penalty. C0 = 1e-3 kg is a fixed reference unit
# (1 gram CO2), making scores comparable across studies and model pools.
lam = 1.0
C0 = 1e-3  # kg CO2 reference unit
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
    sns.barplot(data=subset, x="model", y="ewf1",
                order=models_in_ds, hue="model", hue_order=models_in_ds,
                palette=MODEL_PALETTE, ax=axes[key], dodge=False)
    axes[key].set_title(f"EWF1: {ds}")
    axes[key].set_xlabel("")
    axes[key].set_ylim(0, None)  # auto-scale; EWF1 is no longer bounded by 1
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


# ── Plot 2: CodeCarbon vs. HardwareMonitor — Underestimation Factor ──────────

cc_cols = ["co2eq_codecarbon_kg", "cpu_power_hw_w"]
if all(c in df.columns for c in cc_cols):
    DATASET_PALETTE = {
        "wine":   (123/255, 167/255, 188/255),  # clrLR
        "credit": (212/255, 149/255, 106/255),  # clrRF
        "higgs":  (130/255, 185/255, 154/255),  # clrXC
    }
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

    datasets_order = [d for d in ["wine", "credit", "higgs"] if d in df_ratio["dataset"].values]
    models_in_data = [m for m in MODEL_ORDER if m in df_ratio["model"].values]

    x = np.arange(len(datasets_order))
    n = len(models_in_data)
    width = 0.15
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


# ── Plot 3: XGBoost CPU vs GPU ────────────────────────────────────────────────

df_xgb = df[df["model"].isin(["XGBoost", "XGBoost_GPU"])].copy()

if not df_xgb.empty:
    XGB_PALETTE = {"XGBoost": MODEL_PALETTE["XGBoost"], "XGBoost_GPU": MODEL_PALETTE["XGBoost_GPU"]}
    xgb_kwargs = dict(hue="model", hue_order=["XGBoost", "XGBoost_GPU"],
                      palette=XGB_PALETTE, errorbar=None)

    fig, axes_arr = plt.subplots(2, 2, figsize=(10, 8))
    axes = {"co2": axes_arr[0, 0], "time": axes_arr[0, 1],
            "acc": axes_arr[1, 0], "inf":  axes_arr[1, 1]}
    fig.suptitle("XGBoost: CPU vs. GPU", fontweight="bold")

    metrics = [
        ("co2eq_kg",        "co2", r"CO$_2$ (kg)",          True),
        ("training_time_s", "time", "Training Time (s)",     True),
        ("accuracy",        "acc",  "Accuracy",              False),
        ("inference_time",  "inf",  "Inference Time (s)",    False),
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


# ── Plot 4: MLP architecture variation — depth × width grid on HIGGS ──────────

mlp_var = df_results_main[df_results_main["dataset"] == "higgs"].copy()
mlp_var = mlp_var[mlp_var["model"].str.match(r"^MLP_\d+x\d+$", na=False)]

if not mlp_var.empty:
    parsed = mlp_var["model"].str.extract(r"^MLP_(\d+)x(\d+)$")
    mlp_var["depth"] = pd.to_numeric(parsed[0])
    mlp_var["width"] = pd.to_numeric(parsed[1])
    mlp_var["co2eq_kg"] = pd.to_numeric(mlp_var["co2eq_kg"], errors="coerce")
    mlp_var["f1"] = pd.to_numeric(mlp_var["f1"], errors="coerce")
    mlp_var = mlp_var.dropna(subset=["depth", "width", "f1", "co2eq_kg"])

    widths = sorted(mlp_var["width"].unique())
    depths = sorted(mlp_var["depth"].unique())
    width_palette = dict(zip(widths, sns.color_palette("viridis", len(widths))))
    depth_palette = dict(zip(depths, sns.color_palette("viridis", len(depths))))

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("MLP Architecture Variation — HIGGS", fontsize=14, fontweight="bold")

    for w in widths:
        sub = mlp_var[mlp_var["width"] == w].sort_values("depth")
        axes[0, 0].plot(sub["depth"], sub["f1"], marker="o",
                        color=width_palette[w], label=f"{w}")
        axes[0, 1].plot(sub["depth"], sub["co2eq_kg"], marker="o",
                        color=width_palette[w], label=f"{w}")
    axes[0, 0].set(xlabel="Depth (hidden layers)", ylabel="Weighted F1",
                   title="F1 vs depth (per width)", xticks=depths)
    axes[0, 1].set(xlabel="Depth (hidden layers)", ylabel="CO₂ (kg, log)",
                   title="CO₂ vs depth (per width)", xticks=depths)
    axes[0, 1].set_yscale("log")
    axes[0, 0].legend(title="Width", fontsize=8)
    axes[0, 1].legend(title="Width", fontsize=8)

    for d in depths:
        sub = mlp_var[mlp_var["depth"] == d].sort_values("width")
        label = f"{d} layer" + ("s" if d > 1 else "")
        axes[1, 0].plot(sub["width"], sub["f1"], marker="o",
                        color=depth_palette[d], label=label)
        axes[1, 1].plot(sub["width"], sub["co2eq_kg"], marker="o",
                        color=depth_palette[d], label=label)
    axes[1, 0].set(xlabel="Width (units per layer)", ylabel="Weighted F1",
                   title="F1 vs width (per depth)", xticks=widths)
    axes[1, 1].set(xlabel="Width (units per layer)", ylabel="CO₂ (kg, log)",
                   title="CO₂ vs width (per depth)", xticks=widths)
    axes[1, 1].set_yscale("log")
    axes[1, 0].legend(title="Depth", fontsize=8)
    axes[1, 1].legend(title="Depth", fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(PLOTS_DIR / "mlp_variation.pdf", bbox_inches="tight")
    print("Saved: analysis/plots/mlp_variation.pdf")
    plt.close()
else:
    print("Skipping MLP variation plot -- no MLP_DxW rows in results.csv.")
    print("  Run run_mlp_variation.py first.")
