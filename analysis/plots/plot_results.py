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

df["cf1"] = (df["co2eq_kg"] * 1e6) / (df["f1"] * 100)


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

bp_perf = dict(hue="model", hue_order=MODEL_ORDER, palette=MODEL_PALETTE, errorbar=None,
               linewidth=0.8, edgecolor="white")

def _save_perf_panel(metric, ylabel, filename, pct=False):
    plot_df = df.copy()
    if pct:
        plot_df[metric] = plot_df[metric] * 100
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(data=plot_df, x="dataset", y=metric, ax=ax, **bp_perf)
    ax.set_ylim(0, 109 if pct else 1.2)
    if pct:
        ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_xlabel("", fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)
    ax.tick_params(axis="both", labelsize=17)
    for container in ax.containers:
        labels = [custom_format(v) for v in container.datavalues]
        ax.bar_label(container, labels=labels, padding=4, fontsize=17)
    handles, lbls = ax.get_legend_handles_labels()
    ax.get_legend().remove()
    fig.legend(handles, lbls, loc="upper center", bbox_to_anchor=(0.5, 1.01),
               ncol=5, fontsize=17)
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.savefig(PLOTS_DIR / filename, bbox_inches="tight")
    print(f"Saved: analysis/plots/{filename}")
    plt.close()

_save_perf_panel("accuracy", "ACC",      "vergleich_performance_accuracy.pdf")
_save_perf_panel("f1",       "WF1 (%)", "vergleich_performance_f1.pdf", pct=True)


# ── Plot 1c: Ecological Cost (CO₂ Emissions + Training Time) ─────────────────

def _save_eff_panel(metric, ylabel, filename, scale=1.0):
    plot_df = df.copy()
    if scale != 1.0:
        plot_df[metric] = plot_df[metric] * scale
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(data=plot_df, x="dataset", y=metric, ax=ax,
                hue="model", hue_order=MODEL_ORDER, palette=MODEL_PALETTE, errorbar=None)
    ax.set_yscale("log")
    ax.set_xlabel("", fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)
    ax.tick_params(axis="both", labelsize=17)
    for container in ax.containers:
        labels = [custom_format(v) for v in container.datavalues]
        ax.bar_label(container, labels=labels, padding=4, fontsize=15)
    ylo, yhi = ax.get_ylim()
    ax.set_ylim(bottom=ylo, top=yhi * 2)
    handles, lbls = ax.get_legend_handles_labels()
    ax.get_legend().remove()
    fig.legend(handles, lbls, loc="upper center", bbox_to_anchor=(0.5, 1.01),
               ncol=5, fontsize=17)
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.savefig(PLOTS_DIR / filename, bbox_inches="tight")
    print(f"Saved: analysis/plots/{filename}")
    plt.close()

_save_eff_panel("co2eq_kg", "Emissions (gCO₂eq, log scale)", "vergleich_efficiency_co2.pdf", scale=1000)
_save_eff_panel("training_time_s", "Training Time (s, log scale)", "vergleich_efficiency_time.pdf")


# ── Plot 1d: CF1 per Dataset ─────────────────────────────────────────────────

fig, axes_cf1 = plt.subplots(1, 3, figsize=(14, 5))

for ax, ds in zip(axes_cf1, ["wine", "credit", "higgs"]):
    subset = df[df["dataset"] == ds]
    if subset.empty:
        ax.set_visible(False)
        continue
    models_in_ds = [m for m in MODEL_ORDER if m in subset["model"].values]
    sns.barplot(data=subset, x="model", y="cf1",
                order=models_in_ds, hue="model", hue_order=models_in_ds,
                palette=MODEL_PALETTE, ax=ax, dodge=False, errorbar=None)
    ax.set_title(DATASET_LABELS.get(ds, ds))
    ax.set_xlabel("")
    ax.set_ylabel("CF1 (mg CO₂eq / WF1)" if ds == "wine" else "")
    ax.tick_params(axis="x", rotation=45)
    if ds == "higgs":
        ax.set_yscale("log")
        ax.set_title("HIGGS (log scale)")
    for container in ax.containers:
        labels = [custom_format(v) for v in container.datavalues]
        ax.bar_label(container, labels=labels, padding=4, fontsize=14)
    if ax.get_legend():
        ax.get_legend().remove()
    vals = [v for c in ax.containers for v in c.datavalues if v and v == v]
    if vals:
        if ds == "higgs":
            ax.set_ylim(min(vals) * 0.5, max(vals) * 3)
        else:
            ax.set_ylim(0, max(vals) * 1.20)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "vergleich_cf1.pdf", bbox_inches="tight")
print("Saved: analysis/plots/vergleich_cf1.pdf")
plt.close()

for ds in ["wine", "credit", "higgs"]:
    subset = df[df["dataset"] == ds]
    if subset.empty:
        continue
    fig_s, ax_s = plt.subplots(figsize=(5, 5.5))
    models_in_ds = [m for m in MODEL_ORDER if m in subset["model"].values]
    sns.barplot(data=subset, x="model", y="cf1",
                order=models_in_ds, hue="model", hue_order=models_in_ds,
                palette=MODEL_PALETTE, ax=ax_s, dodge=False, errorbar=None)
    if ds == "higgs":
        ax_s.set_yscale("log")
    ax_s.set_xlabel("", fontsize=18)
    ax_s.set_ylabel("CF1 (mg CO₂eq / WF1)", fontsize=18)
    ax_s.tick_params(axis="x", rotation=45, labelsize=17)
    ax_s.tick_params(axis="y", labelsize=17)
    for container in ax_s.containers:
        labels = [custom_format(v) for v in container.datavalues]
        ax_s.bar_label(container, labels=labels, padding=4, fontsize=18)
    if ax_s.get_legend():
        ax_s.get_legend().remove()
    vals = [v for c in ax_s.containers for v in c.datavalues if v and v == v]
    if vals:
        if ds == "higgs":
            ax_s.set_ylim(min(vals) * 0.5, max(vals) * 3)
        else:
            ax_s.set_ylim(0, max(vals) * 1.20)
    plt.tight_layout()
    fname = f"vergleich_cf1_{ds}.pdf"
    plt.savefig(PLOTS_DIR / fname, bbox_inches="tight")
    print(f"Saved: analysis/plots/{fname}")
    plt.close()


# ── Plot: Tuning CF1 per Dataset ─────────────────────────────────────────────

TUNE_ORDER = ["tune_RFC", "tune_XGB", "tune_MLP"]
TUNE_PALETTE = {
    "tune_RFC": (212/255, 149/255, 106/255),
    "tune_XGB": (130/255, 185/255, 154/255),
    "tune_MLP": (155/255, 135/255, 181/255),
}
TUNE_LABELS = {"tune_RFC": "Random Forest", "tune_XGB": "XGBoost", "tune_MLP": "MLP"}

_df_raw = pd.read_csv(RESULTS_DIR / "results.csv")
_df_raw.columns = _df_raw.columns.str.strip()
_df_raw["model"]   = _df_raw["model"].astype(str).str.strip()
_df_raw["dataset"] = _df_raw["dataset"].astype(str).str.strip()
_df_raw["nrows"]   = _df_raw["nrows"].astype(str).str.strip()
df_tune = _df_raw[_df_raw["model"].str.startswith("tune_") & (_df_raw["nrows"] == "all")].copy()
df_tune["cf1"] = (df_tune["co2eq_kg"].astype(float) * 1e6) / (df_tune["f1"].astype(float) * 100)

fig, axes_tune = plt.subplots(1, 3, figsize=(14, 5))

for ax, ds in zip(axes_tune, ["wine", "credit", "higgs"]):
    subset = df_tune[df_tune["dataset"] == ds]
    if subset.empty:
        ax.set_visible(False)
        continue
    models_in_ds = [m for m in TUNE_ORDER if m in subset["model"].values]
    palette_ds = {TUNE_LABELS[m]: TUNE_PALETTE[m] for m in models_in_ds}
    subset = subset.copy()
    subset["model_label"] = subset["model"].map(TUNE_LABELS)
    sns.barplot(data=subset, x="model_label", y="cf1",
                order=[TUNE_LABELS[m] for m in models_in_ds],
                hue="model_label", hue_order=[TUNE_LABELS[m] for m in models_in_ds],
                palette=palette_ds, ax=ax, dodge=False, errorbar=None)
    if ds == "higgs":
        ax.set_yscale("log")
        ax.set_title("HIGGS (log scale)", fontsize=19)
    else:
        ax.set_title(DATASET_LABELS.get(ds, ds), fontsize=19)
    ax.set_xlabel("")
    ylabel_text = r"CF1" if ds == "wine" else ""
    ax.set_ylabel(ylabel_text, fontsize=17)
    ax.tick_params(axis="x", rotation=45, labelsize=16)
    ax.tick_params(axis="y", labelsize=16)
    for container in ax.containers:
        labels = [custom_format(v) for v in container.datavalues]
        ax.bar_label(container, labels=labels, padding=4, fontsize=15)
    if ax.get_legend():
        ax.get_legend().remove()
    vals = [v for c in ax.containers for v in c.datavalues if v and v == v]
    if vals:
        if ds == "higgs":
            ax.set_ylim(min(vals) * 0.5, max(vals) * 3)
        else:
            ax.set_ylim(0, max(vals) * 1.25)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "tuning_cf1.pdf", bbox_inches="tight")
print("Saved: analysis/plots/tuning_cf1.pdf")
plt.close()


# ── Full dashboard (vergleich.pdf, kept for reference) ────────────────────────

layout = [
    ["emissions", "emissions", "time",        "time",        "inf_time",   "inf_time"],
    ["acc",       "acc",       "acc",          "f1",          "f1",         "f1"],
    ["carb_wine", "carb_wine", "carb_credit",  "carb_credit", "carb_higgs", "carb_higgs"],
]

fig, axes = plt.subplot_mosaic(layout, figsize=(18, 12))

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

_df_dash = df.copy()
_df_dash["f1_pct"] = _df_dash["f1"] * 100
sns.barplot(data=_df_dash, x="dataset", y="f1_pct", ax=axes["f1"], **bp_kwargs)
axes["f1"].set_title("WF1 (%)")
axes["f1"].set_ylim(0, 104)
axes["f1"].set_yticks([0, 20, 40, 60, 80, 100])
axes["f1"].set_xlabel("")

for ds, key in zip(["wine", "credit", "higgs"], ["carb_wine", "carb_credit", "carb_higgs"]):
    subset = df[df["dataset"] == ds]
    if subset.empty:
        axes[key].set_visible(False)
        continue
    models_in_ds = [m for m in MODEL_ORDER if m in subset["model"].values]
    sns.barplot(data=subset, x="model", y="cf1",
                order=models_in_ds, hue="model", hue_order=models_in_ds,
                palette=MODEL_PALETTE, ax=axes[key], dodge=False)
    axes[key].set_title(f"CF1: {ds}")
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


#── XGBoost CPU vs GPU comparison (CO₂, Time, F1, Inference Time) ─────────────
df_xgb = df[df["model"].isin(["XGBoost", "XGBoost_GPU"])].copy()

if not df_xgb.empty:
    df_xgb["co2eq_g"] = df_xgb["co2eq_kg"] * 1000
    df_xgb["inference_time_ms"] = df_xgb["inference_time"] * 1000
    df_xgb["f1_pct"] = df_xgb["f1"] * 100

    XGB_PALETTE = {"XGBoost": MODEL_PALETTE["XGBoost"], "XGBoost_GPU": MODEL_PALETTE["XGBoost_GPU"]}
    
    fig, axes_arr = plt.subplots(2, 2, figsize=(12, 10))
    axes = {"co2": axes_arr[0, 0], "time": axes_arr[0, 1],
            "f1": axes_arr[1, 0], "inf":  axes_arr[1, 1]}
    
    XGB_LABELS = {"XGBoost": "CPU", "XGBoost_GPU": "GPU"}
    df_xgb["model_label"] = df_xgb["model"].map(XGB_LABELS)
    label_order = ["CPU", "GPU"]
    label_palette = {"CPU": MODEL_PALETTE["XGBoost"], "GPU": MODEL_PALETTE["XGBoost_GPU"]}
    
    xgb_kwargs_labeled = dict(hue="model_label", hue_order=label_order,
                              palette=label_palette, errorbar=None)

    metrics = [
        ("co2eq_g",         "co2",  r"Emissions (gCO$_2$eq, log scale)", True),
        ("training_time_s", "time", "Training Time (s, log scale)", True),
        ("f1_pct",          "f1",   "WF1 (%)",                     False),
        ("inference_time_ms", "inf", "Inference Time (ms)",         False),
    ]

    for col, key, ylabel, log in metrics:
        ax = axes[key]
        sns.barplot(data=df_xgb, x="dataset", y=col, ax=ax, **xgb_kwargs_labeled)
        ax.set_xlabel("", fontsize=20)
        ax.set_ylabel(ylabel, fontsize=20)
        ax.tick_params(axis="both", labelsize=17)
        if log:
            ax.set_yscale("log")
        
        for container in ax.containers:
            lbls_obj = ax.bar_label(container,
                                    labels=[custom_format(v) for v in container.datavalues],
                                    padding=4, fontsize=16)
            if key == "f1":
                for lbl in lbls_obj:
                    lbl.set_clip_on(False)

        if ax.get_legend():
            ax.get_legend().remove()

        ylo, yhi = ax.get_ylim()
        if ax.get_yscale() == "log":
            ax.set_ylim(bottom=ylo, top=yhi * 2)
        elif key == "f1":
            ax.set_ylim(0, 109)
            ax.set_yticks([0, 20, 40, 60, 80, 100])
        elif key == "inf":
            ax.set_ylim(top=yhi * 1.15)

    handles, labels = axes["co2"].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02),
               ncol=2, frameon=True, fontsize=19)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(PLOTS_DIR / "xgb_cpu_vs_gpu.pdf", bbox_inches="tight")
    print("Saved: analysis/plots/xgb_cpu_vs_gpu.pdf")
    plt.close()
else:
    print("Skipping XGBoost CPU vs GPU plot — no data found.")
    
# ── MLP architecture variation (depth × width grid on HIGGS) ─────────────────

_mlp_var_cols = ["timestamp", "model", "dataset", "nrows", "accuracy", "f1",
                 "co2eq_kg", "co2eq_codecarbon_kg", "cpu_power_hw_w",
                 "cpu_energy_hw_wh", "gpu_energy_wh", "ram_energy_wh", "training_time_s"]
mlp_var_raw = pd.read_csv(RESULTS_DIR / "mlp_variation.csv", header=None, names=_mlp_var_cols)

mlp_var = mlp_var_raw[mlp_var_raw["dataset"] == "higgs"].copy()
mlp_var = mlp_var[mlp_var["model"].str.match(r"^MLP_\d+x\d+$", na=False)]

if not mlp_var.empty:
    parsed = mlp_var["model"].str.extract(r"^MLP_(\d+)x(\d+)$")
    mlp_var["depth"]   = pd.to_numeric(parsed[0])
    mlp_var["width"]   = pd.to_numeric(parsed[1])
    mlp_var["co2eq_g"] = pd.to_numeric(mlp_var["co2eq_kg"], errors="coerce") * 1000
    mlp_var["f1"]      = pd.to_numeric(mlp_var["f1"], errors="coerce")
    mlp_var = mlp_var.dropna(subset=["depth", "width", "f1", "co2eq_g"])

    piv_f1  = mlp_var.pivot_table(index="depth", columns="width", values="f1",      aggfunc="mean")
    piv_co2 = mlp_var.pivot_table(index="depth", columns="width", values="co2eq_g", aggfunc="mean")

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sns.heatmap(piv_f1 * 100, ax=ax, annot=True, fmt=".1f", cmap="Blues",
                linewidths=0.5, linecolor="white", annot_kws={"size": 19},
                cbar_kws={})
    ax.set_xlabel("Width (units per layer)", fontsize=19)
    ax.set_ylabel("Depth (hidden layers)", fontsize=19)
    ax.tick_params(axis="both", labelsize=18)
    ax.collections[0].colorbar.ax.tick_params(labelsize=17)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "mlp_variation_wf1.pdf", bbox_inches="tight")
    print("Saved: analysis/plots/mlp_variation_wf1.pdf")
    plt.close()

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sns.heatmap(piv_co2, ax=ax, annot=True, fmt=".1f", cmap="Oranges",
                linewidths=0.5, linecolor="white", annot_kws={"size": 19},
                cbar_kws={})
    ax.set_xlabel("Width (units per layer)", fontsize=19)
    ax.set_ylabel("Depth (hidden layers)", fontsize=19)
    ax.tick_params(axis="both", labelsize=18)
    ax.collections[0].colorbar.ax.tick_params(labelsize=17)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "mlp_variation_emissions.pdf", bbox_inches="tight")
    print("Saved: analysis/plots/mlp_variation_emissions.pdf")
    plt.close()
else:
    print("Skipping MLP variation plot -- no MLP_DxW rows in mlp_variation.csv.")
    print("  Run run_mlp_variation.py first.")
