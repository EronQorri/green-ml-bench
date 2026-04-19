import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.parent
RESULTS_DIR = BASE_DIR / "results"
PLOTS_DIR = Path(__file__).parent

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

df = pd.merge(
    df_results,
    df_inf[["model", "dataset", "nrows", "inference_time"]],
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

sns.barplot(data=df, x="dataset", y="co2eq_kg", hue="model", ax=axes["emissions"])
axes["emissions"].set_title("CO₂ Emissions — corrected (kg, log)")
axes["emissions"].set_yscale("log")
axes["emissions"].set_xlabel("")

sns.barplot(data=df, x="dataset", y="training_time_s", hue="model", ax=axes["time"])
axes["time"].set_title("Training Time (s, log)")
axes["time"].set_yscale("log")
axes["time"].set_xlabel("")

sns.barplot(data=df, x="dataset", y="inference_time", hue="model", ax=axes["inf_time"])
axes["inf_time"].set_title("Inference Time per Sample (s, log)")
axes["inf_time"].set_yscale("log")
axes["inf_time"].set_xlabel("")

sns.barplot(data=df, x="dataset", y="accuracy", hue="model", ax=axes["acc"])
axes["acc"].set_title("Accuracy")
axes["acc"].set_ylim(0, 1.25)
axes["acc"].set_xlabel("")

sns.barplot(data=df, x="dataset", y="f1", hue="model", ax=axes["f1"])
axes["f1"].set_title("F1-Score")
axes["f1"].set_ylim(0, 1.25)
axes["f1"].set_xlabel("")

for ds, key in zip(["wine", "credit", "higgs"], ["carb_wine", "carb_credit", "carb_higgs"]):
    subset = df[df["dataset"] == ds]
    if subset.empty:
        axes[key].set_visible(False)
        continue
    sns.barplot(data=subset, x="model", y="carbon_optimal_score", hue="model",
                ax=axes[key], dodge=False)
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

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(data=df_melt, x="model", y="co2_kg", hue="method", ax=ax)
    ax.set_yscale("log")
    ax.set_title("CO₂ Estimate: CodeCarbon vs. HardwareMonitor-corrected")
    ax.set_xlabel("")
    ax.set_ylabel("CO₂ (kg, log)")
    ax.tick_params(axis="x", rotation=20)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "codecarbon_vs_hw.png", dpi=150, bbox_inches="tight")
    print("Saved: results/codecarbon_vs_hw.png")
    plt.close()
else:
    print("Skipping CodeCarbon vs. HardwareMonitor plot — columns not found.")
