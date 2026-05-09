"""
run_lifecycle_analysis.py — Lifecycle CO2 break-even analysis.

For each model and dataset, computes at how many single-row predictions the
cumulative inference CO2 footprint surpasses the one-time training footprint.

Methodology:
  - Training CO2 : co2eq_kg from results.csv (HW-corrected)
  - CO2 per inference: inference_time_s * cpu_power_hw_w / 1000 / 3600 * carbon_intensity
  - Carbon intensity: derived per run as co2eq_kg / (cpu_energy_hw_wh / 1000) [kg/kWh]
  - Break-even: training_co2 / co2_per_inference

Caveat: cpu_power_hw_w is the average power during training; inference power may
differ. This is noted as a methodological limitation in the thesis.

Output: analysis/plots/lifecycle_breakeven.png
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

BASE_DIR   = Path(__file__).parent
RESULTS    = BASE_DIR / "results" / "results.csv"
INFERENCE  = BASE_DIR / "results" / "inference_time.csv"
PLOTS_DIR  = BASE_DIR / "analysis" / "plots"

MODEL_ORDER   = ["LogisticRegression", "RandomForest", "XGBoost", "XGBoost_GPU", "MLP_PyTorch"]
MODEL_LABELS  = {
    "LogisticRegression": "Logistic Regression",
    "RandomForest":       "Random Forest",
    "XGBoost":            "XGBoost (CPU)",
    "XGBoost_GPU":        "XGBoost (GPU)",
    "MLP_PyTorch":        "MLP",
}
DATASET_ORDER = ["wine", "credit", "higgs"]

def load_data():
    df_r = pd.read_csv(RESULTS)
    df_i = pd.read_csv(INFERENCE)
    for df in (df_r, df_i):
        df.columns = df.columns.str.strip()
        df["model"]   = df["model"].astype(str).str.strip()
        df["dataset"] = df["dataset"].astype(str).str.strip()

    df_r = df_r[df_r["accuracy"].notna() & (df_r["accuracy"] != "")]
    df_r = df_r[df_r["nrows"].astype(str) == "all"]
    df_i = df_i[df_i["nrows"].astype(str) == "all"]

    df_r = df_r.drop_duplicates(subset=["dataset", "model"], keep="last")
    df_i = df_i.drop_duplicates(subset=["dataset", "model"], keep="last")

    df = df_r.merge(
        df_i[["model", "dataset", "inference_time"]],
        on=["model", "dataset"],
        how="inner",
    )
    df["co2eq_kg"]       = pd.to_numeric(df["co2eq_kg"],       errors="coerce")
    df["cpu_power_hw_w"] = pd.to_numeric(df["cpu_power_hw_w"], errors="coerce")
    df["cpu_energy_hw_wh"]= pd.to_numeric(df["cpu_energy_hw_wh"], errors="coerce")
    df["inference_time"] = pd.to_numeric(df["inference_time"], errors="coerce")
    return df.dropna(subset=["co2eq_kg", "cpu_power_hw_w", "cpu_energy_hw_wh", "inference_time"])


def compute_breakeven(df):
    # carbon intensity per run [kg CO2 / kWh]
    df = df.copy()
    df["carbon_intensity"] = df["co2eq_kg"] / (df["cpu_energy_hw_wh"] / 1000)

    # CO2 emitted per single inference [kg]
    df["co2_per_inference"] = (
        df["inference_time"] / 3600 * df["cpu_power_hw_w"] / 1000
        * df["carbon_intensity"]
    )

    df["break_even"] = df["co2eq_kg"] / df["co2_per_inference"]
    return df


def plot_cumulative(df):
    datasets = [d for d in DATASET_ORDER if d in df["dataset"].unique()]
    palette  = dict(zip(MODEL_ORDER, sns.color_palette("tab10", len(MODEL_ORDER))))

    fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 5), sharey=False)
    if len(datasets) == 1:
        axes = [axes]

    for ax, dataset in zip(axes, datasets):
        sub = df[df["dataset"] == dataset]
        max_be = sub["break_even"].max()
        x = np.logspace(0, np.log10(max_be * 10), 500)

        for _, row in sub.iterrows():
            model = row["model"]
            label = MODEL_LABELS.get(model, model)
            color = palette.get(model, "gray")

            cumulative_co2 = x * row["co2_per_inference"]
            ax.plot(x, cumulative_co2, color=color, label=label, linewidth=1.8)

            ax.axhline(row["co2eq_kg"], color=color, linestyle="--", linewidth=0.9, alpha=0.6)
            ax.axvline(row["break_even"], color=color, linestyle=":", linewidth=0.9, alpha=0.5)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(dataset.capitalize(), fontsize=12)
        ax.set_xlabel("Number of predictions")
        ax.set_ylabel("Cumulative CO₂ (kg)")
        ax.grid(True, which="both", alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(MODEL_ORDER),
               bbox_to_anchor=(0.5, -0.08), frameon=False, fontsize=9)
    fig.suptitle(
        "Lifecycle CO₂: cumulative inference vs. training footprint\n"
        "(dashed = training CO₂, dotted = break-even point)",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    out = PLOTS_DIR / "lifecycle_breakeven.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def print_summary(df):
    print("\nBreak-even predictions (training CO2 = cumulative inference CO2):")
    print(f"{'Model':<22} {'Dataset':<10} {'Train CO2 (kg)':>15} "
          f"{'CO2/infer (kg)':>16} {'Break-even':>12}")
    print("-" * 78)
    for _, row in df.sort_values(["dataset", "model"]).iterrows():
        label = MODEL_LABELS.get(row["model"], row["model"])
        print(f"{label:<22} {row['dataset']:<10} {row['co2eq_kg']:>15.6f} "
              f"{row['co2_per_inference']:>16.2e} {row['break_even']:>12,.0f}")


if __name__ == "__main__":
    df = load_data()
    df = compute_breakeven(df)
    print_summary(df)
    plot_cumulative(df)
