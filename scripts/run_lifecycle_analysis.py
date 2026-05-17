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

BASE_DIR   = Path(__file__).parent.parent
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
        df["nrows"]   = df["nrows"].astype(str).str.strip()

    df_r = df_r[df_r["accuracy"].notna() & (df_r["accuracy"] != "")]
    df_r = df_r[df_r["nrows"].astype(str) == "all"]
    df_i = df_i[df_i["nrows"].astype(str) == "all"]

    df_r = df_r.drop_duplicates(subset=["dataset", "model"], keep="last")
    df_i = df_i.drop_duplicates(subset=["dataset", "model"], keep="last")

    infer_cols = ["model", "dataset", "inference_time"]
    if "cpu_power_inference_w" in df_i.columns:
        infer_cols.append("cpu_power_inference_w")

    df = df_r.merge(df_i[infer_cols], on=["model", "dataset"], how="inner")
    df["co2eq_kg"]        = pd.to_numeric(df["co2eq_kg"],        errors="coerce")
    df["cpu_power_hw_w"]  = pd.to_numeric(df["cpu_power_hw_w"],  errors="coerce")
    df["cpu_energy_hw_wh"]= pd.to_numeric(df["cpu_energy_hw_wh"],errors="coerce")
    df["inference_time"]  = pd.to_numeric(df["inference_time"],  errors="coerce")
    if "cpu_power_inference_w" in df.columns:
        df["cpu_power_inference_w"] = pd.to_numeric(df["cpu_power_inference_w"], errors="coerce")
    return df.dropna(subset=["co2eq_kg", "cpu_power_hw_w", "cpu_energy_hw_wh", "inference_time"])


def compute_breakeven(df):
    df = df.copy()
    # carbon intensity per run [kg CO2 / kWh]
    df["carbon_intensity"] = df["co2eq_kg"] / (df["cpu_energy_hw_wh"] / 1000)

    # Use measured inference power where available; fall back to training power.
    # Training power is a known overestimate for inference (no backward pass),
    # so inference-phase measurements are preferred.
    if "cpu_power_inference_w" in df.columns:
        df["power_for_inference"] = df["cpu_power_inference_w"].combine_first(df["cpu_power_hw_w"])
    else:
        df["power_for_inference"] = df["cpu_power_hw_w"]

    # CO2 emitted per single inference [kg]
    df["co2_per_inference"] = (
        df["inference_time"] / 3600 * df["power_for_inference"] / 1000
        * df["carbon_intensity"]
    )

    df["break_even"] = df["co2eq_kg"] / df["co2_per_inference"]
    return df


def plot_breakeven_bars(df):
    from matplotlib.transforms import blended_transform_factory

    datasets = [d for d in DATASET_ORDER if d in df["dataset"].unique()]
    palette = {
        "LogisticRegression": (123/255, 167/255, 188/255),
        "RandomForest":       (212/255, 149/255, 106/255),
        "XGBoost":            (130/255, 185/255, 154/255),
        "XGBoost_GPU":        (192/255, 112/255, 112/255),
        "MLP_PyTorch":        (155/255, 135/255, 181/255),
    }
    fs  = 9
    GAP = 1  # blank-position gap between dataset groups

    # Build rows bottom-to-top: reversed DATASET_ORDER puts Wine at the top
    rows         = []  # (y_pos, dataset, model, break_even)
    group_bounds = {}  # dataset -> (y_min, y_max)
    pos = 0
    for i, dataset in enumerate(reversed(datasets)):
        if i > 0:
            pos += GAP
        sub = df[df["dataset"] == dataset].copy()
        sub["model_order"] = sub["model"].map({m: idx for idx, m in enumerate(MODEL_ORDER)})
        sub = sub.sort_values("model_order", ascending=False)
        y_start = pos
        for _, row in sub.iterrows():
            rows.append((pos, dataset, row["model"], row["break_even"]))
            pos += 1
        group_bounds[dataset] = (y_start, pos - 1)

    positions  = np.array([r[0] for r in rows], dtype=float)
    values     = np.array([r[3] for r in rows], dtype=float)
    model_list = [r[2] for r in rows]
    labels     = [MODEL_LABELS[m] for m in model_list]
    colors     = [palette[m] for m in model_list]

    fig, ax = plt.subplots(figsize=(8, 7))

    bars = ax.barh(positions, values, color=colors, height=0.7,
                   edgecolor="white", linewidth=0.6, zorder=2)

    for bar, val in zip(bars, values):
        ax.text(val * 1.05, bar.get_y() + bar.get_height() / 2,
                f"{val:,.0f}", ha="left", va="center", fontsize=fs - 2, zorder=3)

    ax.set_yticks(positions)
    ax.set_yticklabels(labels, fontsize=fs - 1)
    ax.set_xscale("log")
    ax.set_xlabel("Break-even (# predictions)", fontsize=fs - 1)
    ax.grid(True, axis="x", which="major", alpha=0.25, linestyle="--", zorder=1)
    ax.set_axisbelow(True)
    ax.set_xlim(right=values.max() * 4)

    # Dataset section labels on the right (axes-x / data-y coordinates)
    trans = blended_transform_factory(ax.transAxes, ax.transData)
    for dataset in datasets:
        y_min, y_max = group_bounds[dataset]
        ax.text(1.02, (y_min + y_max) / 2, dataset.capitalize(),
                transform=trans, ha="left", va="center",
                fontsize=fs, fontweight="bold")

    # Dashed separator lines between groups
    ds_bottom_to_top = list(reversed(datasets))
    for i in range(len(ds_bottom_to_top) - 1):
        y_sep = (group_bounds[ds_bottom_to_top[i]][1] + group_bounds[ds_bottom_to_top[i + 1]][0]) / 2
        ax.axhline(y_sep, color="gray", linewidth=0.6, linestyle="--", alpha=0.4)

    fig.tight_layout()
    out = PLOTS_DIR / "lifecycle_breakeven.pdf"
    fig.savefig(out, bbox_inches="tight")
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

    print("\n--- LaTeX table rows (paste into tabular) ---")
    for dataset in DATASET_ORDER:
        sub = df[df["dataset"] == dataset].copy()
        sub["model_order"] = sub["model"].map({m: i for i, m in enumerate(MODEL_ORDER)})
        sub = sub.sort_values("model_order")
        print(f"% {dataset}")
        for _, row in sub.iterrows():
            label = MODEL_LABELS.get(row["model"], row["model"])
            train  = f"{row['co2eq_kg']*1e6:.2f}"
            infer  = f"{row['co2_per_inference']*1e9:.4f}"
            be     = f"{row['break_even']:,.0f}"
            print(f"  {label} & {dataset.capitalize()} & {train} & {infer} & {be} \\\\")


if __name__ == "__main__":
    df = load_data()
    df = compute_breakeven(df)
    print_summary(df)
    plot_breakeven_bars(df)
