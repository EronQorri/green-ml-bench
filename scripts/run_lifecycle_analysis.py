"""
run_lifecycle_analysis.py — Lifecycle CO2 break-even analysis.

For each model and dataset, computes at how many single-row predictions the
cumulative inference CO2 footprint surpasses the one-time training footprint.

Methodology:
  - Training CO2 : co2eq_kg from results.csv (HW-corrected)
  - CO2 per inference: energy_per_inference_wh / 1000 * carbon_intensity
      (energy_per_inference_wh = total CPU window energy / n_predictions over 30 s)
      Fallback when unavailable: inference_time_s * cpu_power_inference_w / 3_600_000 * carbon_intensity
  - Carbon intensity: derived per run as co2eq_kg / (total_energy_wh / 1000) [kg/kWh]
      total_energy_wh includes CPU + GPU + RAM
  - Break-even: training_co2 / co2_per_inference

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
    "LogisticRegression": "LR",
    "RandomForest":       "RF",
    "XGBoost":            "XGB CPU",
    "XGBoost_GPU":        "XGB GPU",
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
    for col in ("cpu_power_inference_w", "energy_per_inference_wh"):
        if col in df_i.columns:
            infer_cols.append(col)

    df = df_r.merge(df_i[infer_cols], on=["model", "dataset"], how="inner")
    df["co2eq_kg"]        = pd.to_numeric(df["co2eq_kg"],        errors="coerce")
    df["cpu_power_hw_w"]  = pd.to_numeric(df["cpu_power_hw_w"],  errors="coerce")
    df["cpu_energy_hw_wh"]= pd.to_numeric(df["cpu_energy_hw_wh"],errors="coerce")
    df["gpu_energy_wh"]   = pd.to_numeric(df["gpu_energy_wh"],   errors="coerce").fillna(0)
    df["ram_energy_wh"]   = pd.to_numeric(df["ram_energy_wh"],   errors="coerce").fillna(0)
    df["inference_time"]  = pd.to_numeric(df["inference_time"],  errors="coerce")
    if "cpu_power_inference_w" in df.columns:
        df["cpu_power_inference_w"] = pd.to_numeric(df["cpu_power_inference_w"], errors="coerce")
    if "energy_per_inference_wh" in df.columns:
        df["energy_per_inference_wh"] = pd.to_numeric(df["energy_per_inference_wh"], errors="coerce")
    return df.dropna(subset=["co2eq_kg", "cpu_power_hw_w", "cpu_energy_hw_wh", "inference_time"])


def compute_breakeven(df):
    df = df.copy()
    # carbon intensity per run [kg CO2 / kWh] — use total measured energy (CPU+GPU+RAM)
    df["total_energy_wh"] = df["cpu_energy_hw_wh"] + df["gpu_energy_wh"] + df["ram_energy_wh"]
    df["carbon_intensity"] = df["co2eq_kg"] / (df["total_energy_wh"] / 1000)

    # CO2 per inference: prefer directly measured energy_per_inference_wh (total monitor energy
    # divided by number of predictions over the 30-second window). Fall back to t × P only
    # when the direct measurement is unavailable.
    if "energy_per_inference_wh" in df.columns and df["energy_per_inference_wh"].notna().any():
        if "cpu_power_inference_w" in df.columns:
            df["power_for_inference"] = df["cpu_power_inference_w"].combine_first(df["cpu_power_hw_w"])
        else:
            df["power_for_inference"] = df["cpu_power_hw_w"]
        fallback = df["inference_time"] / 3600 * df["power_for_inference"] / 1000 * df["carbon_intensity"]
        df["co2_per_inference"] = (
            df["energy_per_inference_wh"] / 1000 * df["carbon_intensity"]
        ).combine_first(fallback)
    else:
        if "cpu_power_inference_w" in df.columns:
            df["power_for_inference"] = df["cpu_power_inference_w"].combine_first(df["cpu_power_hw_w"])
        else:
            df["power_for_inference"] = df["cpu_power_hw_w"]
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
    fs  = 15
    GAP = 1  # blank-position gap between dataset groups

    # Build columns left-to-right: wine | credit | higgs
    cols         = []  # (x_pos, dataset, model, break_even)
    group_bounds = {}  # dataset -> (x_min, x_max)
    pos = 0
    for i, dataset in enumerate(datasets):
        if i > 0:
            pos += GAP
        sub = df[df["dataset"] == dataset].copy()
        sub["model_order"] = sub["model"].map({m: idx for idx, m in enumerate(MODEL_ORDER)})
        sub = sub.sort_values("model_order")
        x_start = pos
        for _, row in sub.iterrows():
            cols.append((pos, dataset, row["model"], row["break_even"]))
            pos += 1
        group_bounds[dataset] = (x_start, pos - 1)

    positions  = np.array([c[0] for c in cols], dtype=float)
    values     = np.array([c[3] for c in cols], dtype=float)
    model_list = [c[2] for c in cols]
    labels     = [MODEL_LABELS[m] for m in model_list]
    colors     = [palette[m] for m in model_list]

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(positions, values, color=colors, width=0.7,
                  edgecolor="white", linewidth=0.6, zorder=2)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val * 1.15,
                f"{val:,.0f}", ha="center", va="bottom", fontsize=fs - 2,
                zorder=3)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=fs - 1, rotation=45, ha="right")
    ax.set_yscale("log")
    ax.set_ylabel("Break-even (# predictions, log scale)", fontsize=fs - 1)
    ax.tick_params(axis="y", labelsize=fs - 1)
    ax.grid(True, axis="y", which="major", alpha=0.25, linestyle="--", zorder=1)
    ax.set_axisbelow(True)
    ax.set_ylim(top=values.max() * 8)

    # Dataset section labels on top (data-x / axes-y coordinates)
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    for dataset in datasets:
        x_min, x_max = group_bounds[dataset]
        label = "HIGGS" if dataset == "higgs" else dataset.capitalize()
        ax.text((x_min + x_max) / 2, 1.02, label,
                transform=trans, ha="center", va="bottom",
                fontsize=fs)

    # Dashed separator lines between groups
    for i in range(len(datasets) - 1):
        x_sep = (group_bounds[datasets[i]][1] + group_bounds[datasets[i + 1]][0]) / 2
        ax.axvline(x_sep, color="gray", linewidth=0.6, linestyle="--", alpha=0.4)

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
