"""
run_carbon_intensity_analysis.py — Temporal carbon intensity analysis.

Fetches one year of hourly carbon intensity data for the German grid from the
ElectricityMaps API, then counterfactually re-computes the CO2 emissions of
every existing training run in results.csv under different timing scenarios.

This is a pure post-training analysis: no models are retrained. Energy is
derived from the recorded co2eq_kg by dividing out the static 0.381 kg/kWh
factor that CodeCarbon used:
    total_kwh = co2eq_kg / 0.381
The resulting kWh is then multiplied by various carbon intensity values from
the year-long API data (annual mean, best hour, worst hour, summer midday,
winter evening, etc.) to estimate what each training run would have emitted
under different timing.

Outputs:
- results/carbon_intensity_year.csv     (raw API data, one row per hour)
- results/co2_counterfactual.csv        (per-run CO2 under each scenario)
- analysis/carbon_intensity/diurnal_seasonal.png

Requires ELECTRICITYMAPS_API_KEY in .env.
"""

import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent.parent
load_dotenv()
API_KEY = os.getenv("ELECTRICITYMAPS_API_KEY")
if not API_KEY:
    sys.exit("ELECTRICITYMAPS_API_KEY not found in .env")

ZONE = "DE"
STATIC_FACTOR = 0.381  # kg/kWh, CodeCarbon's Germany default
CHUNK_DAYS = 10        # ElectricityMaps free-tier past-range cap

# Year of data: full previous calendar year
END = datetime(2025, 1, 1, tzinfo=timezone.utc)
START = END - timedelta(days=365)

CARBON_FILE = BASE_DIR / "results" / "carbon_intensity_year.csv"
RESULTS_FILE = BASE_DIR / "results" / "results.csv"
COUNTERFACTUAL_FILE = BASE_DIR / "results" / "co2_counterfactual.csv"
PLOT_DIR = BASE_DIR / "analysis" / "carbon_intensity"
PLOT_FILE = PLOT_DIR / "diurnal_seasonal.pdf"
SCENARIO_INTENSITY_PLOT = PLOT_DIR / "scenario_intensities.pdf"
SCENARIO_PER_RUN_PLOT = PLOT_DIR / "scenario_per_run.pdf"

SCENARIO_LABELS = {
    "static_codecarbon":   "Static (CC)",
    "year_mean":           "Year mean",
    "year_best_hour":      "Best hour",
    "year_worst_hour":     "Worst hour",
    "winter_mean":         "Winter mean",
    "summer_mean":         "Summer mean",
    "summer_midday_mean":  "Summer midday",
    "winter_evening_mean": "Winter evening",
}

# Reduced set for the per-run plot
PLOT_SCENARIOS = ["static_codecarbon", "summer_mean", "winter_mean", "year_best_hour", "year_worst_hour"]


# ── 1. Fetch (chunked + cached) ───────────────────────────────────────────────

def fetch_year():
    if CARBON_FILE.exists():
        print(f"Cached: {CARBON_FILE}")
        return pd.read_csv(CARBON_FILE, parse_dates=["datetime"])

    print(f"Fetching {START.date()} to {END.date()} (chunks of {CHUNK_DAYS} days)...")
    records = []
    cursor = START
    while cursor < END:
        chunk_end = min(cursor + timedelta(days=CHUNK_DAYS), END)
        fmt = lambda dt: dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        url = (
            "https://api.electricitymaps.com/v4/carbon-intensity/past-range"
            f"?zone={ZONE}&start={fmt(cursor)}&end={fmt(chunk_end)}"
        )
        r = requests.get(url, headers={"auth-token": API_KEY}, timeout=30)
        r.raise_for_status()
        for e in r.json().get("data", []):
            records.append({"datetime": e["datetime"], "carbon_intensity": e["carbonIntensity"]})
        print(f"  {cursor.date()} -> {chunk_end.date()}: {len(records)} rows total")
        cursor = chunk_end
        time.sleep(0.25)

    df = pd.DataFrame(records)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.drop_duplicates(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
    CARBON_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CARBON_FILE, index=False)
    print(f"Saved: {CARBON_FILE} ({len(df)} hourly records)")
    return df


# ── 2. Scenarios ──────────────────────────────────────────────────────────────

def build_scenarios(df):
    df = df.copy()
    df["hour"] = df["datetime"].dt.hour
    df["month"] = df["datetime"].dt.month
    df["season"] = df["month"].map(lambda m:
        "winter" if m in (12, 1, 2)
        else "spring" if m in (3, 4, 5)
        else "summer" if m in (6, 7, 8)
        else "autumn"
    )

    summer_midday = df[(df["season"] == "summer") & df["hour"].between(11, 14)]["carbon_intensity"]
    winter_evening = df[(df["season"] == "winter") & df["hour"].between(17, 20)]["carbon_intensity"]

    scenarios = {
        "static_codecarbon":   STATIC_FACTOR * 1000,
        "year_mean":           df["carbon_intensity"].mean(),
        "year_best_hour":      df["carbon_intensity"].min(),
        "year_worst_hour":     df["carbon_intensity"].max(),
        "winter_mean":         df[df["season"] == "winter"]["carbon_intensity"].mean(),
        "summer_mean":         df[df["season"] == "summer"]["carbon_intensity"].mean(),
        "summer_midday_mean":  summer_midday.mean(),
        "winter_evening_mean": winter_evening.mean(),
    }
    return df, scenarios


# ── 3. Counterfactual per training run ────────────────────────────────────────

def counterfactual(scenarios):
    if not RESULTS_FILE.exists():
        print(f"No results.csv at {RESULTS_FILE} -- run experiments first.")
        return None

    df = pd.read_csv(RESULTS_FILE)
    df = df.dropna(subset=["co2eq_kg"]).copy()
    df["total_kwh"] = df["co2eq_kg"] / STATIC_FACTOR

    for name, gco2_per_kwh in scenarios.items():
        df[f"co2_{name}_kg"] = df["total_kwh"] * gco2_per_kwh / 1000.0

    COUNTERFACTUAL_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(COUNTERFACTUAL_FILE, index=False)
    print(f"Saved: {COUNTERFACTUAL_FILE}")
    return df


# ── 4. Plot: hour-of-day x month heatmap ──────────────────────────────────────

def plot_diurnal_seasonal(df):
    pivot = df.pivot_table(index="hour", columns="month", values="carbon_intensity", aggfunc="mean")
    pivot = pivot.reindex(columns=range(1, 13))

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn_r", origin="lower")
    ax.set_xticks(range(12))
    ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.set_yticks(range(0, 24, 2))
    ax.set_yticklabels([f"{h:02d}" for h in range(0, 24, 2)])
    ax.set_xlabel("Month")
    ax.set_ylabel("Hour of day (UTC)")
    ax.set_title(f"Mean carbon intensity (gCO2/kWh) — Germany {START.year}/{END.year - 1}")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("gCO2/kWh")

    PLOT_FILE.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(PLOT_FILE, bbox_inches="tight")
    plt.close()
    print(f"Saved: {PLOT_FILE}")


# ── 5. Plot: scenario carbon intensities ─────────────────────────────────────

def plot_scenario_intensities(scenarios):
    keys = list(scenarios.keys())
    labels = [SCENARIO_LABELS[k] for k in keys]
    values = [scenarios[k] for k in keys]
    static = scenarios["static_codecarbon"]

    # Muted palette — shared keys match scenario_per_run exactly
    palette = {
        "static_codecarbon":   "#888888",
        "year_mean":           "#A09880",
        "year_best_hour":      "#82B99A",
        "year_worst_hour":     "#C07070",
        "winter_mean":         "#7BA7BC",
        "summer_mean":         "#C9B49A",
        "summer_midday_mean":  "#D4C088",
        "winter_evening_mean": "#4A7090",
    }
    colors = [palette[k] for k in keys]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, values, color=colors, width=0.6)
    ax.axhline(static, color="#888888", linestyle="--", linewidth=1)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 5, f"{v:.0f}",
                ha="center", va="bottom", fontsize=13)
    ax.set_ylim(top=max(values) * 1.18)
    ax.set_ylabel("Carbon intensity (gCO₂eq/kWh)", fontsize=15)
    ax.tick_params(axis="x", labelsize=14, rotation=20)
    ax.tick_params(axis="y", labelsize=13)
    plt.tight_layout()
    plt.savefig(SCENARIO_INTENSITY_PLOT, bbox_inches="tight")
    plt.close()
    print(f"Saved: {SCENARIO_INTENSITY_PLOT}")


# ── 6. Plot: CO2 per dataset, averaged over models, across scenarios ──────────

def plot_scenario_per_run(cf, scenarios):
    if cf is None or cf.empty:
        return

    keys = PLOT_SCENARIOS
    cf = cf[~cf["model"].str.startswith("tune_")].copy()
    cf = cf.drop_duplicates(subset=["dataset", "model"], keep="last")
    if cf.empty:
        return

    dataset_order  = ["wine", "credit", "higgs"]
    dataset_labels = {"wine": "Wine", "credit": "Credit Card", "higgs": "HIGGS"}
    groups = [d for d in dataset_order if d in cf["dataset"].unique()]

    # The % deviation from static is identical for every run/dataset — it is
    # purely the ratio of intensity factors. One bar per scenario is enough.
    # static, summer mean, winter mean, best hour, worst hour
    colors = ["#888888", "#C9B49A", "#7BA7BC", "#82B99A", "#C07070"]

    static_intensity = scenarios["static_codecarbon"]
    labels, values, bar_colors = [], [], []
    for k, c in zip(keys, colors):
        labels.append(SCENARIO_LABELS[k])
        values.append(scenarios[k] / static_intensity * 100)
        bar_colors.append(c)

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, values, color=bar_colors, width=0.55)
    ax.axhline(100, color="#888888", linestyle="--", linewidth=1)

    for bar, val in zip(bars, values):
        sign = "+" if val - 100 > 0 else ""
        pct_diff = val - 100
        annotation = "baseline" if pct_diff == 0 else f"{sign}{pct_diff:.0f}%"
        ax.text(bar.get_x() + bar.get_width() / 2, val + 1.5, annotation,
                ha="center", va="bottom", fontsize=15)

    ax.set_ylabel("Relative carbon emissions (%)", fontsize=17)
    ax.set_ylim(0, max(values) * 1.2)
    ax.tick_params(axis="x", labelsize=16)
    ax.tick_params(axis="y", labelsize=15)
    plt.tight_layout()
    plt.savefig(SCENARIO_PER_RUN_PLOT, bbox_inches="tight")
    plt.close()
    print(f"Saved: {SCENARIO_PER_RUN_PLOT}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df_raw = fetch_year()
    df, scenarios = build_scenarios(df_raw)

    print("\nCarbon intensity scenarios (gCO2/kWh):")
    for k, v in scenarios.items():
        print(f"  {k:25s} {v:7.1f}")

    cf = counterfactual(scenarios)
    if cf is not None and len(cf) > 0:
        ratio = scenarios["year_worst_hour"] / scenarios["year_best_hour"]
        print(f"\nWorst-to-best timing factor: {ratio:.1f}x")
        print(f"Counterfactual table written for {len(cf)} runs.")

    plot_diurnal_seasonal(df)
    plot_scenario_intensities(scenarios)
    plot_scenario_per_run(cf, scenarios)
