import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from dotenv import load_dotenv
import os

output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "electricity_mix_germany_combined.pdf")
load_dotenv()
API_KEY = os.getenv("ELECTRICITYMAPS_API_KEY")

SOURCES = ["nuclear", "solar", "wind", "hydro", "biomass", "coal", "gas", "oil"]
COLORS = {
    "nuclear": "#3df500",
    "solar":   "#ff9d00",
    "wind":    "#9dc2db",
    "hydro":   "#006aff",
    "biomass": "#004e00",
    "coal":    "#0b0f14",
    "gas":     "#ff0000",
    "oil":     "#8e44ad",
}

FONT_LABEL  = 15
FONT_TITLE  = 17
FONT_TICK   = 15
FONT_LEGEND = 15
FONT_NOTE   = 13


def fetch(start, end):
    url = (
        "https://api.electricitymaps.com/v4/power-breakdown/past-range"
        f"?zone=DE&start={start}&end={end}"
    )
    resp = requests.get(url, headers={"auth-token": API_KEY})
    resp.raise_for_status()
    history = resp.json()["data"]
    timestamps = [
        datetime.fromisoformat(e["datetime"].replace("Z", "+00:00"))
        for e in history
    ]
    by_source = {
        s: [e.get("powerProductionBreakdown", {}).get(s) or 0 for e in history]
        for s in SOURCES
    }
    return timestamps, by_source


ts_jan, data_jan = fetch("2025-01-21T00:01:00Z", "2025-01-30T23:59:00Z")
ts_jul, data_jul = fetch("2025-07-01T00:01:00Z", "2025-07-10T23:59:00Z")

fig, axes = plt.subplots(2, 1, figsize=(14, 12), sharex=False)

for ax, ts, data, title in [
    (axes[0], ts_jan, data_jan, "Electricity Mix – Germany (DE) – January 2025"),
    (axes[1], ts_jul, data_jul, "Electricity Mix – Germany (DE) – July 2025"),
]:
    ax.stackplot(
        ts,
        [data[s] for s in SOURCES],
        labels=SOURCES,
        colors=[COLORS[s] for s in SOURCES],
        alpha=0.8,
    )
    ax.set_xlabel("Date", fontsize=FONT_LABEL)
    ax.set_ylabel("Power Production (MW)", fontsize=FONT_LABEL)
    ax.set_title(title, fontsize=FONT_TITLE)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax.tick_params(axis="both", labelsize=FONT_TICK)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.legend(loc="upper left", ncol=4, fontsize=FONT_LEGEND)
    ax.grid(True, alpha=0.3)

fig.text(
    0.5, -0.01,
    "Each data point represents a one-hour average of power production in MW, "
    "recorded at hourly intervals by ElectricityMaps (source: ENTSO-E).",
    ha="center", fontsize=FONT_NOTE, style="italic", color="#444444",
)

plt.tight_layout(rect=[0, 0.02, 1, 1])
plt.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"Saved to {output_path}")
