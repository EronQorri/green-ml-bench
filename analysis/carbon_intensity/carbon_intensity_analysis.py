import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from dotenv import load_dotenv
import os

base = os.path.dirname(os.path.abspath(__file__))

load_dotenv()
API_KEY = os.getenv("ELECTRICITYMAPS_API_KEY")

seasons = [
    {"label": "Winter",  "start": "2024-01-14T00:00:00.000Z", "end": "2024-01-23T23:59:59.000Z"},
    {"label": "Spring",  "start": "2024-04-09T00:00:00.000Z", "end": "2024-04-18T23:59:59.000Z"},
    {"label": "Summer",  "start": "2024-07-15T00:00:00.000Z", "end": "2024-07-24T23:59:59.000Z"},
    {"label": "Autumn",  "start": "2024-10-16T00:00:00.000Z", "end": "2024-10-25T23:59:59.000Z"},
]

single_day = {"label": "Jul 24, 2024", "start": "2024-07-24T00:00:00.000Z", "end": "2024-07-24T23:59:59.000Z"}


def fetch(season):
    r = requests.get(
        "https://api.electricitymaps.com/v3/carbon-intensity/past-range",
        params={"zone": "DE", "start": season["start"], "end": season["end"]},
        headers={"auth-token": API_KEY}
    )
    history = r.json()["data"]
    timestamps  = [datetime.fromisoformat(e["datetime"].replace("Z", "+00:00")) for e in history]
    intensities = [e["carbonIntensity"] for e in history]
    return timestamps, intensities


# ── Individual seasonal plots ─────────────────────────────────────────────────
for season in seasons:
    timestamps, intensities = fetch(season)
    mean = sum(intensities) / len(intensities)
    print(f"{season['label']}: min={min(intensities)}, max={max(intensities)}, mean={mean:.1f} gCO2/kWh")

    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.plot(timestamps, intensities, color="#82B99A", linewidth=1.5)
    ax.fill_between(timestamps, intensities, alpha=0.15, color="#82B99A")
    ax.axhline(y=381, color="#C07070", linestyle="--", linewidth=1.2,
               label="CodeCarbon baseline (381 gCO₂/kWh)")
    ax.set_ylabel("gCO₂eq/kWh", fontsize=13)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, fontsize=12)
    ax.tick_params(axis="y", labelsize=12)
    plt.tight_layout()
    fname = f"carbon_intensity_{season['label'].lower()}.pdf"
    plt.savefig(os.path.join(base, fname), bbox_inches="tight")
    plt.close()
    print(f"Saved: {fname}")


# ── Single-day plot ───────────────────────────────────────────────────────────
timestamps, intensities = fetch(single_day)

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(timestamps, intensities, color="#82B99A", linewidth=2)
ax.fill_between(timestamps, intensities, alpha=0.15, color="#82B99A")
ax.axhline(y=381, color="#C07070", linestyle="--", linewidth=1.5,
           label="CodeCarbon baseline (381 gCO₂/kWh)")
ax.set_xlabel("Time (UTC)", fontsize=14)
ax.set_ylabel("Carbon Intensity (gCO₂eq/kWh)", fontsize=14)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
ax.tick_params(axis="both", labelsize=13)
plt.xticks(rotation=45)
ax.legend(fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(base, "carbon_intensity_july_day.pdf"), bbox_inches="tight")
plt.close()

mean = sum(intensities) / len(intensities)
print(f"{single_day['label']}: min={min(intensities)}, max={max(intensities)}, mean={mean:.1f} gCO2/kWh")
print("Saved: carbon_intensity_july_day.pdf")
