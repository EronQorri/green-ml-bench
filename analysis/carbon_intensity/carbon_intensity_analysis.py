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
    {"label": "Winter",  "start": "2025-01-10T00:00:00.000Z", "end": "2025-01-19T23:59:59.000Z"},
    {"label": "Spring",  "start": "2025-04-10T00:00:00.000Z", "end": "2025-04-19T23:59:59.000Z"},
    {"label": "Summer",  "start": "2025-07-10T00:00:00.000Z", "end": "2025-07-19T23:59:59.000Z"},
    {"label": "Autumn",  "start": "2025-10-10T00:00:00.000Z", "end": "2025-10-19T23:59:59.000Z"},
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


# ── Combined 2x2 seasonal plot ────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharey=True)
axes = axes.flatten()

for ax, season in zip(axes, seasons):
    timestamps, intensities = fetch(season)

    ax.plot(timestamps, intensities, color="#82B99A", linewidth=1.5)
    ax.fill_between(timestamps, intensities, alpha=0.15, color="#82B99A")
    ax.axhline(y=381, color="#C07070", linestyle="--", linewidth=1.2,
               label="Static DEU (381 gCO₂/kWh)")
    ax.set_title(season["label"], fontsize=10)
    ax.set_ylabel("gCO₂eq/kWh", fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, fontsize=8)
    ax.tick_params(axis="y", labelsize=8)

    mean = sum(intensities) / len(intensities)
    print(f"{season['label']}: min={min(intensities)}, max={max(intensities)}, mean={mean:.1f} gCO2/kWh")

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=1, fontsize=9,
           bbox_to_anchor=(0.5, -0.02), frameon=False)
fig.suptitle("Hourly Carbon Intensity – Germany (DE)", fontsize=11)
plt.tight_layout(rect=[0, 0.06, 1, 1])
plt.savefig(os.path.join(base, "carbon_intensity_seasonal.pdf"), bbox_inches="tight")
plt.close()
print("Saved: carbon_intensity_seasonal.pdf")

# ── Single-day plot ───────────────────────────────────────────────────────────
timestamps, intensities = fetch(single_day)

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(timestamps, intensities, color="#82B99A", linewidth=2)
ax.fill_between(timestamps, intensities, alpha=0.15, color="#82B99A")
ax.axhline(y=381, color="#C07070", linestyle="--", linewidth=1.5,
           label="Static DEU factor (381 gCO₂/kWh)")
ax.set_title(f"Hourly Carbon Intensity – Germany (DE) – {single_day['label']}", fontsize=11)
ax.set_xlabel("Time (UTC)", fontsize=10)
ax.set_ylabel("Carbon Intensity (gCO₂eq/kWh)", fontsize=10)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
plt.xticks(rotation=45)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(base, "carbon_intensity_july_day.pdf"), bbox_inches="tight")
plt.close()

mean = sum(intensities) / len(intensities)
print(f"{single_day['label']}: min={min(intensities)}, max={max(intensities)}, mean={mean:.1f} gCO2/kWh")
print("Saved: carbon_intensity_july_day.pdf")
