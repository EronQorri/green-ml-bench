import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from dotenv import load_dotenv
import os

output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "electricity_mix_germany_july.pdf")
load_dotenv()
API_KEY = os.getenv("ELECTRICITYMAPS_API_KEY")

response = requests.get(
    "https://api.electricitymaps.com/v4/power-breakdown/past-range?zone=DE&start=2025-07-01T00:01:00Z&end=2025-07-10T23:59:00Z",
    headers={"auth-token": API_KEY}
)

data = response.json()
history = data["data"]

timestamps = [datetime.fromisoformat(entry["datetime"].replace("Z", "+00:00")) for entry in history]

sources = ["nuclear", "solar", "wind", "hydro", "biomass", "coal", "gas", "oil"]
colors = {
    "nuclear": "#3df500",
    "solar": "#ff9d00",
    "wind": "#9dc2db",
    "hydro": "#006aff",
    "biomass": "#004e00",
    "coal": "#0b0f14",
    "gas": "#ff0000",
    "oil": "#8e44ad",
}

data_by_source = {}
for source in sources:
    data_by_source[source] = [
        entry.get("powerProductionBreakdown", {}).get(source) or 0
        for entry in history
    ]

fig, ax = plt.subplots(figsize=(14, 6))
ax.stackplot(
    timestamps,
    [data_by_source[s] for s in sources],
    labels=sources,
    colors=[colors[s] for s in sources],
    alpha=0.8
)

ax.set_xlabel("Date", fontsize=15)
ax.set_ylabel("Power Production (MW)", fontsize=15)

ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))
ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
ax.tick_params(axis="both", labelsize=15)
plt.xticks(rotation=45)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=4, fontsize=15, frameon=True)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.subplots_adjust(bottom=0.25)
plt.savefig(output_path, dpi=300)
print(f"Saved to {output_path}")
