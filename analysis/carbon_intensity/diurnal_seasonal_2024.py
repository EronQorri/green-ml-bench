import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
import os

base = os.path.dirname(os.path.abspath(__file__))
load_dotenv()
API_KEY = os.getenv("ELECTRICITYMAPS_API_KEY")

all_data = []
start = datetime(2024, 1, 1, tzinfo=timezone.utc)
end_year = datetime(2025, 1, 1, tzinfo=timezone.utc)
step = timedelta(days=10)

current = start
while current < end_year:
    batch_end = min(current + step, end_year)
    r = requests.get(
        "https://api.electricitymaps.com/v3/carbon-intensity/past-range",
        params={
            "zone":  "DE",
            "start": current.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            "end":   (batch_end - timedelta(seconds=1)).strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        },
        headers={"auth-token": API_KEY},
    )
    chunk = r.json().get("data", [])
    all_data.extend(chunk)
    print(f"{current.strftime('%Y-%m-%d')} – {batch_end.strftime('%Y-%m-%d')}: {len(chunk)} obs")
    current = batch_end

print(f"Total: {len(all_data)} observations")

sums   = np.zeros((24, 12))
counts = np.zeros((24, 12))

for entry in all_data:
    dt = datetime.fromisoformat(entry["datetime"].replace("Z", "+00:00"))
    sums[dt.hour, dt.month - 1]   += entry["carbonIntensity"]
    counts[dt.hour, dt.month - 1] += 1

matrix = np.where(counts > 0, sums / counts, np.nan)

month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

fig, ax = plt.subplots(figsize=(12, 7.5))
im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn_r", origin="upper",
               vmin=np.nanmin(matrix), vmax=np.nanmax(matrix))

ax.set_xticks([0, 3, 6, 9])
ax.set_xticklabels(["Jan", "Apr", "Jul", "Oct"], fontsize=17)
ax.set_yticks(range(0, 24, 2))
ax.set_yticklabels([f"{h:02d}" for h in range(0, 24, 2)], fontsize=17)
ax.set_xlabel("Month", fontsize=17)
ax.set_ylabel("Hour of Day (UTC)", fontsize=17)
cbar = plt.colorbar(im, ax=ax)
cbar.set_label("gCO₂eq/kWh", fontsize=17)
cbar.ax.tick_params(labelsize=16)
plt.tight_layout()

out = os.path.join(base, "diurnal_seasonal_2024.pdf")
plt.savefig(out, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")
