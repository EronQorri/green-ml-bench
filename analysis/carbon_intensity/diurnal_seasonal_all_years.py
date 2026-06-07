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


def fetch_year(year):
    start = datetime(year, 1, 1, tzinfo=timezone.utc)
    end   = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
    step  = timedelta(days=10)
    data  = []
    cur   = start
    while cur < end:
        batch_end = min(cur + step, end)
        r = requests.get(
            "https://api.electricitymaps.com/v3/carbon-intensity/past-range",
            params={
                "zone":  "DE",
                "start": cur.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                "end":   (batch_end - timedelta(seconds=1)).strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            },
            headers={"auth-token": API_KEY},
        )
        data.extend(r.json().get("data", []))
        cur = batch_end
    print(f"{year}: {len(data)} observations")
    return data


def build_matrix(data):
    sums   = np.zeros((24, 12))
    counts = np.zeros((24, 12))
    for entry in data:
        dt = datetime.fromisoformat(entry["datetime"].replace("Z", "+00:00"))
        sums[dt.hour, dt.month - 1]   += entry["carbonIntensity"]
        counts[dt.hour, dt.month - 1] += 1
    return np.where(counts > 0, sums / counts, np.nan)


years    = [2020, 2021, 2022, 2023, 2024, 2025]
matrices = {y: build_matrix(fetch_year(y)) for y in years}

all_vals = np.concatenate([m.ravel() for m in matrices.values()])
vmin, vmax = np.nanmin(all_vals), np.nanmax(all_vals)

month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

fig, axes = plt.subplots(2, 3, figsize=(15, 8))

for idx, (ax, year) in enumerate(zip(axes.ravel(), years)):
    im = ax.imshow(matrices[year], aspect="auto", cmap="RdYlGn_r",
                   origin="upper", vmin=vmin, vmax=vmax)
    ax.set_title(str(year), fontsize=15)
    ax.set_xticks([0, 3, 6, 9])
    ax.set_xticklabels(["Jan", "Apr", "Jul", "Oct"], fontsize=11)
    ax.set_yticks(range(0, 24, 4))
    ax.set_yticklabels([f"{h:02d}" for h in range(0, 24, 4)], fontsize=11)
    if idx % 3 == 0:
        ax.set_ylabel("Hour (UTC)", fontsize=12)

fig.subplots_adjust(right=0.88, hspace=0.35, wspace=0.25)
cax = fig.add_axes([0.91, 0.1, 0.02, 0.75])
cb = fig.colorbar(im, cax=cax)
cb.set_label("gCO₂eq/kWh", fontsize=14)
cb.ax.tick_params(labelsize=12)

out = os.path.join(base, "diurnal_seasonal_all_years.pdf")
plt.savefig(out, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")
