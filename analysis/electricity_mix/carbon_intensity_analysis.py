import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from dotenv import load_dotenv

import os
output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "carbon_intensity_germany.png")

load_dotenv()
API_KEY = os.getenv("ELECTRICITYMAPS_API_KEY")

response = requests.get(
    "https://api.electricitymaps.com/v4/carbon-intensity/history?zone=DE",
    headers={"auth-token": API_KEY}
)

data = response.json()
history = data["history"]

timestamps = [datetime.fromisoformat(entry["datetime"].replace("Z", "+00:00")) for entry in history]
intensities = [entry["carbonIntensity"] for entry in history]

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(timestamps, intensities, color="#2ecc71", linewidth=2)
ax.fill_between(timestamps, intensities, alpha=0.1, color="#2ecc71")
ax.axhline(y=381, color="#e74c3c", linestyle="--", linewidth=1.5, label="Static DEU factor (381 gCO₂/kWh)")
ax.set_xlabel("Time (UTC)")
ax.set_ylabel("Carbon Intensity (gCO₂eq/kWh)")
ax.set_title("Hourly Carbon Intensity – Germany (DE)")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m %H:%M"))
ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
plt.xticks(rotation=45)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(output_path, dpi=300)
plt.show()

print(f"Min: {min(intensities)} gCO₂/kWh")
print(f"Max: {max(intensities)} gCO₂/kWh")
print(f"Mean: {sum(intensities)/len(intensities):.1f} gCO₂/kWh")
print(f"Static DEU factor: 381 gCO₂/kWh")