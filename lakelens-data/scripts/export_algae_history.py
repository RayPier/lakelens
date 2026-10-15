print("SCRIPT STARTED", flush=True)

import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POINTS = ROOT / "processed" / "cyan_points.json"
OUTPUT = ROOT / "processed" / "cyan_history.json"

API = "https://qed.epa.gov/cyanweb/cyan/cyano/location/data"

with open(POINTS) as f:
    points = json.load(f)["lakes"]

records = []

for lake_id, lake_points in points.items():
    print(f"\nProcessing lake: {lake_id}", flush=True)

    for point in lake_points:
        lat = point["lat"]
        lon = point["lon"]

        url = f"{API}/{lat}/{lon}/all"

        try:
            with urllib.request.urlopen(
                url, timeout=60
            ) as response:
                data = json.load(response)

            for obs in data.get("outputs") or []:
                if obs.get("satelliteImageFrequency") != "Weekly":
                    continue

                timestamp = obs.get("imageDateLong")
                concentration = obs.get("cellConcentration")

                if timestamp is None or concentration is None:
                    continue

                date = datetime.fromtimestamp(
                    timestamp / 1000, timezone.utc
                ).date().isoformat()

                records.append({
                    "lake": lake_id,
                    "lat": lat,
                    "lon": lon,
                    "area_km2": point["km2"],
                    "date": date,
                    "cells_ml": concentration
                })

        except Exception as e:
            print(f"Failed {lake_id} {lat},{lon}: {e}")

    print(
        f"Collected {sum(r['lake'] == lake_id for r in records)} "
        f"records for {lake_id}",
        flush=True
    )

OUTPUT.write_text(json.dumps(records, indent=2))

print("\nTotal historical records:", len(records))
print("First 5 records:")
print(json.dumps(records[:5], indent=2))

if records:
    dates = [r["date"] for r in records]
    print("Date range:", min(dates), "to", max(dates))
