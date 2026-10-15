
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "processed" / "historical_weather.json"

LAKES = {
    "lew": (33.07, -97.00),
    "grp": (32.97, -97.08),
    "rr": (33.37, -97.05),
    "tex": (33.82, -96.60),
}

BASE = "https://archive-api.open-meteo.com/v1/archive"

weather = {}

for lake_id, (lat, lon) in LAKES.items():
    print(f"Fetching weather for {lake_id}...", flush=True)

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": "2020-01-01",
        "end_date": "2026-10-10",
        "daily": ",".join([
            "temperature_2m_mean",
            "temperature_2m_max",
            "precipitation_sum",
            "wind_speed_10m_mean"
        ]),
        "timezone": "America/Chicago"
    }

    url = BASE + "?" + urllib.parse.urlencode(params)

    try:
        with urllib.request.urlopen(url, timeout=90) as response:
            data = json.load(response)

        weather[lake_id] = data["daily"]

        print(
            f"  Retrieved {len(data['daily']['time'])} days",
            flush=True
        )

    except Exception as e:
        print(f"  Failed {lake_id}: {e}")

    time.sleep(1)

OUT.write_text(json.dumps(weather, indent=2))

print(f"\nSaved weather data to {OUT}")
