
from datetime import datetime, timedelta, timezone
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

ARCHIVE_API = "https://archive-api.open-meteo.com/v1/archive"
RECENT_API = "https://api.open-meteo.com/v1/forecast"

VARIABLES = [
    "temperature_2m_mean",
    "temperature_2m_max",
    "precipitation_sum",
    "wind_speed_10m_mean",
]

def fetch_daily(base, params):
    url = base + "?" + urllib.parse.urlencode(params)

    with urllib.request.urlopen(url, timeout=90) as response:
        data = json.load(response)

    if "daily" not in data:
        raise ValueError(f"No daily weather data: {data}")

    return data["daily"]


def combine_daily(archive, recent):
    # Archive is the historical foundation.
    # Recent data fills gaps and takes precedence on overlapping dates.
    by_date = {}

    for source in [archive, recent]:
        dates = source["time"]

        for i, date in enumerate(dates):
            by_date[date] = {
                variable: source[variable][i]
                for variable in VARIABLES
            }

    dates = sorted(by_date)

    combined = {"time": dates}

    for variable in VARIABLES:
        combined[variable] = [
            by_date[date][variable]
            for date in dates
        ]

    return combined


weather = {}

today = datetime.now(timezone.utc).date()
end_date = today - timedelta(days=2)

# Forecast API supports recent historical days.
# Request up to 92 days back to cover archive delays.
recent_start = today - timedelta(days=91)

for lake_id, (lat, lon) in LAKES.items():
    print(f"Fetching weather for {lake_id}...", flush=True)

    common = {
        "latitude": lat,
        "longitude": lon,
        "daily": ",".join(VARIABLES),
        "timezone": "America/Chicago",
    }

    archive_params = {
        **common,
        "start_date": "2020-01-01",
        "end_date": end_date.isoformat(),
    }

    recent_params = {
        **common,
        "past_days": 92,
        "forecast_days": 0,
    }

    try:
        archive = fetch_daily(ARCHIVE_API, archive_params)
        print("  Archive ends:", archive["time"][-1])

        recent = fetch_daily(RECENT_API, recent_params)
        print("  Recent API ends:", recent["time"][-1])

        # Do not include days after our requested cutoff.
        recent = {
            key: [
                value
                for date, value in zip(recent["time"], values)
                if recent_start.isoformat() <= date <= end_date.isoformat()
            ]
            for key, values in recent.items()
        }

        combined = combine_daily(archive, recent)

        # Ensure no missing dates in the recent period.
        dates = set(combined["time"])
        required_dates = {
            (recent_start + timedelta(days=i)).isoformat()
            for i in range((end_date - recent_start).days + 1)
        }

        missing_dates = required_dates - dates

        if missing_dates:
            raise ValueError(
                f"Missing recent weather dates: {sorted(missing_dates)}"
            )

        weather[lake_id] = combined

        print("  Combined ends:", combined["time"][-1])
        print("  Total days:", len(combined["time"]))

    except Exception as e:
        raise RuntimeError(
            f"Weather refresh failed for {lake_id}: {e}"
        ) from e

    time.sleep(1)

# Only overwrite the existing file if all four lakes succeeded.
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(weather, indent=2))

print(f"\nSaved weather data to {OUT}")
