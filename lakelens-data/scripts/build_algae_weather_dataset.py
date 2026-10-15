
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

ALGAE = ROOT / "processed" / "algae_lake_dates.csv"
WEATHER = ROOT / "processed" / "historical_weather.json"
OUTPUT = ROOT / "processed" / "algae_with_weather.csv"

algae = pd.read_csv(ALGAE, parse_dates=["date"])
weather_json = json.loads(WEATHER.read_text())

# Convert daily weather JSON to a DataFrame
weather_frames = []

for lake_id, daily in weather_json.items():
    frame = pd.DataFrame(daily)
    frame["lake"] = lake_id
    frame["date"] = pd.to_datetime(frame["time"])
    weather_frames.append(frame)

weather = pd.concat(weather_frames, ignore_index=True)

weather = weather.sort_values(["lake", "date"])

# Weather features from the previous 7 and 14 days,
# excluding the current satellite observation date.
features = []

for lake_id, group in weather.groupby("lake"):
    group = group.copy().sort_values("date")

    group["temp_mean_7d"] = (
        group["temperature_2m_mean"]
        .shift(1)
        .rolling(7, min_periods=7)
        .mean()
    )

    group["temp_max_7d"] = (
        group["temperature_2m_max"]
        .shift(1)
        .rolling(7, min_periods=7)
        .max()
    )

    group["rain_sum_7d"] = (
        group["precipitation_sum"]
        .shift(1)
        .rolling(7, min_periods=7)
        .sum()
    )

    group["rain_sum_14d"] = (
        group["precipitation_sum"]
        .shift(1)
        .rolling(14, min_periods=14)
        .sum()
    )

    group["wind_mean_7d"] = (
        group["wind_speed_10m_mean"]
        .shift(1)
        .rolling(7, min_periods=7)
        .mean()
    )

    features.append(group)

weather = pd.concat(features, ignore_index=True)

weather_columns = [
    "lake", "date",
    "temp_mean_7d",
    "temp_max_7d",
    "rain_sum_7d",
    "rain_sum_14d",
    "wind_mean_7d"
]

# Merge on exact lake and observation date
merged = algae.merge(
    weather[weather_columns],
    on=["lake", "date"],
    how="left",
    validate="one_to_one"
)

merged.to_csv(OUTPUT, index=False)

print("Total rows:", len(merged))
print("\nMissing weather features:")
print(merged[weather_columns[2:]].isna().sum())

print("\nSample:")
print(merged.head(10).to_string(index=False))

print("\nSaved:", OUTPUT)
