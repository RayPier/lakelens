
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "models" / "algae_model.joblib"
DATA_PATH = ROOT / "processed" / "algae_with_weather.csv"
OUTPUT_PATH = ROOT / "processed" / "algae_predictions.json"

LAKE_NAMES = {
    "lew": "Lewisville",
    "grp": "Grapevine",
    "rr": "Ray Roberts",
    "tex": "Texoma",
}

# Load trained model and its feature list
artifact = joblib.load(MODEL_PATH)
model = artifact["model"]
features = artifact["features"]

# Load historical satellite/weather dataset
df = pd.read_csv(DATA_PATH, parse_dates=["date"])
df = df.sort_values(["lake", "date"]).copy()

print("\nLatest observation dates by lake:")

for lake_id, group in df.groupby("lake"):
    print(lake_id, group["date"].max().date())

print("\nLatest observation in dataset:", df["date"].max().date())

# Recreate the same features used during training
df["month"] = df["date"].dt.month
df["day_of_year"] = df["date"].dt.dayofyear

df["previous_mean"] = (
    df.groupby("lake")["detected_mean_cells_ml"].shift(1)
)

df["previous_detection_fraction"] = (
    df.groupby("lake")["detection_fraction"].shift(1)
)

df["days_since_previous"] = (
    df.groupby("lake")["date"].diff().dt.days
)

# Recreate one-hot lake columns
df = pd.get_dummies(df, columns=["lake"])

results = {}

for lake_id, name in LAKE_NAMES.items():
    lake_col = f"lake_{lake_id}"

    if lake_col not in df.columns:
        results[lake_id] = {
            "name": name,
            "status": "unavailable",
            "reason": "No historical satellite data",
        }
        continue

    lake_rows = df[df[lake_col]].sort_values("date")

    # Require all model features to be available
    usable = lake_rows.dropna(subset=features)

    if usable.empty:
        results[lake_id] = {
            "name": name,
            "status": "unavailable",
            "reason": "Insufficient data for model features",
        }
        continue

    latest = usable.iloc[-1]
    observation_date = latest["date"]

    # Do not issue a current forecast from stale observations
    age_days = (
        pd.Timestamp.now(tz="UTC").normalize()
        - observation_date.tz_localize("UTC")
    ).days

    if age_days > 14:
        results[lake_id] = {
            "name": name,
            "status": "stale",
            "last_observation": observation_date.strftime("%Y-%m-%d"),
            "age_days": int(age_days),
            "reason": "Latest usable satellite data is over 14 days old",
        }
        continue

    X = pd.DataFrame([latest[features].to_dict()])
    X = X[features]

    probability = float(model.predict_proba(X)[0, 1])

    results[lake_id] = {
        "name": name,
        "status": "experimental",
        "last_observation": observation_date.strftime("%Y-%m-%d"),
        "forecast_date": (
            observation_date + pd.Timedelta(days=7)
        ).strftime("%Y-%m-%d"),
        "probability_elevated": round(probability, 3),
        "predicted_elevated": bool(probability >= 0.5),
        "indicator": "next_7d_detected_mean_above_100k",
        "model": artifact["model_name"],
    }

output = {
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "description": (
        "Experimental prediction of elevated satellite-derived "
        "cyanobacteria indicator at the next weekly observation. "
        "Not a toxin measurement or swimming safety determination."
    ),
    "lakes": results,
}

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH.write_text(json.dumps(output, indent=2))

print(json.dumps(output, indent=2))
print(f"\nSaved predictions to {OUTPUT_PATH}")
