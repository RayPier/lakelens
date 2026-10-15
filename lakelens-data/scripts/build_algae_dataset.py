
import json
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "processed" / "cyan_history.json"
OUTPUT = ROOT / "processed" / "algae_lake_dates.csv"

df = pd.DataFrame(json.loads(INPUT.read_text()))

df["date"] = pd.to_datetime(df["date"])
df["cells_ml"] = pd.to_numeric(df["cells_ml"])
df["area_km2"] = pd.to_numeric(df["area_km2"])

# One record per lake, date and satellite detection
df["weighted_concentration"] = (
    df["cells_ml"] * df["area_km2"]
)

lake_dates = (
    df.groupby(["lake", "date"])
    .agg(
        detected_points=("cells_ml", "size"),
        detected_area_km2=("area_km2", "sum"),
        total_weighted_concentration=(
            "weighted_concentration", "sum"
        ),
        max_cells_ml=("cells_ml", "max")
    )
    .reset_index()
)

lake_dates["detected_mean_cells_ml"] = (
    lake_dates["total_weighted_concentration"]
    / lake_dates["detected_area_km2"]
)

# Load the number of known sampling points per lake
POINTS = ROOT / "processed" / "cyan_points.json"

with open(POINTS) as f:
    lake_points = json.load(f)["lakes"]

total_points = {
    lake: len(points)
    for lake, points in lake_points.items()
}

lake_dates["total_grid_points"] = (
    lake_dates["lake"].map(total_points)
)

lake_dates["detection_fraction"] = (
    lake_dates["detected_points"]
    / lake_dates["total_grid_points"]
)

lake_dates = lake_dates.drop(
    columns=["total_weighted_concentration"]
)

lake_dates = lake_dates.sort_values(
    ["lake", "date"]
)

lake_dates.to_csv(OUTPUT, index=False)

print("Lake-date rows:", len(lake_dates))
print("\nFirst 10 rows:")
print(lake_dates.head(10).to_string(index=False))

print("\nDates by lake:")
print(lake_dates.groupby("lake")["date"].nunique())

print("\nSaved:", OUTPUT)
