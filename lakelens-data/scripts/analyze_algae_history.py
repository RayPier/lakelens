
import json
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "processed" / "cyan_history.json"

df = pd.DataFrame(json.loads(INPUT.read_text()))

df["date"] = pd.to_datetime(df["date"])
df["cells_ml"] = pd.to_numeric(df["cells_ml"], errors="coerce")

print("Total observations:", len(df))
print("Unique lake-date combinations:",
      df.groupby(["lake", "date"]).ngroups)

print("\nObservations by lake:")
print(df.groupby("lake").agg(
    observations=("cells_ml", "size"),
    unique_dates=("date", "nunique"),
    first_date=("date", "min"),
    last_date=("date", "max"),
    median_cells=("cells_ml", "median")
).to_string())

print("\nObservations by year:")
print(
    df.groupby([df["date"].dt.year, "lake"])
      .size()
      .unstack(fill_value=0)
      .to_string()
)

print("\nDuplicate lake-point-date records:",
      df.duplicated(["lake", "lat", "lon", "date"]).sum())

print("\nMissing concentrations:", df["cells_ml"].isna().sum())
