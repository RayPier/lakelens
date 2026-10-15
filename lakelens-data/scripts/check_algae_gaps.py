import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

df = pd.read_csv(
    ROOT / "processed" / "algae_lake_dates.csv",
    parse_dates=["date"]
)

for lake, group in df.groupby("lake"):
    group = group.sort_values("date").copy()
    gaps = group["date"].diff().dt.days.dropna()

    print(f"\nLake: {lake}")
    print("Observation dates:", len(group))
    print("Median gap (days):", gaps.median())
    print("Maximum gap (days):", gaps.max())
    print("Gaps over 14 days:", (gaps > 14).sum())

    print("Detection fraction:")
    print(group["detection_fraction"].describe().round(3))