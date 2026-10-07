"""Fetch satellite cyanobacteria (blue-green algae) for each lake from EPA CyAN.

Source: EPA CyAN (https://qed.epa.gov/cyanweb/), weekly Sentinel-3 OLCI composites at 300 m,
reported as cyanobacteria cells/mL. The API answers per point and only reports pixels where
cyanobacteria were detected (lowest reported value is ~6,600 cells/mL).

Area-weighted lake average: each lake is covered by an even grid of points, and each point stands
for the grid cell around it (cell area in km², stored with the point). The lake value is
sum(value × area) / sum(area) over all water points. A water point missing from the lake's newest
weekly image counts as 0 (below detection), so clean water pulls the average down instead of
being skipped.

Two steps:
  --discover   one-time (or when lakes change): sample the grid, keep points CyAN has ever reported
               as water, write lakelens-data/processed/cyan_points.json
  (default)    daily: read each water point's latest weekly value, write lakelens-data/processed/algae.json

Usage:  python lakelens-data/scripts/fetch_algae.py [--discover]
"""
import argparse
import json
import math
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

API = "https://qed.epa.gov/cyanweb/cyan/cyano/location/data"
PROCESSED = Path(__file__).resolve().parents[1] / "processed"
POINTS = PROCESSED / "cyan_points.json"
OUT = PROCESSED / "algae.json"

# Grid box around each lake (lat_min, lat_max, lon_min, lon_max) and spacing in degrees (~1 km = 0.01).
# Texoma is ~10x larger, so it uses a 2 km grid to keep discovery to a few hundred requests.
LAKES = {
    "lew": ("Lewisville", (33.04, 33.24, -97.10, -96.90), 0.01),
    "grp": ("Grapevine", (32.94, 33.04, -97.15, -97.02), 0.01),
    "rr": ("Ray Roberts", (33.30, 33.50, -97.15, -96.95), 0.01),
    "tex": ("Texoma", (33.68, 34.10, -97.05, -96.40), 0.02),
}
# WHO recreational guidance for cyanobacteria (cells/mL): < 20,000 low, 20,000-100,000 moderate, > 100,000 high
LOW_MAX, MEDIUM_MAX = 20_000, 100_000
MAX_AGE_DAYS = 14  # newest image older than this = no current satellite view of the lake


def get(url: str, tries: int = 3):
    for i in range(tries):
        try:
            return json.load(urllib.request.urlopen(url, timeout=60))
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(2 * (i + 1))


def image_date(o) -> datetime:
    return datetime.fromtimestamp(o["imageDateLong"] / 1000, timezone.utc)


def cell_km2(lat: float, step: float) -> float:
    return (step * 111.32) * (step * 111.32 * math.cos(math.radians(lat)))


def discover():
    jobs = []
    for lid, (_, (s, n, w, e), step) in LAKES.items():
        for i in range(round((n - s) / step) + 1):
            for j in range(round((e - w) / step) + 1):
                jobs.append((lid, round(s + i * step, 4), round(w + j * step, 4), step))
    print(f"Probing {len(jobs)} grid points ...")

    def probe(job):
        lid, la, lo, step = job
        d = get(f"{API}/{la}/{lo}/all")
        weekly = [o for o in d.get("outputs") or [] if o["satelliteImageFrequency"] == "Weekly"]
        return job, bool(weekly)

    with ThreadPoolExecutor(6) as pool:
        found = [job for job, water in pool.map(probe, jobs) if water]

    points = {lid: [] for lid in LAKES}
    for lid, la, lo, step in found:
        points[lid].append({"lat": la, "lon": lo, "km2": round(cell_km2(la, step), 3)})
    for lid, pts in points.items():
        print(f"  {LAKES[lid][0]:12} {len(pts)} water points, {sum(p['km2'] for p in pts):.0f} km²")
    POINTS.write_text(json.dumps({
        "source": "EPA CyAN, points with at least one weekly cyanobacteria detection",
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "lakes": points,
    }, indent=1))
    print(f"Wrote {POINTS}")


def category(cells: float) -> str:
    return "low" if cells < LOW_MAX else "medium" if cells <= MEDIUM_MAX else "high"


def daily():
    now = datetime.now(timezone.utc)
    points = json.loads(POINTS.read_text())["lakes"]
    jobs = [(lid, p) for lid, pts in points.items() for p in pts]
    print(f"Reading latest weekly image at {len(jobs)} water points ...")

    def latest(job):
        lid, p = job
        try:
            o = (get(f"{API}/{p['lat']}/{p['lon']}").get("outputs") or [None])[0]
        except Exception as e:
            print(f"  skip {lid} {p['lat']},{p['lon']}: {e}")
            return job, "error"
        return job, o

    with ThreadPoolExecutor(6) as pool:
        results = list(pool.map(latest, jobs))

    lakes = {}
    for lid, (name, _, _) in LAKES.items():
        rows = [(p, o) for (l, p), o in results if l == lid and o != "error"]
        dates = [image_date(o) for _, o in rows if o]
        newest = max(dates, default=None)
        if not rows or newest is None or now - newest > timedelta(days=MAX_AGE_DAYS):
            lakes[lid] = {"name": name, "category": None, "reason": "no satellite image of this lake in the last 14 days",
                          "newest_image": newest.date().isoformat() if newest else None}
            print(f"  {name:12} no current image")
            continue
        # Same weekly image only: a point with nothing in it was below detection
        vals = [(p["km2"], o["cellConcentration"] if o and image_date(o) == newest else 0.0) for p, o in rows]
        area = sum(a for a, _ in vals)
        mean = sum(a * v for a, v in vals) / area
        detected = [a for a, v in vals if v > 0]
        lakes[lid] = {
            "name": name,
            "category": category(mean),
            "mean_cells_ml": round(mean),
            "max_cells_ml": round(max(v for _, v in vals)),
            "image_date": newest.date().isoformat(),
            "points": len(vals),
            "area_km2": round(area),
            "detected_area_pct": round(100 * sum(detected) / area),
        }
        print(f"  {name:12} {lakes[lid]['category']:6} avg {mean:,.0f} cells/mL over {area:.0f} km² "
              f"(algae detected on {lakes[lid]['detected_area_pct']}%), image {newest:%Y-%m-%d}")

    OUT.write_text(json.dumps({
        "source": "EPA CyAN weekly cyanobacteria (Sentinel-3 OLCI, 300 m), area-weighted lake average",
        "generated": now.isoformat(timespec="seconds"),
        "thresholds_cells_ml": {"low_below": LOW_MAX, "high_above": MEDIUM_MAX},
        "lakes": lakes,
    }, indent=2))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--discover", action="store_true", help="rebuild the grid of water points (slow, one-time)")
    (discover if ap.parse_args().discover else daily)()
