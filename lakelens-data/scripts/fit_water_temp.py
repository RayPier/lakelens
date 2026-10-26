"""Fit the water-temperature estimate used by lakelens.html (WT_FIT).

None of the four lakes has a live water-temperature sensor, so the page estimates surface temperature
from air temperature:

    water °C = a + b·EMA(air) + s·sin(2π·doy/365.25) + c·cos(2π·doy/365.25)

EMA(air) is an exponentially weighted mean of daily mean air temperature over the last 60 days with a
14-day time constant (lakes lag the weather by weeks); the sin/cos terms capture that lakes are cooler than
the air suggests in spring and warmer in fall.

Training data: USGS 15-min water temperature (parameter 00010) at Lewisville (USGS-330404096585000) and
Ray Roberts (USGS-332138097024101), averaged per day, 2010-2025 where available; air temperature from the
Open-Meteo historical archive at each lake point. Accuracy is measured leaving one year out at a time.

Usage:  python lakelens-data/scripts/fit_water_temp.py
Then copy the printed coefficients into WT_FIT in lakelens.html.
"""
import collections
import json
import math
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import date

import numpy as np

SITES = {"lew": ("USGS-330404096585000", 33.07, -97.0), "rr": ("USGS-332138097024101", 33.37, -97.05)}
YEARS = range(2010, 2026)
TAU, WINDOW = 14, 60  # days


def get(url):
    for i in range(3):
        try:
            return json.load(urllib.request.urlopen(url, timeout=180))
        except Exception:
            if i == 2:
                raise


def water_year(site, yr):
    d = get("https://api.waterdata.usgs.gov/ogcapi/v0/collections/continuous/items?f=json"
            f"&monitoring_location_id={site}&parameter_code=00010"
            f"&datetime={yr}-01-01T00:00:00Z/{yr}-12-31T23:59:59Z&limit=50000&properties=time,value")
    by = collections.defaultdict(list)
    for f in d.get("features", []):
        try:
            by[f["properties"]["time"][:10]].append(float(f["properties"]["value"]))
        except (TypeError, ValueError):
            pass
    return {k: sum(v) / len(v) for k, v in by.items() if len(v) >= 48}  # at least half a day of readings


def air(lat, lon):
    d = get(f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}"
            f"&start_date={YEARS[0] - 1}-10-01&end_date={YEARS[-1]}-12-31&daily=temperature_2m_mean&timezone=America/Chicago")
    return d["daily"]["time"], np.array([np.nan if v is None else v for v in d["daily"]["temperature_2m_mean"]])


def main():
    rows = []  # (year, ema_air, sin, cos, water)
    k = np.arange(WINDOW)[::-1]
    weights = np.exp(-k / TAU)
    for lid, (site, lat, lon) in SITES.items():
        with ThreadPoolExecutor(6) as pool:
            water = {d: w for part in pool.map(lambda y: water_year(site, y), YEARS) for d, w in part.items()}
        days, a = air(lat, lon)
        idx = {d: i for i, d in enumerate(days)}
        for d, w in water.items():
            i = idx.get(d)
            if i is None or i < WINDOW:
                continue
            seg = a[i - WINDOW + 1:i + 1]
            ok = ~np.isnan(seg)
            ang = 2 * math.pi * date.fromisoformat(d).timetuple().tm_yday / 365.25
            rows.append((int(d[:4]), np.sum(seg[ok] * weights[ok]) / np.sum(weights[ok]), math.sin(ang), math.cos(ang), w))
        print(f"{lid}: {len(water)} days of water temperature")

    r = np.array(rows)
    yrs, y = r[:, 0], r[:, 4]
    X = np.column_stack([np.ones(len(r)), r[:, 1], r[:, 2], r[:, 3]])
    err = np.empty(len(r))
    for yr in np.unique(yrs):
        tr, te = yrs != yr, yrs == yr
        c, *_ = np.linalg.lstsq(X[tr], y[tr], rcond=None)
        err[te] = X[te] @ c - y[te]
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)

    print(f"{len(r)} lake-days, years {sorted(int(v) for v in np.unique(yrs))}")
    print(f"Held-out-year error: RMSE {math.sqrt(np.mean(err ** 2)):.2f} °C, MAE {np.mean(np.abs(err)):.2f} °C, "
          f"90th pct {np.percentile(np.abs(err), 90):.2f} °C")
    a_, b_, s_, c_ = (round(float(v), 4) for v in coef)
    print(f"WT_FIT={{a:{a_},b:{b_},s:{s_},c:{c_},tau:{TAU},days:{WINDOW}}}")


if __name__ == "__main__":
    main()
