"""Fetch real-time lightning near each lake from GOES-19 GLM.

Source: NOAA GOES-19 Geostationary Lightning Mapper, Level 2 "LCFA" product
(the real-time feed of https://www.ncei.noaa.gov/products/goes-terrestrial-weather-abi-glm),
published to the NOAA Open Data bucket s3://noaa-goes19 as a new file every 20 s.

Writes lakelens-data/processed/lightning.json, which lakelens.html reads.

Usage:  python lakelens-data/scripts/fetch_lightning.py [--minutes 30]
"""
import argparse
import json
import math
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path

import h5py
import numpy as np

BUCKET = "https://noaa-goes19.s3.amazonaws.com"
PRODUCT = "GLM-L2-LCFA"
OUT = Path(__file__).resolve().parents[1] / "processed" / "lightning.json"

# Lake center points (same ids as lakelens.html)
LAKES = {
    "lew": ("Lewisville", 33.07, -97.0),
    "grp": ("Grapevine", 32.97, -97.08),
    "rr": ("Ray Roberts", 33.37, -97.05),
    "lav": ("Lavon", 33.05, -96.48),
    "jp": ("Joe Pool", 32.62, -97.0),
    "ben": ("Benbrook", 32.63, -97.45),
    "tex": ("Texoma", 33.82, -96.6),
    "rh": ("Ray Hubbard", 32.85, -96.52),
    "em": ("Eagle Mountain", 32.9, -97.47),
    "lw": ("Lake Worth", 32.8, -97.42),
}
# Only keep flashes inside this box (North Texas + margin) before computing distances
BOX = (31.5, 35.5, -99.5, -94.5)  # lat_min, lat_max, lon_min, lon_max
NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
START_RE = re.compile(r"_s(\d{4})(\d{3})(\d{2})(\d{2})(\d{2})")


def list_keys(hour: datetime) -> list[str]:
    prefix = f"{PRODUCT}/{hour:%Y}/{hour:%j}/{hour:%H}/"
    keys, token = [], None
    while True:
        url = f"{BUCKET}/?list-type=2&prefix={prefix}"
        if token:
            url += "&continuation-token=" + urllib.parse.quote(token)
        root = ET.fromstring(urllib.request.urlopen(url, timeout=30).read())
        keys += [k.text for k in root.findall("s3:Contents/s3:Key", NS)]
        token = root.findtext("s3:NextContinuationToken", namespaces=NS)
        if not token:
            return keys


def file_start(key: str) -> datetime:
    y, doy, hh, mm, ss = map(int, START_RE.search(key).groups())
    return datetime(y, 1, 1, hh, mm, ss, tzinfo=timezone.utc) + timedelta(days=doy - 1)


def decode(ds) -> np.ndarray:
    """Apply netCDF packing (unsigned ints, scale_factor, add_offset)."""
    v = ds[()]
    if ds.attrs.get("_Unsigned", b"false") in (b"true", "true"):
        v = v.view(v.dtype.str.replace("i", "u"))
    v = v.astype("f8")
    v *= float(ds.attrs.get("scale_factor", [1])[0])
    v += float(ds.attrs.get("add_offset", [0])[0])
    return v


def read_flashes(key: str):
    data = urllib.request.urlopen(f"{BUCKET}/{key}", timeout=60).read()
    with h5py.File(BytesIO(data)) as f:
        lat, lon = f["flash_lat"][()], f["flash_lon"][()]
        good = f["flash_quality_flag"][()] == 0
        t = decode(f["flash_time_offset_of_first_event"])
        units = f["flash_time_offset_of_first_event"].attrs["units"].decode()
    base = datetime.fromisoformat(units.removeprefix("seconds since ").replace(" ", "T")).replace(tzinfo=timezone.utc)
    m = good & (lat >= BOX[0]) & (lat <= BOX[1]) & (lon >= BOX[2]) & (lon <= BOX[3])
    return [(float(a), float(o), base + timedelta(seconds=float(s))) for a, o, s in zip(lat[m], lon[m], t[m])]


def miles(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 3958.8 * 2 * math.asin(math.sqrt(a))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=int, default=30, help="look-back window (NWS 30-minute rule)")
    args = ap.parse_args()

    now = datetime.now(timezone.utc)
    since = now - timedelta(minutes=args.minutes)
    hours = {since.replace(minute=0, second=0, microsecond=0) + timedelta(hours=i)
             for i in range(int((now - since).total_seconds() // 3600) + 2)}
    keys = [k for h in sorted(hours) if h <= now for k in list_keys(h)]
    keys = [k for k in keys if file_start(k) >= since - timedelta(seconds=20)]
    print(f"Downloading {len(keys)} GLM files since {since:%H:%M} UTC ...")

    with ThreadPoolExecutor(8) as pool:
        flashes = [fl for batch in pool.map(read_flashes, keys) for fl in batch]
    flashes = [fl for fl in flashes if fl[2] >= since]
    print(f"{len(flashes)} flashes over North Texas in the last {args.minutes} min")

    lakes = {}
    for lid, (name, la, lo) in LAKES.items():
        near = [(miles(la, lo, a, o), t) for a, o, t in flashes]
        close = min(near, default=None)
        lakes[lid] = {
            "name": name,
            "nearest_mi": round(close[0], 1) if close else None,
            "nearest_min_ago": round((now - close[1]).total_seconds() / 60, 1) if close else None,
            "last_within_10mi_min_ago": min((round((now - t).total_seconds() / 60, 1) for d, t in near if d <= 10), default=None),
            "count_10mi": sum(d <= 10 for d, _ in near),
            "count_20mi": sum(d <= 20 for d, _ in near),
        }
        print(f"  {name:15} nearest {lakes[lid]['nearest_mi']} mi, {lakes[lid]['count_20mi']} flashes within 20 mi")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "source": f"NOAA GOES-19 GLM {PRODUCT} (s3://noaa-goes19)",
        "generated": now.isoformat(timespec="seconds"),
        "window_min": args.minutes,
        "files": len(keys),
        "latest_file_start": file_start(keys[-1]).isoformat() if keys else None,
        "lakes": lakes,
    }, indent=2))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
