"""Heat stress at reefs from NOAA Coral Reef Watch virtual stations.

A virtual station is a reef area that Coral Reef Watch reports on every day.
Each station has a daily record from 1985 to the present that includes the
degree heating weeks (DHW): the heat dose a reef has taken over the last twelve
weeks, in Celsius-weeks above its maximum monthly mean temperature. Four
Celsius-weeks is the level at which bleaching is expected, and eight is the
level at which mortality is expected.

The atlas uses the stations for two things the survey records cannot give:

- the heat dose in years and places where nobody surveyed, so the risk of a
  region is not limited to the years it was visited;
- the trend in the annual maximum heat dose since 1985, which sets the size of
  the warming step in the scenarios.

`annual_dhw()` returns one row per station and year. The daily files total
about 270 MB, so the first call takes a few minutes; the yearly summary is
cached as a small CSV afterwards.
"""
from __future__ import annotations

import gzip
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

from . import sources

STATION_DATA = "https://coralreefwatch.noaa.gov/product/vs/data/{slug}.txt"
DAILY_DIR = sources.CACHE / "crw_daily"
ANNUAL_CSV = sources.CACHE / "crw_annual_dhw.csv"
COLUMNS = ["year", "month", "day", "sst_min", "sst_max", "sst_hs90", "ssta_hs90",
           "hs", "dhw", "baa_max"]


def stations(*, offline: bool = False) -> pd.DataFrame:
    """Virtual stations: slug, name, latitude, longitude."""
    doc = json.loads(sources.fetch("crw_stations", offline=offline).read_text())
    rows = {}
    for feature in doc["features"]:
        props = feature["properties"]
        page = props.get("gauge_page") or ""
        if not page.endswith(".php") or feature["geometry"]["type"] != "Point":
            continue
        lon, lat = feature["geometry"]["coordinates"][:2]
        rows.setdefault(page[:-4], {"slug": page[:-4], "name": props.get("name", ""),
                                    "lat": float(lat), "lon": float(lon)})
    return pd.DataFrame(sorted(rows.values(), key=lambda r: r["slug"]))


def _download(slug: str) -> Path:
    path = DAILY_DIR / f"{slug}.txt.gz"
    if path.exists():
        return path
    DAILY_DIR.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        STATION_DATA.format(slug=slug),
        headers={"User-Agent": sources.USER_AGENT, "Accept-Encoding": "gzip"},
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        payload = response.read()
        if response.headers.get("Content-Encoding") == "gzip":
            payload = gzip.decompress(payload)
    part = path.with_suffix(".part")
    with gzip.open(part, "wb") as out:
        out.write(payload)
    part.replace(path)
    return path


def daily(slug: str) -> pd.DataFrame:
    """The daily record of one station."""
    with gzip.open(_download(slug), "rt") as handle:
        lines = [line for line in handle if line[:4].isdigit() and len(line.split()) == len(COLUMNS)]
    frame = pd.DataFrame([line.split() for line in lines], columns=COLUMNS).astype(float)
    return frame[frame["dhw"] > -90]


def annual_dhw(slugs: Optional[Iterable[str]] = None, *, workers: int = 8,
               refresh: bool = False) -> pd.DataFrame:
    """One row per station and year: the maximum and mean DHW of that year."""
    if ANNUAL_CSV.exists() and not refresh:
        return pd.read_csv(ANNUAL_CSV)
    wanted: List[str] = list(slugs) if slugs is not None else stations()["slug"].tolist()
    rows = []

    def one(slug: str) -> pd.DataFrame:
        frame = daily(slug)
        out = frame.groupby("year")["dhw"].agg(dhw_max="max", dhw_mean="mean").reset_index()
        out.insert(0, "slug", slug)
        return out

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(one, wanted):
            rows.append(result)
    annual = pd.concat(rows, ignore_index=True)
    annual = annual[annual["year"] >= 1985]
    ANNUAL_CSV.parent.mkdir(parents=True, exist_ok=True)
    annual.to_csv(ANNUAL_CSV, index=False)
    return annual


def trends(annual: pd.DataFrame, first_year: int = 1985, last_year: int = 2025) -> pd.DataFrame:
    """Per station: the trend in the annual maximum DHW, and its recent level.

    ``dhw_per_decade`` is the slope of a straight line through the annual
    maxima. ``dhw_recent`` is the mean of the last ten complete years, and
    ``dhw_2050`` carries the trend forward from the middle of that decade.
    """
    frame = annual[(annual["year"] >= first_year) & (annual["year"] <= last_year)]
    rows = []
    for slug, group in frame.groupby("slug"):
        if len(group) < 20:
            continue
        slope, _ = np.polyfit(group["year"], group["dhw_max"], 1)
        recent = group[group["year"] >= last_year - 10]
        mid = float(recent["year"].mean())
        level = float(recent["dhw_max"].mean())
        rows.append({"slug": slug, "years": len(group), "dhw_per_decade": slope * 10,
                     "dhw_recent": level, "dhw_recent_mid_year": mid,
                     "dhw_2050": max(0.0, level + slope * (2050 - mid))})
    return pd.DataFrame(rows)


def nearest_station(lat: np.ndarray, lon: np.ndarray, station_table: pd.DataFrame) -> pd.DataFrame:
    """For each point, the nearest station by great-circle distance."""
    lat1 = np.radians(np.asarray(lat, dtype=float))[:, None]
    lon1 = np.radians(np.asarray(lon, dtype=float))[:, None]
    lat2 = np.radians(station_table["lat"].to_numpy())[None, :]
    lon2 = np.radians(station_table["lon"].to_numpy())[None, :]
    d = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    distance_km = 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(d, 0, 1)))
    pick = np.argmin(distance_km, axis=1)
    return pd.DataFrame({"slug": station_table["slug"].to_numpy()[pick],
                         "station_name": station_table["name"].to_numpy()[pick],
                         "station_km": distance_km[np.arange(len(pick)), pick]})
