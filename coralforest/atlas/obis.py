"""Reef-building coral species recorded near a reef, from OBIS.

The Ocean Biodiversity Information System pools occurrence records from many
datasets, each under its own licence. This module counts species only from
datasets whose stated rights allow reuse with attribution: CC0, or CC BY
without a non-commercial or no-derivatives clause. Every count says how many
datasets it used and how many it left out, so the filter is visible.

A count is "hard-coral species (Scleractinia) recorded inside this box", not
"species that live on this reef". Recording effort is uneven, so a box with
more survey history holds more species for that reason alone.
"""
from __future__ import annotations

import hashlib
import json
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from . import sources

API = "https://api.obis.org/v3"
CACHE = sources.CACHE / "obis"
SCLERACTINIA = 1363  # OBIS taxon id for the stony corals
OPEN_TOKENS = ("cc0", "creative commons attribution", "cc-by", "cc by", "publicdomain", "zero")
CLOSED_TOKENS = ("noncommercial", "non-commercial", "nc ", "cc-by-nc", "cc by-nc",
                 "noderiv", "no derivative", "-nd", " nd ")


def _get(path: str, params: Dict[str, object]) -> dict:
    query = urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
    url = f"{API}/{path}?{query}"
    CACHE.mkdir(parents=True, exist_ok=True)
    cached = CACHE / f"{hashlib.sha256(url.encode()).hexdigest()[:24]}.json"
    if cached.exists():
        return json.loads(cached.read_text())
    request = urllib.request.Request(url, headers={"User-Agent": sources.USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = json.loads(response.read().decode())
    cached.write_text(json.dumps(payload))
    return payload


def box(lat_min: float, lat_max: float, lon_min: float, lon_max: float, pad: float = 0.25) -> str:
    """A well-known-text rectangle, padded, for the OBIS geometry parameter."""
    lat0, lat1 = max(-90.0, lat_min - pad), min(90.0, lat_max + pad)
    lon0, lon1 = max(-180.0, lon_min - pad), min(180.0, lon_max + pad)
    corners = [(lon0, lat0), (lon1, lat0), (lon1, lat1), (lon0, lat1), (lon0, lat0)]
    return "POLYGON((" + ",".join(f"{x:.4f} {y:.4f}" for x, y in corners) + "))"


def is_open(rights: Optional[str]) -> bool:
    text = (rights or "").lower()
    if any(token in text for token in CLOSED_TOKENS):
        return False
    return any(token in text for token in OPEN_TOKENS)


def open_datasets(geometry: str, taxonid: int = SCLERACTINIA) -> Tuple[List[str], int]:
    """Ids of the openly licensed datasets in the box, and how many were left out."""
    payload = _get("dataset", {"geometry": geometry, "taxonid": taxonid, "size": 500})
    results = payload.get("results", [])
    keep = [row["id"] for row in results if is_open(row.get("intellectualrights"))]
    return keep, len(results) - len(keep)


def richness(geometry: str, taxonid: int = SCLERACTINIA) -> Dict[str, object]:
    """Species and record counts in the box, from openly licensed datasets only."""
    keep, dropped = open_datasets(geometry, taxonid)
    if not keep:
        return {"species": 0, "records": 0, "datasets_used": 0, "datasets_dropped": dropped}
    stats = _get("statistics", {"geometry": geometry, "taxonid": taxonid,
                                "datasetid": ",".join(keep)})
    return {"species": int(stats.get("species") or 0), "records": int(stats.get("records") or 0),
            "datasets_used": len(keep), "datasets_dropped": dropped}


def species_list(geometry: str, taxonid: int = SCLERACTINIA) -> Dict[str, object]:
    """The species inside the box, from openly licensed datasets only.

    Counting species inside one box per country is wrong for a country whose
    territories are scattered: the box around France would span the Atlantic
    and the Pacific. Work region by region, where a box is a reasonable
    outline, and take the union of the lists for a country.
    """
    keep, dropped = open_datasets(geometry, taxonid)
    if not keep:
        return {"species": set(), "records": 0, "datasets_used": 0, "datasets_dropped": dropped}
    payload = _get("checklist", {"geometry": geometry, "taxonid": taxonid,
                                 "datasetid": ",".join(keep), "size": 10000})
    rows = payload.get("results", [])
    species = {int(row["taxonID"]) for row in rows
               if row.get("taxonRank") == "Species" and row.get("taxonID")}
    records = sum(int(row.get("records") or 0) for row in rows
                  if row.get("taxonRank") == "Species")
    return {"species": species, "records": records, "datasets_used": len(keep),
            "datasets_dropped": dropped}


def richness_for_boxes(boxes: Sequence[Tuple[str, str]], taxonid: int = SCLERACTINIA) -> List[Dict[str, object]]:
    """`boxes` is a sequence of (name, geometry); one row per name."""
    rows = []
    for name, geometry in boxes:
        row = {"name": name}
        row.update(richness(geometry, taxonid))
        rows.append(row)
    return rows


def boxes_from_points(frame, group: str, lat: str = "lat", lon: str = "lon",
                      pad: float = 0.25) -> List[Tuple[str, str]]:
    """A padded bounding box per group of points."""
    bounds = frame.groupby(group).agg(lat_min=(lat, "min"), lat_max=(lat, "max"),
                                      lon_min=(lon, "min"), lon_max=(lon, "max"))
    out = []
    for name, row in bounds.iterrows():
        if row["lon_max"] - row["lon_min"] > 180:  # a group that straddles the date line
            continue
        out.append((str(name), box(row["lat_min"], row["lat_max"], row["lon_min"], row["lon_max"], pad)))
    return out
