"""Public datasets the atlas uses, with their licences, and a download cache.

Every dataset here is free to download without an account and carries a licence
that allows a derived map to be published with attribution. Files land in
``data/external/`` (git-ignored) and are only fetched once.

Datasets that were considered and left out, so the choice is on the record:

- UNEP-WCMC "Global Distribution of Warm-Water Coral Reefs" (WCMC-008): the
  best reef-extent polygons, but the licence forbids commercial use and
  redistribution.
- Sea Around Us reef fisheries catch: CC BY-NC 4.0, non-commercial only.
- Allen Coral Atlas: CC BY 4.0, but a global download needs an account.
"""
from __future__ import annotations

import hashlib
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

ROOT = Path(__file__).resolve().parent.parent.parent
CACHE = ROOT / "data" / "external"
USER_AGENT = "coralforest-atlas/1.0 (research; https://github.com/MikeHLee/Coral.Forest)"


@dataclass(frozen=True)
class Source:
    key: str
    url: str
    filename: str
    licence: str
    licence_url: str
    citation: str
    note: str = ""

    @property
    def path(self) -> Path:
        return CACHE / self.filename


SOURCES: Dict[str, Source] = {s.key: s for s in [
    Source(
        key="bleaching",
        url=("https://datadocs.bco-dmo.org/dataset/773466/file/B11vA82u7y2Owp/"
             "global_bleaching_environmental.csv"),
        filename="global_bleaching_environmental.csv",
        licence="CC BY 4.0",
        licence_url="https://www.bco-dmo.org/dataset/773466",
        citation=("van Woesik, R. and Kratochwill, C. (2022). A global coral-bleaching database, "
                  "1980-2020. Scientific Data 9:20. doi:10.1038/s41597-022-01121-y. "
                  "Data: BCO-DMO dataset 773466, doi:10.26008/1912/bco-dmo.773466.2"),
        note="One row per substrate; sample-level rows repeat. Temperature columns are in kelvin.",
    ),
    Source(
        key="crw_stations",
        url="https://coralreefwatch.noaa.gov/product/vs/vs_polygons.json",
        filename="crw_vs_polygons.json",
        licence="US Government work, public domain",
        licence_url="https://coralreefwatch.noaa.gov/product/vs/data.php",
        citation=("NOAA Coral Reef Watch (2019, updated daily). Daily Global 5km Satellite Coral "
                  "Bleaching Heat Stress Monitoring Product Suite Version 3.1, virtual stations."),
        note="428 station points with today's SST, SSTA, DHW and alert level.",
    ),
    Source(
        key="land",
        url=("https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/"
             "geojson/ne_110m_land.geojson"),
        filename="ne_110m_land.geojson",
        licence="Public domain",
        licence_url="https://www.naturalearthdata.com/about/terms-of-use/",
        citation="Natural Earth, 1:110m physical land polygons (nvkelso/natural-earth-vector).",
    ),
]}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(key: str, *, offline: bool = False) -> Path:
    """The local path of a dataset, downloading it on first use."""
    source = SOURCES[key]
    if source.path.exists():
        return source.path
    if offline:
        raise FileNotFoundError(f"{source.filename} is not in {CACHE} and offline was requested")
    CACHE.mkdir(parents=True, exist_ok=True)
    part = source.path.with_suffix(source.path.suffix + ".part")
    request = urllib.request.Request(source.url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=300) as response, part.open("wb") as out:
        shutil.copyfileobj(response, out)
    part.replace(source.path)
    return source.path


def attribution() -> str:
    """One line per dataset, for a figure caption or a page footer."""
    return "\n".join(f"{s.citation} Licence: {s.licence} ({s.licence_url})." for s in SOURCES.values())
