"""People living near a reef, from the JRC population grid.

`population_near()` sums the population of every grid cell whose centre lies
within a radius of a survey site. `population_by_group()` does the same for a
set of sites and counts each cell once, so overlapping circles around the
sites of one country do not add the same people twice.

Data: European Commission Joint Research Centre, GHS-POP R2023A, 30 arcsec,
EPSG:4326, epoch 2020. Licence: CC BY 4.0.
Citation: Schiavina, M., Freire, S., Carioli, A. and MacManus, K. (2023).
GHS-POP R2023A - GHS population grid multitemporal (1975-2030). European
Commission, Joint Research Centre. doi:10.2905/2FF68A52-5B5B-4A22-8F40-C41DA8332CFE

Only the GeoTIFF member of each 10-degree tile archive is downloaded, with
HTTP range requests, and each tile is kept in the cache. A run over the reef
sites of the world fetches about 100 MB once.

Three things about this grid are easy to get wrong, and the module handles
them explicitly: the global raster is 43,202 cells wide, not 43,200, so it
overhangs a full turn of longitude by two cells; tile column 36 is 1,202 cells
wide and tile row 18 is 984 cells tall; and a cell counts in full or not at
all, by its centre, so a radius below about 3 km is meaningless at this
resolution. Small islands are under-counted: the grid puts 4 people within
10 km of Heron Island, which has about 100.
"""

from __future__ import annotations

import argparse
import io
import math
import os
import sys
import time
import zipfile
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import rasterio
import rasterio.windows
import requests

# ---------------------------------------------------------------------------
# GHS-POP grid constants.
#
# The values come from the header of the GHS-POP global GeoTIFF, which was read
# with a range request. The global raster is 43202 x 21384 cells. The grid is
# NOT aligned to whole degrees and it is NOT exactly 360 degrees wide:
#
#   left  -180.00791593130032   right  180.008749309419
#   top     89.0995831776456    bottom -89.10041610517223
#
# The width 43202 is 43200 plus 2 cells. The grid therefore overhangs a full
# turn of longitude by 2 cells. Global column 43200 covers the same ground as
# global column 0, and global column 43201 covers the same ground as global
# column 1. The published cell values in that 2-cell strip differ between the
# two ends. See the "Limitations" section of README.md.
#
# The 10 deg x 10 deg tiles partition the global raster exactly. Tile columns
# 1 to 35 are 1200 cells wide. Tile column 36 is 1202 cells wide, because it
# carries the 2 overhang columns. Tile rows 1 to 17 are 1200 cells tall and
# tile row 18 is 984 cells tall.
# ---------------------------------------------------------------------------
PIXEL_DEG = 0.00833333330032682          # nominal 30 arcsec
TILE_PX = 1200                           # nominal cells per tile side
TILE_DEG = TILE_PX * PIXEL_DEG           # 9.999999960392184
GRID_LEFT = -180.00791593130032          # west edge of the global raster
GRID_TOP = 89.0995831776456              # north edge of the global raster
TILE_COLS = 36                           # tile columns around the globe
GLOBAL_COLS = 43202                      # cells across the global raster

EARTH_RADIUS_KM = 6371.0087714           # IUGG mean radius

BASE_URL = (
    "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_POP_GLOBE_R2023A/"
    "GHS_POP_E{epoch}_GLOBE_R2023A_4326_30ss/V1-0/tiles/"
    "GHS_POP_E{epoch}_GLOBE_R2023A_4326_30ss_V1_0_R{row}_C{col}.zip"
)
TILE_STEM = "GHS_POP_E{epoch}_GLOBE_R2023A_4326_30ss_V1_0_R{row}_C{col}"

from . import sources

USER_AGENT = sources.USER_AGENT
CACHE = sources.CACHE / "ghs_pop"
EPOCH = 2020
DEFAULT_RADIUS_KM = 50.0
TILE_MEMORY = 12  # tiles kept in memory; each is about 11 MB


# ---------------------------------------------------------------------------
# Download statistics.
# ---------------------------------------------------------------------------
@dataclass
class Stats:
    tiles_fetched: int = 0
    tiles_from_cache: int = 0
    tiles_absent_new: int = 0
    bytes_downloaded: int = 0
    http_requests: int = 0
    fetched_names: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Range-request file object over a remote zip.
# ---------------------------------------------------------------------------
class RangeReader(io.RawIOBase):
    """A seekable read-only file over an HTTP URL that supports byte ranges.

    The object caches fixed-size blocks. The zip central directory is small,
    so the directory parse costs only a few blocks.
    """

    BLOCK = 8192

    def __init__(self, session: requests.Session, url: str, stats: Stats):
        self.session = session
        self.url = url
        self.stats = stats
        self.pos = 0
        self._blocks: dict[int, bytes] = {}
        head = session.head(url, allow_redirects=True, timeout=60)
        stats.http_requests += 1
        head.raise_for_status()
        if "Content-Length" not in head.headers:
            raise RuntimeError(f"no Content-Length for {url}")
        self.size = int(head.headers["Content-Length"])
        if head.headers.get("Accept-Ranges", "").lower() == "none":
            raise RuntimeError(f"server refuses byte ranges for {url}")

    # -- io.RawIOBase plumbing ------------------------------------------
    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 0:
            self.pos = offset
        elif whence == 1:
            self.pos += offset
        else:
            self.pos = self.size + offset
        self.pos = max(0, min(self.pos, self.size))
        return self.pos

    # -- the actual transfer --------------------------------------------
    def _http_range(self, start: int, end_inclusive: int) -> bytes:
        headers = {"Range": f"bytes={start}-{end_inclusive}"}
        resp = self.session.get(self.url, headers=headers, timeout=120)
        self.stats.http_requests += 1
        if resp.status_code not in (206, 200):
            resp.raise_for_status()
        if resp.status_code == 200 and (start != 0 or end_inclusive != self.size - 1):
            raise RuntimeError(f"server ignored the Range header for {self.url}")
        data = resp.content
        self.stats.bytes_downloaded += len(data)
        return data

    def _block(self, index: int) -> bytes:
        if index not in self._blocks:
            start = index * self.BLOCK
            end = min(start + self.BLOCK, self.size) - 1
            self._blocks[index] = self._http_range(start, end)
        return self._blocks[index]

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0:
            return b""
        out = bytearray()
        remaining = n
        while remaining > 0:
            index = self.pos // self.BLOCK
            offset = self.pos - index * self.BLOCK
            chunk = self._block(index)[offset : offset + remaining]
            if not chunk:
                break
            out += chunk
            self.pos += len(chunk)
            remaining -= len(chunk)
        return bytes(out)

    def readinto(self, buf) -> int:  # type: ignore[override]
        data = self.read(len(buf))
        buf[: len(data)] = data
        return len(data)

    def read_exact(self, start: int, length: int) -> bytes:
        """One single range request. Used for the compressed member payload."""
        if length <= 0:
            return b""
        return self._http_range(start, min(start + length, self.size) - 1)


def extract_tif_from_remote_zip(
    session: requests.Session, url: str, stats: Stats
) -> bytes:
    """Return the decompressed bytes of the single .tif member of a remote zip.

    Only the zip central directory, the member local header and the member
    payload are transferred. The PDF and the XLSX members are not transferred.
    """
    reader = RangeReader(session, url, stats)
    with zipfile.ZipFile(reader) as zf:
        members = [i for i in zf.infolist() if i.filename.lower().endswith(".tif")]
        if len(members) != 1:
            raise RuntimeError(
                f"expected 1 .tif member in {url}, found {len(members)}"
            )
        info = members[0]

    # Parse the local file header to find where the compressed payload starts.
    header = reader.read_exact(info.header_offset, 30)
    if header[:4] != b"PK\x03\x04":
        raise RuntimeError(f"bad local file header in {url}")
    name_len = int.from_bytes(header[26:28], "little")
    extra_len = int.from_bytes(header[28:30], "little")
    data_start = info.header_offset + 30 + name_len + extra_len

    payload = reader.read_exact(data_start, info.compress_size)
    if len(payload) != info.compress_size:
        raise RuntimeError(f"short payload for {info.filename} in {url}")

    if info.compress_type == zipfile.ZIP_STORED:
        raw = payload
    elif info.compress_type == zipfile.ZIP_DEFLATED:
        import zlib

        raw = zlib.decompressobj(-zlib.MAX_WBITS).decompress(payload)
    else:
        raise RuntimeError(f"unsupported zip compression {info.compress_type}")

    if len(raw) != info.file_size:
        raise RuntimeError(
            f"size mismatch for {info.filename}: "
            f"{len(raw)} bytes, expected {info.file_size}"
        )
    return raw


# ---------------------------------------------------------------------------
# Tile index arithmetic.
# ---------------------------------------------------------------------------
def tile_row_index(lat: float) -> int:
    """Continuous tile row index. Row 1 is the northernmost row."""
    return int(math.floor((GRID_TOP - lat) / TILE_DEG)) + 1


def tile_col_continuous(lon: float) -> int:
    """Continuous tile column index. The value may fall outside 1..36."""
    return int(math.floor((lon - GRID_LEFT) / TILE_DEG)) + 1


def wrap_col(col_continuous: int) -> int:
    """Map a continuous column index into the range 1..36."""
    return ((col_continuous - 1) % TILE_COLS) + 1


def lon_shift_for(col_continuous: int) -> float:
    """Whole turns of longitude between the continuous column and the file."""
    return 360.0 * ((col_continuous - 1) // TILE_COLS)


def fetch_tile(
    row: int,
    col: int,
    epoch: int,
    cache_dir: Path,
    session: requests.Session,
    stats: Stats,
) -> Path | None:
    """Return the cached GeoTIFF path, or None when the tile does not exist.

    GHS-POP publishes 348 of the 540 possible tiles. The missing tiles hold no
    land and therefore no population. A missing tile is cached as an `.absent`
    marker file so that the script never asks for the tile twice.
    """
    stem = TILE_STEM.format(epoch=epoch, row=row, col=col)
    tif_path = cache_dir / f"{stem}.tif"
    absent_path = cache_dir / f"{stem}.absent"

    if tif_path.exists() and tif_path.stat().st_size > 0:
        stats.tiles_from_cache += 1
        return tif_path
    if absent_path.exists():
        stats.tiles_from_cache += 1
        return None

    url = BASE_URL.format(epoch=epoch, row=row, col=col)
    try:
        raw = extract_tif_from_remote_zip(session, url, stats)
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else None
        if status in (403, 404):
            absent_path.write_text(f"{url}\nHTTP {status}\n")
            stats.tiles_absent_new += 1
            stats.tiles_fetched += 1
            return None
        raise

    tmp_path = cache_dir / f"{stem}.tif.part"
    tmp_path.write_bytes(raw)
    os.replace(tmp_path, tif_path)
    stats.tiles_fetched += 1
    stats.fetched_names.append(stem)
    return tif_path


# ---------------------------------------------------------------------------
# Tiles in memory.
#
# Thousands of sites read the same few tiles, so a tile is opened once and its
# array is kept. The cache holds the most recent TILE_MEMORY tiles.
# ---------------------------------------------------------------------------
_TILES: "OrderedDict[Path, tuple]" = OrderedDict()


def tile_array(path: Path):
    """(values, left, top, pixel_x, pixel_y, height, width) for a cached tile."""
    if path in _TILES:
        _TILES.move_to_end(path)
        return _TILES[path]
    with rasterio.open(path) as ds:
        values = ds.read(1).astype(np.float64)
        if ds.nodata is not None:
            values[values == ds.nodata] = 0.0
        values[~np.isfinite(values)] = 0.0
        values[values < 0.0] = 0.0
        transform = ds.transform
        entry = (values, transform.c, transform.f, transform.a, -transform.e, ds.height, ds.width)
    _TILES[path] = entry
    if len(_TILES) > TILE_MEMORY:
        _TILES.popitem(last=False)
    return entry


# ---------------------------------------------------------------------------
# Geometry.
# ---------------------------------------------------------------------------
def haversine_km(
    lat0: float, lon0: float, lat_grid: np.ndarray, lon_grid: np.ndarray
) -> np.ndarray:
    """Great-circle distance in km from one point to an array of points."""
    lat0_r = math.radians(lat0)
    lat_r = np.radians(lat_grid)
    # Normalise the longitude difference into -180..180 degrees.
    dlon = np.remainder(lon_grid - lon0 + 180.0, 360.0) - 180.0
    dlon_r = np.radians(dlon)
    dlat_r = lat_r - lat0_r
    a = (
        np.sin(dlat_r * 0.5) ** 2
        + math.cos(lat0_r) * np.cos(lat_r) * np.sin(dlon_r * 0.5) ** 2
    )
    a = np.clip(a, 0.0, 1.0)
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def search_box(lat: float, lon: float, radius_km: float):
    """Return (lat_min, lat_max, lon_min, lon_max, full_circle).

    The longitude half-width is the exact spherical-cap value, not radius
    divided by the cosine of the latitude. When the cap touches a pole, the
    function reports a full circle of longitude.
    """
    ang = radius_km / EARTH_RADIUS_KM  # angular radius in radians
    dlat = math.degrees(ang)
    lat_min = lat - dlat
    lat_max = lat + dlat

    lat_r = math.radians(lat)
    denom = math.cos(lat_r)
    full_circle = False
    if lat_max >= 90.0 or lat_min <= -90.0 or denom <= 1e-12:
        full_circle = True
    else:
        ratio = math.sin(ang) / denom
        if ratio >= 1.0:
            full_circle = True
        else:
            dlon = math.degrees(math.asin(ratio))

    lat_min = max(lat_min, -90.0)
    lat_max = min(lat_max, 90.0)
    if full_circle:
        return lat_min, lat_max, lon - 180.0, lon + 180.0, True
    return lat_min, lat_max, lon - dlon, lon + dlon, False


# ---------------------------------------------------------------------------
# Per-site population.
# ---------------------------------------------------------------------------
def site_cells(
    lat: float,
    lon: float,
    radius_km: float,
    epoch: int,
    cache_dir: Path,
    session: requests.Session,
    stats: Stats,
    verbose: bool = False,
):
    """Return (keys, distances_km, values) for the cells within the radius.

    A key is the cell's position in the global raster, so two tiles that hold
    the same cell give the same key and the cell is counted once.
    """
    r_max = radius_km
    lat_min, lat_max, lon_min, lon_max, full_circle = search_box(lat, lon, r_max)

    row_hi = tile_row_index(lat_max)   # smallest row number, north
    row_lo = tile_row_index(lat_min)   # largest row number, south
    rows = range(min(row_hi, row_lo), max(row_hi, row_lo) + 1)

    if full_circle:
        cols_continuous = list(range(1, TILE_COLS + 1))
    else:
        c0 = tile_col_continuous(lon_min)
        c1 = tile_col_continuous(lon_max)
        if c1 - c0 + 1 >= TILE_COLS:
            cols_continuous = list(range(1, TILE_COLS + 1))
        else:
            cols_continuous = list(range(c0, c1 + 1))

    keys: list[np.ndarray] = []
    dists: list[np.ndarray] = []
    vals: list[np.ndarray] = []
    tiles_used = 0

    for row in rows:
        if row < 1:
            continue
        for col_cont in cols_continuous:
            col = wrap_col(col_cont)
            shift = lon_shift_for(col_cont)
            path = fetch_tile(row, col, epoch, cache_dir, session, stats)
            if path is None:
                continue

            block_values, left, top, px, py, height, width = tile_array(path)

            # Where this tile starts inside the global raster, in cells. The
            # value must be a whole number of cells: a tile that does not line
            # up would break the cell identity, so the script stops instead of
            # guessing.
            col_off_f = (left - GRID_LEFT) / px
            row_off_f = (GRID_TOP - top) / py
            col_off = int(round(col_off_f))
            row_off = int(round(row_off_f))
            if abs(col_off_f - col_off) > 0.01 or abs(row_off_f - row_off) > 0.01:
                raise RuntimeError(
                    f"tile R{row}_C{col} does not line up with the GHS-POP global grid: "
                    f"column offset {col_off_f}, row offset {row_off_f}"
                )

            # Map the search box into this tile's own longitude frame.
            box_lon_min = lon_min - shift
            box_lon_max = lon_max - shift

            j0 = max(0, int(math.floor((box_lon_min - left) / px)) - 1)
            j1 = min(width, int(math.ceil((box_lon_max - left) / px)) + 1)
            i0 = max(0, int(math.floor((top - lat_max) / py)) - 1)
            i1 = min(height, int(math.ceil((top - lat_min) / py)) + 1)
            if j1 <= j0 or i1 <= i0:
                continue
            block = block_values[i0:i1, j0:j1].astype(np.float64)

            jj = np.arange(j0, j1)
            ii = np.arange(i0, i1)
            cell_lon = left + (jj + 0.5) * px
            cell_lat = top - (ii + 0.5) * py

            lon_2d = np.broadcast_to(cell_lon[None, :], block.shape)
            lat_2d = np.broadcast_to(cell_lat[:, None], block.shape)
            dist = haversine_km(lat, lon, lat_2d, lon_2d)

            sel = dist <= r_max
            if not sel.any():
                continue
            tiles_used += 1

            sel_i = np.broadcast_to(ii[:, None], block.shape)[sel]
            sel_j = np.broadcast_to(jj[None, :], block.shape)[sel]
            # Identify every cell by its position in the global raster.
            # Two tiles that hold the same cell give the same key, so the
            # script adds that cell once.
            g_row = row_off + sel_i.astype(np.int64)
            g_col = col_off + sel_j.astype(np.int64)
            keys.append(g_row * GLOBAL_COLS + g_col)
            dists.append(dist[sel])
            vals.append(block[sel])

    if not keys:
        empty = np.zeros(0)
        return empty.astype(np.int64), empty, empty

    all_keys = np.concatenate(keys)
    all_dist = np.concatenate(dists)
    all_vals = np.concatenate(vals)

    # Count every grid cell once, even when two tiles cover the same cell.
    unique_keys, first = np.unique(all_keys, return_index=True)
    if verbose:
        print(f"    tiles used {tiles_used}, unique cells {unique_keys.size}", file=sys.stderr)
    return unique_keys, all_dist[first], all_vals[first]


def population_for_site(lat: float, lon: float, radii_km: Sequence[float], epoch: int,
                        cache_dir: Path, session, stats: Stats, verbose: bool = False):
    """Population within each radius of one site, largest radius read once."""
    keys, distances, values = site_cells(lat, lon, max(radii_km), epoch, cache_dir,
                                         session, stats, verbose)
    pops = [float(values[distances <= r].sum()) for r in radii_km]
    return pops, int(keys.size), int(keys.size)


# ---------------------------------------------------------------------------
# Atlas entry points
# ---------------------------------------------------------------------------
def _session() -> "requests.Session":
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    return session


def population_near(lat: float, lon: float, radius_km: float = DEFAULT_RADIUS_KM,
                    cache: Optional[Path] = None, epoch: int = EPOCH) -> float:
    """People living within `radius_km` of one point."""
    cache = cache or CACHE
    cache.mkdir(parents=True, exist_ok=True)
    pops, _, _ = population_for_site(lat, ((lon + 180.0) % 360.0) - 180.0, [radius_km],
                                     epoch, cache, _session(), Stats())
    return pops[0]


def _cells_near(lat: float, lon: float, radius_km: float, epoch: int, cache: Path,
                session, stats: Stats) -> Dict[int, float]:
    """Global cell key -> population, for the cells inside the radius."""
    keys, _distances, values = site_cells(lat, lon, radius_km, epoch, cache, session, stats)
    return dict(zip(keys.tolist(), values.tolist()))


def population_by_group(sites, group: str, radius_km: float = DEFAULT_RADIUS_KM,
                        cache: Optional[Path] = None, epoch: int = EPOCH,
                        progress: bool = False):
    """People within `radius_km` of any site of each group, counting each cell once.

    `sites` needs the columns `group`, `lat` and `lon`. Returns a DataFrame with
    the group, the number of sites, the population, and the number of grid cells
    it came from.
    """
    cache = cache or CACHE
    cache.mkdir(parents=True, exist_ok=True)
    session, stats = _session(), Stats()
    rows = []
    for name, part in sites.groupby(group):
        keys: List[np.ndarray] = []
        values: List[np.ndarray] = []
        points = part[["lat", "lon"]].drop_duplicates()
        for lat, lon in points.itertuples(index=False):
            k, _d, v = site_cells(float(lat), ((float(lon) + 180.0) % 360.0) - 180.0,
                                  radius_km, epoch, cache, session, stats)
            if k.size:
                keys.append(k)
                values.append(v)
        if keys:
            all_keys = np.concatenate(keys)
            all_values = np.concatenate(values)
            unique, first = np.unique(all_keys, return_index=True)
            total, cells = float(all_values[first].sum()), int(unique.size)
        else:
            total, cells = 0.0, 0
        rows.append({group: name, "sites": int(len(points)), "population": total, "cells": cells})
        if progress:
            print(f"  {name}: {len(points)} sites, {total:,.0f} people", file=sys.stderr)
    frame = pd.DataFrame(rows)
    frame.attrs["bytes_downloaded"] = stats.bytes_downloaded
    frame.attrs["tiles_fetched"] = stats.tiles_fetched
    return frame
