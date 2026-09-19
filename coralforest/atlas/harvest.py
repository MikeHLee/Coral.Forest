"""Refresh the two community layers that need many requests.

    python -m coralforest.atlas.harvest --species --population

Both write into `data/community/`, which is committed, so an ordinary build
reads them from the repository and touches neither service:

- `reef_species.csv`: hard-coral species recorded near each country's survey
  sites, from OBIS, counting only datasets whose licence allows reuse.
- `population_near_reefs.csv`: people living within 50 km of any survey site of
  the country, from the JRC population grid, each grid cell counted once.

Per-region versions go to `data/external/`, which is not committed; the build
picks them up for the map when they are there.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import sys

from . import bleaching, obis, sources

COMMUNITY = Path(__file__).resolve().parent.parent.parent / "data" / "community"


def site_points(offline: bool = False) -> pd.DataFrame:
    frame = bleaching.samples(offline=offline)
    return frame.dropna(subset=["lat", "lon"])[["country", "ecoregion", "lat", "lon"]].drop_duplicates()


def species(points: pd.DataFrame) -> tuple:
    """Species lists per region, and the union per country.

    One box per region, because a box around a country with scattered
    territories would cover oceans it has no reef in.
    """
    regions = points.dropna(subset=["ecoregion"])
    boxes = dict(obis.boxes_from_points(regions, "ecoregion"))
    per_region, sets = [], {}
    for name, geometry in boxes.items():
        result = obis.species_list(geometry)
        sets[name] = result["species"]
        per_region.append({"ecoregion": name, "habitat": len(result["species"]),
                           "records": result["records"], "datasets_used": result["datasets_used"],
                           "datasets_dropped": result["datasets_dropped"]})
        print(f"  {name}: {len(result['species'])} species "
              f"({result['datasets_used']} datasets, {result['datasets_dropped']} left out)",
              file=sys.stderr)
    by_region = pd.DataFrame(per_region)
    rows = []
    for country, part in regions.groupby("country"):
        names = [n for n in part["ecoregion"].unique() if n in sets]
        union = set().union(*[sets[n] for n in names]) if names else set()
        rows.append({"country": country, "habitat": len(union), "regions": len(names)})
    return pd.DataFrame(rows), by_region


def population(points: pd.DataFrame, level: str, radius_km: float) -> pd.DataFrame:
    from . import population as pop  # imported here: it needs rasterio

    frame = pop.population_by_group(points.dropna(subset=[level]), level,
                                    radius_km=radius_km, progress=True)
    return frame.rename(columns={"population": f"population_{int(radius_km)}km"})


def global_total(points: pd.DataFrame, radius_km: float) -> dict:
    """People within the radius of any survey site anywhere, counting each cell once."""
    from . import population as pop

    frame = pop.population_by_group(points.assign(all="world"), "all", radius_km=radius_km)
    row = frame.iloc[0]
    return {"radius_km": radius_km, "epoch": pop.EPOCH, "sites": int(row["sites"]),
            "cells": int(row["cells"]), "population": float(row["population"])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--species", action="store_true", help="refresh the OBIS species counts")
    parser.add_argument("--population", action="store_true", help="refresh the population layer")
    parser.add_argument("--radius", type=float, default=50.0, help="population radius in km")
    parser.add_argument("--regions", action="store_true", default=True,
                        help="also write the per-region versions")
    parser.add_argument("--global-total", action="store_true",
                        help="also count the people near any survey site anywhere")
    args = parser.parse_args()
    if not (args.species or args.population):
        parser.error("choose --species, --population, or both")

    points = site_points()
    COMMUNITY.mkdir(parents=True, exist_ok=True)
    sources.CACHE.mkdir(parents=True, exist_ok=True)

    if args.species:
        by_country, by_region = species(points)
        by_country.to_csv(COMMUNITY / "reef_species.csv", index=False)
        by_region.to_csv(sources.CACHE / "species_by_region.csv", index=False)
        print(f"reef_species.csv: {len(by_country)} countries from {len(by_region)} regions; "
              f"{by_region['datasets_dropped'].sum()} dataset appearances left out for their licence")

    if args.population:
        by_country = population(points, "country", args.radius)
        by_country.to_csv(COMMUNITY / "population_near_reefs.csv", index=False)
        total = by_country[f"population_{int(args.radius)}km"].sum()
        print(f"population_near_reefs.csv: {len(by_country)} countries, {total:,.0f} people "
              f"within {int(args.radius)} km of a survey site")
        if args.regions:
            population(points, "ecoregion", args.radius).to_csv(
                sources.CACHE / "population_by_region.csv", index=False)
        if args.global_total:
            import json

            summary = global_total(points, args.radius)
            (COMMUNITY / "population_near_reefs.meta.json").write_text(json.dumps(summary, indent=2))
            print(f"  worldwide, each cell counted once: {summary['population']:,.0f} people "
                  f"within {int(args.radius)} km of one of {summary['sites']:,} survey sites")


if __name__ == "__main__":
    main()
