"""Build the atlas: survey records and heat stress in, map and tables out.

    python -m coralforest.atlas.build [--n 2000] [--seed 0] [--offline]

Steps:

1. Read the survey records and keep the surveys that carry both a bleaching
   measure and a heat dose.
2. Find the nearest Coral Reef Watch station for every survey site, and read
   that station's yearly maximum heat dose from 1985 on.
3. Fit the region-year model of `risk.py` and score it on regions and on years
   it did not see.
4. Convert each station's trend into a warming step on the survey scale, and
   ask the model what share of coral bleaches at today's heat dose and at the
   2050 heat dose.
5. Roll the regions up to countries, attach the community tables, and run the
   flow model to get the benefit exposed to bleaching.
6. Write the tables, the two figures, and the page.

Outputs land in `results/atlas/` (tables and a JSON summary), `figures/`, and
`docs/atlas.html`. `results/` is not committed; the figures and the page are.
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from . import bleaching, figures, network, page, risk, services, sources, thermal

ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS = ROOT / "results" / "atlas"
FIGURES = ROOT / "figures"
PAGE = ROOT / "docs" / "atlas.html"
SEVERE_LABEL = f"{int(risk.SEVERE_THRESHOLD * 100)}% of coral or more"


def survey_table(offline: bool = False) -> pd.DataFrame:
    frame = bleaching.modelling_frame(bleaching.samples(offline=offline))
    stations = thermal.stations(offline=offline)
    nearest = thermal.nearest_station(frame["lat"].to_numpy(), frame["lon"].to_numpy(), stations)
    return pd.concat([frame.reset_index(drop=True), nearest], axis=1)


def calibration(region_table: pd.DataFrame, surveys: pd.DataFrame, annual: pd.DataFrame) -> Dict[str, float]:
    """Scale between the survey heat dose and the station's yearly maximum.

    The two come from different products: the survey records carry a thermal
    stress anomaly accumulated at the survey date, the station carries the
    yearly maximum of Coral Reef Watch's own heat dose. They rank the same
    years alike but not on the same scale, so a warming step in station units
    is multiplied by this slope before it is used with the fitted model.
    """
    modal = (surveys.groupby(["ecoregion", "year"])
             .agg(slug=("slug", lambda x: x.mode().iloc[0]), station_km=("station_km", "median"))
             .reset_index().rename(columns={"ecoregion": "region"}))
    joined = (region_table.merge(modal, on=["region", "year"])
              .merge(annual.rename(columns={"dhw_max": "station_dhw_max"}), on=["slug", "year"], how="left")
              .dropna(subset=["station_dhw_max"]))
    slope, intercept = np.polyfit(joined["station_dhw_max"], joined["dhw"], 1)
    spearman = float(joined[["dhw", "station_dhw_max"]].corr(method="spearman").iloc[0, 1])
    return {"slope": float(slope), "intercept": float(intercept), "spearman": spearman,
            "region_years": int(len(joined)), "median_station_km": float(joined["station_km"].median())}


def warming_steps(surveys: pd.DataFrame, trends: pd.DataFrame, slope: float) -> pd.Series:
    """Increase in heat dose by 2050 for each region, on the survey scale."""
    per_site = surveys.merge(trends, on="slug", how="left")
    step = (per_site["dhw_2050"] - per_site["dhw_recent"]) * slope
    return step.groupby(per_site["ecoregion"]).mean()


def region_risk(model, regions: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    out = regions.copy()
    for label, column in (("now", "dhw_dev_now"), ("2050", "dhw_dev_2050")):
        for outcome in ("impairment", "share_severe"):
            values = risk.expected(model, out[column], out["dhw_region"], outcome=outcome, n=n, seed=seed)
            out[f"{outcome}_{label}"] = values["mean"].to_numpy()
            out[f"{outcome}_{label}_p05"] = values["p05"].to_numpy()
            out[f"{outcome}_{label}_p95"] = values["p95"].to_numpy()
    out["impairment_change"] = out["impairment_2050"] - out["impairment_now"]
    return out


def region_extras(regions: pd.DataFrame) -> pd.DataFrame:
    """Add the harvested per-region layers when they are in the cache."""
    out = regions
    for filename, columns in (("species_by_region.csv", {"habitat": "coral_species"}),
                              ("population_by_region.csv", {"population_50km": "population_50km",
                                                            "population": "population_50km"})):
        path = sources.CACHE / filename
        if not path.exists():
            continue
        frame = pd.read_csv(path).rename(columns={"ecoregion": "region"})
        keep = {source: target for source, target in columns.items() if source in frame.columns}
        if not keep:
            continue
        frame = frame[["region", *keep]].rename(columns=keep)
        out = out.merge(frame, on="region", how="left")
    return out


def country_risk(surveys: pd.DataFrame, regions: pd.DataFrame) -> pd.DataFrame:
    """Region values rolled up to countries, weighted by surveys."""
    weights = (surveys.groupby(["country", "ecoregion"]).size().rename("surveys").reset_index()
               .rename(columns={"ecoregion": "region"}))
    joined = weights.merge(regions, on="region", suffixes=("", "_region"))
    if joined.empty:
        return pd.DataFrame()

    def weighted(group: pd.DataFrame, column: str) -> float:
        return float(np.average(group[column], weights=group["surveys"]))

    rows = []
    for country, group in joined.groupby("country"):
        rows.append({
            "country": country,
            "surveys": int(group["surveys"].sum()),
            "regions": int(group["region"].nunique()),
            "lat": weighted(group, "lat"), "lon": weighted(group, "lon"),
            "dhw_region": weighted(group, "dhw_region"),
            "dhw_dev_now": weighted(group, "dhw_dev_now"),
            "dhw_dev_2050": weighted(group, "dhw_dev_2050"),
            "warming_c_weeks": weighted(group, "warming_c_weeks"),
            "impairment_now": weighted(group, "impairment_now"),
            "impairment_2050": weighted(group, "impairment_2050"),
            "share_severe_now": weighted(group, "share_severe_now"),
            "share_severe_2050": weighted(group, "share_severe_2050"),
        })
    return pd.DataFrame(rows).sort_values("country").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--n", type=int, default=2000, help="samples per simulated case")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--offline", action="store_true", help="fail rather than download anything")
    parser.add_argument("--level", default="ecoregion", choices=["ecoregion", "realm", "country"])
    args = parser.parse_args()

    surveys = survey_table(offline=args.offline)
    table = risk.region_years(surveys, level=args.level)
    model = risk.fit(table)
    coefficients = risk.coefficients(model)
    scores = pd.concat([risk.evaluate(table, outcome).assign(outcome=outcome)
                        for outcome in risk.OUTCOMES], ignore_index=True)

    annual = thermal.annual_dhw()
    annual["year"] = annual["year"].astype(int)
    trends = thermal.trends(annual)
    scale = calibration(table, surveys, annual)
    steps = warming_steps(surveys, trends, scale["slope"])
    regions = risk.scenario_frame(table, steps)
    regions = region_extras(region_risk(model, regions, n=args.n, seed=args.seed))

    countries = country_risk(surveys, regions)
    benefits = services.load_benefits()
    countries = services.attach(countries, benefits)
    gaps = services.unmatched(countries, benefits)
    cases = network.cases_from_tables(countries)
    exposure = network.assess_all(cases, model, n=args.n, seed=args.seed) if cases else pd.DataFrame()

    RESULTS.mkdir(parents=True, exist_ok=True)
    table.to_csv(RESULTS / "region_years.csv", index=False)
    regions.to_csv(RESULTS / "regions.csv", index=False)
    countries.to_csv(RESULTS / "countries.csv", index=False)
    coefficients.to_csv(RESULTS / "coefficients.csv", index=False)
    scores.to_csv(RESULTS / "evaluation.csv", index=False)
    if not exposure.empty:
        exposure.to_csv(RESULTS / "exposure.csv", index=False)

    summary = {
        "built": date.today().isoformat(),
        "level": args.level,
        "surveys": int(len(surveys)),
        "sites": int(surveys["site_id"].nunique()),
        "years": [int(surveys["year"].min()), int(surveys["year"].max())],
        "regions": int(table["region"].nunique()),
        "region_years": int(len(table)),
        "severe_threshold": risk.SEVERE_THRESHOLD,
        "calibration": scale,
        "warming_c_weeks_median": float(np.nanmedian(regions["warming_c_weeks"])),
        "station_trend_per_decade_median": float(trends["dhw_per_decade"].median()),
        "within_between": risk.within_between(table),
        "coefficients": coefficients.to_dict("records"),
        "evaluation": scores.to_dict("records"),
        "impairment_now_median": float(regions["impairment_now"].median()),
        "impairment_2050_median": float(regions["impairment_2050"].median()),
        "benefit_gaps": gaps,
        "sources": [s.__dict__ for s in sources.SOURCES.values()],
        "community_sources": services.citations(benefits),
    }
    (RESULTS / "atlas.json").write_text(json.dumps(summary, indent=2, default=str))

    figures.world_map(
        regions, FIGURES / "atlas_map.png", column="impairment_2050",
        title="Coral bleaching risk at 2050 heat dose",
        subtitle=(f"{len(regions)} reef regions, {len(surveys):,} surveys 1980-2020; "
                  "circle size is survey count"), offline=args.offline)
    if not exposure.empty:
        for service, label in (("shoreline", "people whose flood exposure the reef lowers"),
                               ("tourism", "reef tourism value, US dollars a year"),
                               ("fishery", "reef fishers")):
            part = exposure[exposure["service"] == service]
            if not part.empty:
                figures.country_bars(part, FIGURES / f"atlas_{service}.png", label=label,
                                     title=f"Benefit exposed to bleaching: {service}")
    page.write(PAGE, summary=summary, regions=regions, countries=countries, exposure=exposure)

    print(f"regions {len(regions)} | region-years {len(table)} | surveys {len(surveys):,}")
    print(f"within-region slope {summary['within_between']['within_region_per_c_week']:+.3f} logit per C-week; "
          f"station trend {summary['station_trend_per_decade_median']:+.2f} C-weeks per decade (median)")
    print(scores.round(3).to_string(index=False))
    print(f"median share of coral bleached: now {summary['impairment_now_median']:.1%}, "
          f"2050 {summary['impairment_2050_median']:.1%}")
    if not exposure.empty:
        for service, group in exposure.groupby("service"):
            if service == "habitat":
                # Species counts do not add up across countries: a species that
                # lives in several of them would be counted once per country.
                top = group.nlargest(1, "exposed_2050").iloc[0]
                print(f"  {service:10s} per country, most exposed {top['country']}: "
                      f"{top['exposed_now']:,.0f} -> {top['exposed_2050']:,.0f} of "
                      f"{top['benefit']:,.0f} {group['unit'].iloc[0]}"
                      f" ({group['country'].nunique()} countries)")
                continue
            print(f"  {service:10s} exposed now {group['exposed_now'].sum():,.0f} -> "
                  f"2050 {group['exposed_2050'].sum():,.0f} {group['unit'].iloc[0]}"
                  f" ({group['country'].nunique()} countries)")
    print(f"wrote {RESULTS}, {FIGURES}, {PAGE}")


if __name__ == "__main__":
    main()
