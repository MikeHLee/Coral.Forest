"""What the reefs of a country are worth to the people and the species near them.

Three published country tables and one population layer, all kept in
`data/community/` with their sources in `data/community/README.md`:

| File | Column | Meaning |
|---|---|---|
| `reef_fishers.csv` | `fishery` | people who fish the reefs of that country |
| `reef_tourism.csv` | `tourism` | reef-associated tourism value, US dollars a year |
| `reef_flood_protection.csv` | `shoreline` | people whose annual flood exposure the reefs lower |
| `reef_species.csv` | `habitat` | reef-associated species recorded near its reefs |
| `population_near_reefs.csv` | `population_50km` | people living within 50 km of a survey site |

Country names differ between the tables and the survey records, so every name
passes through `normalise()`. `load_benefits()` reports which countries it could
not match instead of dropping them silently.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

DATA = Path(__file__).resolve().parent.parent.parent / "data" / "community"
FILES = {
    "fishery": ("reef_fishers.csv", "fishery"),
    "tourism": ("reef_tourism.csv", "tourism"),
    "shoreline": ("reef_flood_protection.csv", "shoreline"),
    "habitat": ("reef_species.csv", "habitat"),
    "population_50km": ("population_near_reefs.csv", "population_50km"),
}

#: where each benefit column comes from, for the page footer and the summary
CITATIONS = {
    "fishery": ("Teh, L.S.L., Teh, L.C.L. and Sumaila, U.R. (2013). A global estimate of the number "
                "of coral reef fishers. PLoS ONE 8(6): e65397. doi:10.1371/journal.pone.0065397",
                "CC BY 4.0", "https://doi.org/10.1371/journal.pone.0065397"),
    "tourism": ("Spalding, M. et al. (2017). Mapping the global value and distribution of coral reef "
                "tourism. Marine Policy 82: 104-113. doi:10.1016/j.marpol.2017.05.014",
                "CC BY 4.0", "https://doi.org/10.1016/j.marpol.2017.05.014"),
    "shoreline": ("Beck, M.W. et al. (2018). The global flood protection savings provided by coral "
                  "reefs. Nature Communications 9: 2186. doi:10.1038/s41467-018-04568-z",
                  "CC BY 4.0", "https://doi.org/10.1038/s41467-018-04568-z"),
    "habitat": ("Ocean Biodiversity Information System (OBIS), reef-associated species records.",
                "see the dataset records", "https://obis.org"),
    "population_50km": ("European Commission Joint Research Centre, Global Human Settlement Layer "
                        "GHS-POP R2023A.", "CC BY 4.0", "https://ghsl.jrc.ec.europa.eu/"),
}

#: names that differ between the survey records and the published tables
ALIASES = {
    "bahamas, the": "bahamas",
    "brunei darsm": "brunei",
    "egypt, arab rep.": "egypt",
    "iran, islamic rep.": "iran",
    "micronesia, fed. sts.": "micronesia",
    "venezuela, rb": "venezuela",
    "netherland antilles": "netherlands antilles",
    "comoro islands": "comoros",
    "turks and caicos": "turks and caicos islands",
    "united states virgin islands": "virgin islands, u.s.",
    "virgin islands (u.s.)": "virgin islands, u.s.",
    "northern mariana islands": "commonwealth of the northern mariana islands",
    "united states (florida)": "florida & us gulf of mexico",
    "united states (hawaii)": "hawaii",
    "tanzania": "united republic of tanzania",
    "united states": "united states of america",
    "usa": "united states of america",
    "us": "united states of america",
    "vietnam": "viet nam",
    "cote d'ivoire": "ivory coast",
    "cape verde": "cabo verde",
    "east timor": "timor-leste",
    "burma": "myanmar",
    "federated states of micronesia": "micronesia",
    "micronesia, federated states of": "micronesia",
    "republic of korea": "south korea",
    "korea, republic of": "south korea",
    "iran (islamic republic of)": "iran",
    "venezuela (bolivarian republic of)": "venezuela",
    "bolivia (plurinational state of)": "bolivia",
    "brunei darussalam": "brunei",
    "syrian arab republic": "syria",
    "russian federation": "russia",
    "saint vincent and grenadines": "saint vincent and the grenadines",
    "st. vincent and the grenadines": "saint vincent and the grenadines",
    "st. lucia": "saint lucia",
    "st. kitts and nevis": "saint kitts and nevis",
    "trinidad & tobago": "trinidad and tobago",
    "antigua & barbuda": "antigua and barbuda",
    "turks & caicos islands": "turks and caicos islands",
    "british virgin islands": "virgin islands, british",
    "us virgin islands": "virgin islands, u.s.",
    "u.s. virgin islands": "virgin islands, u.s.",
    "papua new guinea ": "papua new guinea",
}


def normalise(name: object) -> str:
    text = str(name).strip().lower().replace("’", "'")
    text = " ".join(text.split())
    return ALIASES.get(text, text)


def _read(path: Path, column: str) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    if "country" not in frame.columns or column not in frame.columns:
        raise ValueError(f"{path.name}: needs columns 'country' and {column!r}, has {list(frame.columns)}")
    frame = frame[["country", column]].copy()
    frame["key"] = frame["country"].map(normalise)
    return frame.drop(columns="country").groupby("key", as_index=False).sum()


def load_benefits(directory: Optional[Path] = None) -> pd.DataFrame:
    """One row per country with every benefit column that is available."""
    directory = directory or DATA
    merged: Optional[pd.DataFrame] = None
    for column, (filename, source_column) in FILES.items():
        frame = _read(directory / filename, source_column)
        if frame is None:
            continue
        frame = frame.rename(columns={source_column: column})
        merged = frame if merged is None else merged.merge(frame, on="key", how="outer")
    if merged is None:
        return pd.DataFrame(columns=["key"])
    return merged


def citations(benefits: pd.DataFrame) -> list:
    """Citation, licence, and link for every benefit column that is present."""
    return [{"citation": CITATIONS[column][0], "licence": CITATIONS[column][1],
             "licence_url": CITATIONS[column][2]}
            for column in CITATIONS if column in getattr(benefits, "columns", [])]


def attach(countries: pd.DataFrame, benefits: pd.DataFrame) -> pd.DataFrame:
    """Join benefits onto a country risk table, keeping unmatched countries."""
    out = countries.copy()
    out["key"] = out["country"].map(normalise)
    if benefits.empty:
        return out
    return out.merge(benefits, on="key", how="left")


def unmatched(countries: pd.DataFrame, benefits: pd.DataFrame) -> Dict[str, List[str]]:
    """Countries on one side only, so a gap is visible rather than silent."""
    if benefits.empty:
        return {"surveys_without_benefits": sorted(countries["country"].unique()), "benefits_without_surveys": []}
    survey_keys = set(countries["country"].map(normalise))
    benefit_keys = set(benefits["key"])
    return {
        "surveys_without_benefits": sorted(k for k in survey_keys - benefit_keys),
        "benefits_without_surveys": sorted(k for k in benefit_keys - survey_keys),
    }
