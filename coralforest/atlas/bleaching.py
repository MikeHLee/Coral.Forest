"""Survey records from the global coral-bleaching database, cleaned for modelling.

The published file has one row per substrate class, so a survey appears several
times. `samples()` returns one row per survey (`Sample_ID`) with the columns the
atlas uses, and repairs two things that the file's metadata gets wrong:

- the temperature columns are in kelvin, although the metadata says Celsius;
- numeric columns arrive as text when a row holds a stray value.

Columns returned (units in brackets):

    sample_id, site_id, source, lat, lon, year, month, country, ecoregion,
    realm, ocean, depth_m [m], distance_to_shore_m [m], turbidity [Kd490],
    cyclone_frequency, sst_mean_c [C], ssta_c [C], dhw [C-weeks],
    percent_bleaching [%], bleached [0/1], impairment [0-1]

`dhw` is `TSA_DHW`: degree heating weeks accumulated above the site's maximum
monthly mean at the time of the survey, the standard measure of the heat dose a
reef has taken. `impairment` is the share of the surveyed coral recorded as
bleached, which the flow model reads as the share of reef function at risk.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from . import sources

KELVIN = 273.15
NUMERIC = ["Latitude_Degrees", "Longitude_Degrees", "Date_Year", "Date_Month", "Depth_m",
           "Distance_to_Shore", "Turbidity", "Cyclone_Frequency", "Temperature_Mean",
           "SSTA", "TSA_DHW", "Percent_Bleaching"]
RENAME = {
    "Sample_ID": "sample_id", "Site_ID": "site_id", "Data_Source": "source",
    "Latitude_Degrees": "lat", "Longitude_Degrees": "lon", "Date_Year": "year",
    "Date_Month": "month", "Country_Name": "country", "Ecoregion_Name": "ecoregion",
    "Realm_Name": "realm", "Ocean_Name": "ocean", "Depth_m": "depth_m",
    "Distance_to_Shore": "distance_to_shore_m", "Turbidity": "turbidity",
    "Cyclone_Frequency": "cyclone_frequency", "SSTA": "ssta_c", "TSA_DHW": "dhw",
    "Percent_Bleaching": "percent_bleaching",
}


def load_raw(path: Optional[Path] = None, *, offline: bool = False) -> pd.DataFrame:
    return pd.read_csv(path or sources.fetch("bleaching", offline=offline), low_memory=False)


def samples(raw: Optional[pd.DataFrame] = None, **kwargs) -> pd.DataFrame:
    """One row per survey, cleaned. `kwargs` go to `load_raw`."""
    df = load_raw(**kwargs) if raw is None else raw
    df = df.drop_duplicates("Sample_ID").copy()
    for column in NUMERIC:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df["sst_mean_c"] = df["Temperature_Mean"] - KELVIN
    out = df.rename(columns=RENAME)[[*RENAME.values(), "sst_mean_c"]]
    out = out[out["lat"].between(-90, 90) & out["lon"].between(-180, 180)]
    out["bleached"] = (out["percent_bleaching"] > 0).astype("float")
    out.loc[out["percent_bleaching"].isna(), "bleached"] = np.nan
    out["impairment"] = out["percent_bleaching"] / 100.0
    return out.reset_index(drop=True)


def modelling_frame(sample_table: pd.DataFrame) -> pd.DataFrame:
    """Surveys that can enter the dose-response fit: bleaching and heat dose known."""
    frame = sample_table.dropna(subset=["percent_bleaching", "dhw"]).copy()
    frame["depth_m"] = frame["depth_m"].fillna(frame["depth_m"].median())
    frame["log_depth"] = np.log1p(frame["depth_m"])
    frame["log_turbidity"] = np.log(frame["turbidity"].clip(lower=1e-3))
    return frame.reset_index(drop=True)


def by_region(sample_table: pd.DataFrame, level: str = "ecoregion") -> pd.DataFrame:
    """Survey counts, mean heat dose, and observed bleaching per region."""
    frame = sample_table.dropna(subset=[level])
    grouped = frame.groupby(level)
    out = grouped.agg(
        surveys=("sample_id", "size"),
        sites=("site_id", "nunique"),
        first_year=("year", "min"),
        last_year=("year", "max"),
        lat=("lat", "mean"),
        lon=("lon", "mean"),
        dhw_mean=("dhw", "mean"),
        dhw_p90=("dhw", lambda x: float(np.nanpercentile(x, 90))),
        share_bleached=("bleached", "mean"),
        mean_impairment=("impairment", "mean"),
    )
    return out.reset_index()
