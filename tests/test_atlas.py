"""The atlas modules on small built tables, with no download and no network."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from coralforest.atlas import bleaching, services, thermal

ecoanalyst = pytest.importorskip("ecoanalyst", reason="the atlas needs ecoanalyst 3.0 or later")
risk = pytest.importorskip("coralforest.atlas.risk")
network = pytest.importorskip("coralforest.atlas.network")

KELVIN = 273.15


def raw_rows() -> pd.DataFrame:
    """Two surveys, the first recorded once per substrate as the published file does."""
    base = {
        "Site_ID": 1, "Sample_ID": 10, "Data_Source": "Reef_Check", "Latitude_Degrees": -18.0,
        "Longitude_Degrees": 147.0, "Ocean_Name": "Pacific", "Reef_ID": "r", "Realm_Name": "Central Indo-Pacific",
        "Ecoregion_Name": "Central Great Barrier Reef", "Country_Name": "Australia",
        "State_Island_Province_Name": "", "City_Town_Name": "", "Site_Name": "", "Distance_to_Shore": 400.0,
        "Exposure": "", "Turbidity": 0.05, "Cyclone_Frequency": 50.0, "Date_Day": 1, "Date_Month": 3,
        "Date_Year": 2016, "Depth_m": 6.0, "Substrate_Name": "Hard Coral", "Percent_Cover": 30.0,
        "Bleaching_Level": "Population", "Percent_Bleaching": 40.0, "ClimSST": 300.0,
        "Temperature_Kelvin": 302.0, "Temperature_Mean": 301.15, "Temperature_Minimum": 297.0,
        "Temperature_Maximum": 304.0, "Temperature_Kelvin_Standard_Deviation": 0.5, "Windspeed": 5.0,
        "SSTA": 0.9, "SSTA_Standard_Deviation": 0.2, "SSTA_Mean": 0.3, "SSTA_Minimum": -1.0,
        "SSTA_Maximum": 2.0, "SSTA_Frequency": 1, "SSTA_Frequency_Standard_Deviation": 0.1,
        "SSTA_FrequencyMax": 2, "SSTA_FrequencyMean": 1, "SSTA_DHW": 3.0,
        "SSTA_DHW_Standard_Deviation": 0.4, "SSTA_DHWMax": 5.0, "SSTA_DHWMean": 1.0, "TSA": 1.1,
        "TSA_Standard_Deviation": 0.3, "TSA_Minimum": -0.5, "TSA_Maximum": 2.5, "TSA_Mean": 0.4,
        "TSA_Frequency": 1, "TSA_Frequency_Standard_Deviation": 0.1, "TSA_FrequencyMax": 3,
        "TSA_FrequencyMean": 1, "TSA_DHW": 6.0, "TSA_DHW_Standard_Deviation": 0.5, "TSA_DHWMax": 9.0,
        "TSA_DHWMean": 2.0, "Date": "2016-03-01", "Site_Comments": "", "Sample_Comments": "",
        "Bleaching_Comments": "",
    }
    second_substrate = {**base, "Substrate_Name": "Soft Coral", "Percent_Cover": 12.0}
    other = {**base, "Site_ID": 2, "Sample_ID": 11, "Percent_Bleaching": 0.0, "TSA_DHW": 0.0,
             "Date_Year": 2013, "Temperature_Mean": 300.15}
    return pd.DataFrame([base, second_substrate, other])


def test_samples_keeps_one_row_per_survey_and_converts_temperature():
    table = bleaching.samples(raw_rows())
    assert len(table) == 2
    assert set(table["sample_id"]) == {10, 11}
    assert table.loc[0, "sst_mean_c"] == pytest.approx(301.15 - KELVIN)
    assert table.loc[0, "impairment"] == pytest.approx(0.40)
    assert list(table["bleached"]) == [1.0, 0.0]


def synthetic_region_years(n_regions=12, n_years=14, slope=0.5, seed=0) -> pd.DataFrame:
    """Regions differ in baseline and in heat; only the within-region step is `slope`."""
    rng = np.random.default_rng(seed)
    rows = []
    for region in range(n_regions):
        base_dhw = rng.uniform(0.2, 4.0)
        offset = rng.normal(0, 0.8)  # a fixed regional difference, unrelated to heat
        for year in range(2000, 2000 + n_years):
            dhw = max(0.0, base_dhw + rng.normal(0, 1.2))
            logit = -3.0 + offset + slope * (dhw - base_dhw) + 0.25 * base_dhw + rng.normal(0, 0.3)
            share = 1 / (1 + np.exp(-logit))
            rows.append({"region": f"r{region}", "year": year, "surveys": 20, "sites": 5,
                         "lat": rng.uniform(-20, 20), "lon": rng.uniform(-180, 180), "dhw": dhw,
                         "dhw_max": dhw, "share_severe": min(0.99, share * 1.5),
                         "impairment": share, "sources": 1})
    table = pd.DataFrame(rows)
    table["dhw_region"] = table.groupby("region")["dhw"].transform("mean")
    table["dhw_dev"] = table["dhw"] - table["dhw_region"]
    table["region_years"] = n_years
    return table


def test_region_years_splits_the_heat_dose_within_and_between():
    surveys = bleaching.samples(raw_rows())
    surveys["percent_bleaching"] = [40.0, 0.0]
    table = risk.region_years(surveys, min_surveys=1)
    assert set(table["region"]) == {"Central Great Barrier Reef"}
    assert table["dhw_region"].nunique() == 1
    assert table["dhw_dev"].sum() == pytest.approx(0.0)
    assert table.loc[table["year"] == 2016, "impairment"].iloc[0] == pytest.approx(0.4)


def test_fit_recovers_the_within_region_step():
    table = synthetic_region_years(slope=0.5)
    model = risk.fit(table)
    terms = dict(zip(model.variables["impairment"].posterior.terms,
                     model.variables["impairment"].posterior.mean))
    assert terms["dhw_dev"] == pytest.approx(0.5, abs=0.08)
    assert risk.within_between(table)["within_region_per_c_week"] == pytest.approx(0.5, abs=0.08)


def test_heat_blind_control_loses_the_ranking():
    """Negative control: with the year's heat dose removed, the ranking collapses."""
    table = synthetic_region_years(slope=0.7)
    scores = risk.evaluate(table).set_index("label")
    assert scores.loc["in-sample", "spearman"] > scores.loc["heat-blind control", "spearman"] + 0.2
    assert scores.loc["leave-one-region-out", "skill"] > 0


def test_more_heat_means_more_bleaching():
    model = risk.fit(synthetic_region_years(slope=0.6))
    values = risk.expected(model, [0.0, 2.0, 4.0], [1.0, 1.0, 1.0], n=500, seed=1)
    assert values["mean"].is_monotonic_increasing
    assert (values["p05"] < values["p95"]).all()


def case(**kwargs) -> network.CountryCase:
    defaults = dict(country="Testland", benefits={"fishery": 1000.0, "tourism": 5.0e6,
                                                  "shoreline": 20000.0, "habitat": 300.0},
                    dhw_dev_now=0.0, dhw_dev_2050=2.0, dhw_region=1.0)
    defaults.update(kwargs)
    return network.CountryCase(**defaults)


def test_exposure_rises_with_the_heat_dose_and_stays_inside_the_benefit():
    model = risk.fit(synthetic_region_years(slope=0.6))
    table = network.assess(case(), model, n=400, seed=0).set_index("service")
    assert (table["exposed_2050"] > table["exposed_now"]).all()
    assert (table["exposed_2050"] <= table["benefit"]).all()
    assert table.loc["tourism", "additional"] == pytest.approx(
        table.loc["tourism", "exposed_2050"] - table.loc["tourism", "exposed_now"], rel=1e-6)


def test_no_warming_means_no_additional_exposure():
    model = risk.fit(synthetic_region_years(slope=0.6))
    table = network.assess(case(dhw_dev_2050=0.0), model, n=400, seed=0)
    assert table["additional"].abs().max() == pytest.approx(0.0, abs=1e-9)


def test_dependence_share_scales_the_exposure():
    model = risk.fit(synthetic_region_years(slope=0.6))
    full = network.assess(case(), model, n=400, seed=0).set_index("service")
    half = network.assess(case(dependence={"tourism": 0.5}), model, n=400, seed=0).set_index("service")
    assert half.loc["tourism", "exposed_2050"] == pytest.approx(0.5 * full.loc["tourism", "exposed_2050"], rel=1e-6)
    assert half.loc["fishery", "exposed_2050"] == pytest.approx(full.loc["fishery", "exposed_2050"], rel=1e-6)


def test_nearest_station_picks_the_closer_one():
    stations = pd.DataFrame([{"slug": "near", "name": "Near", "lat": -18.2, "lon": 147.1},
                             {"slug": "far", "name": "Far", "lat": 25.0, "lon": -80.0}])
    out = thermal.nearest_station(np.array([-18.0]), np.array([147.0]), stations)
    assert out.loc[0, "slug"] == "near"
    assert out.loc[0, "station_km"] < 30


def test_country_names_are_matched_through_aliases():
    assert services.normalise("United States") == services.normalise("USA")
    assert services.normalise(" Viet Nam ") == "viet nam"
    countries = pd.DataFrame({"country": ["Fiji", "United States"]})
    benefits = pd.DataFrame({"key": ["fiji", "united states of america"], "fishery": [10.0, 20.0]})
    joined = services.attach(countries, benefits)
    assert list(joined["fishery"]) == [10.0, 20.0]
    assert services.unmatched(countries, benefits)["surveys_without_benefits"] == []


def test_benefit_gaps_are_reported(tmp_path: Path):
    (tmp_path / "reef_fishers.csv").write_text("country,fishery\nFiji,100\nTonga,50\n")
    benefits = services.load_benefits(tmp_path)
    assert set(benefits["key"]) == {"fiji", "tonga"}
    gaps = services.unmatched(pd.DataFrame({"country": ["Fiji", "Belize"]}), benefits)
    assert gaps["surveys_without_benefits"] == ["belize"]
    assert gaps["benefits_without_surveys"] == ["tonga"]
