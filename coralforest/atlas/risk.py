"""How much of a reef bleaches at a given heat dose, fitted per region and year.

A single survey is a noisy measure: programmes differ in what they record, and a
survey can miss the week a reef bleached. One region in one year is the unit
this module models. For each region and year it takes the mean heat dose of the
surveys (degree heating weeks) and two outcomes:

- `share_severe`, the share of surveys in which at least 10% of the coral was
  bleached;
- `impairment`, the mean share of surveyed coral that was bleached, which the
  flow model in `network.py` reads as the share of reef service at risk.

Both are rates, so the equations are linear on the logit scale, fitted with the
conjugate Bayesian update in `ecoanalyst.causal`.

Regions differ for reasons that have nothing to do with heat: which programme
surveys them, which species live there, how deep the reefs are. The heat dose is
therefore split into two parents, following Mundlak (1978):

    dhw_region  the region's own long-run mean heat dose  (between regions)
    dhw_dev     this year's heat dose minus that mean     (within a region)

The coefficient on `dhw_dev` is the one the scenarios use: it is identified by
years when a region was hotter or cooler than its own normal, so a fixed
difference between regions cannot produce it. The coefficient on `dhw_region`
absorbs those fixed differences.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from ecoanalyst import CausalModel
from ecoanalyst.causal import link

SEVERE_THRESHOLD = 0.10  # share of coral bleached at which a survey counts as severe
MIN_SURVEYS = 10         # surveys a region-year needs to enter the fit
RATE_CLIP = (0.005, 0.995)
PARENTS = ("dhw_dev", "dhw_region")
OUTCOMES = ("impairment", "share_severe")


def region_years(frame: pd.DataFrame, level: str = "ecoregion",
                 min_surveys: int = MIN_SURVEYS) -> pd.DataFrame:
    """One row per region and year, from the survey table of `bleaching.samples`."""
    work = frame.dropna(subset=[level, "percent_bleaching", "dhw"]).copy()
    work["severe"] = (work["impairment"] >= SEVERE_THRESHOLD).astype(float)
    grouped = work.groupby([level, "year"])
    table = grouped.agg(
        surveys=("sample_id", "size"),
        sites=("site_id", "nunique"),
        lat=("lat", "mean"),
        lon=("lon", "mean"),
        dhw=("dhw", "mean"),
        dhw_max=("dhw", "max"),
        share_severe=("severe", "mean"),
        impairment=("impairment", "mean"),
        sources=("source", "nunique"),
    ).reset_index().rename(columns={level: "region"})
    table = table[table["surveys"] >= min_surveys].reset_index(drop=True)
    table["dhw_region"] = table.groupby("region")["dhw"].transform("mean")
    table["dhw_dev"] = table["dhw"] - table["dhw_region"]
    table["region_years"] = table.groupby("region")["year"].transform("size")
    return table


def build_model() -> CausalModel:
    model = CausalModel()
    model.add_variable("dhw_dev", unit="C-weeks",
                       description="heat dose of the year minus the region's mean")
    model.add_variable("dhw_region", unit="C-weeks", description="the region's mean heat dose")
    model.add_variable("impairment", kind="rate", parents=list(PARENTS),
                       description="share of surveyed coral that is bleached",
                       binds="node:reef:waste_rate")
    model.add_variable("share_severe", kind="rate", parents=list(PARENTS),
                       description="share of surveys with at least 10% of coral bleached")
    return model


def _columns(table: pd.DataFrame) -> Dict[str, np.ndarray]:
    data = {name: table[name].to_numpy(dtype=float) for name in PARENTS}
    for outcome in OUTCOMES:
        data[outcome] = table[outcome].clip(*RATE_CLIP).to_numpy(dtype=float)
    return data


def fit(table: pd.DataFrame, model: Optional[CausalModel] = None) -> CausalModel:
    model = model or build_model()
    model.fit(_columns(table))
    return model


def coefficients(model: CausalModel) -> pd.DataFrame:
    rows = []
    for outcome in OUTCOMES:
        posterior = model.variables[outcome].posterior
        spread = np.sqrt(np.diag(np.asarray(posterior.scale)) * posterior.b / (posterior.a - 1))
        for term, mean, sd in zip(posterior.terms, posterior.mean, spread):
            rows.append({"outcome": outcome, "term": term, "mean": mean, "sd": sd,
                         "n_obs": posterior.n_obs})
    return pd.DataFrame(rows)


def predict(model: CausalModel, table: pd.DataFrame, outcome: str = "impairment") -> np.ndarray:
    return model.predict({name: table[name].to_numpy(dtype=float) for name in PARENTS},
                         variables=[outcome])[outcome]


@dataclass
class Scores:
    label: str
    n: int
    mae: float
    mae_baseline: float
    skill: float
    spearman: float

    def as_dict(self) -> Dict[str, object]:
        return self.__dict__.copy()


def _score(label: str, observed: np.ndarray, predicted: np.ndarray, baseline: float) -> Scores:
    mae = float(np.mean(np.abs(predicted - observed)))
    mae0 = float(np.mean(np.abs(baseline - observed)))
    order = pd.Series(predicted).rank().corr(pd.Series(observed).rank())
    return Scores(label, len(observed), mae, mae0, 1 - mae / mae0 if mae0 else float("nan"),
                  float(order))


def leave_one_region_out(table: pd.DataFrame, outcome: str = "impairment") -> Scores:
    """Fit on all regions but one, predict that region's years. Repeat for each region."""
    predicted = np.full(len(table), np.nan)
    for region, index in table.groupby("region").groups.items():
        held = table.index.isin(index)
        train = table[~held]
        if train["region"].nunique() < 5:
            continue
        predicted[held] = predict(fit(train), table[held], outcome)
    ok = np.isfinite(predicted)
    return _score("leave-one-region-out", table.loc[ok, outcome].to_numpy(), predicted[ok],
                  float(table.loc[ok, outcome].mean()))


def later_years(table: pd.DataFrame, split_year: int = 2010, outcome: str = "impairment") -> Scores:
    train, test = table[table["year"] <= split_year], table[table["year"] > split_year]
    predicted = predict(fit(train), test, outcome)
    return _score(f"train <= {split_year}, test after", test[outcome].to_numpy(), predicted,
                  float(train[outcome].mean()))


def heat_blind_control(table: pd.DataFrame, outcome: str = "impairment") -> Scores:
    """Negative control: the same fit with the heat dose replaced by its region mean.

    Every year of a region then gets the same prediction, so any skill left is
    the skill of knowing which region it is, not how hot the year was.
    """
    flat = table.copy()
    flat["dhw_dev"] = 0.0
    predicted = predict(fit(table), flat, outcome)
    return _score("heat-blind control", table[outcome].to_numpy(), predicted,
                  float(table[outcome].mean()))


def evaluate(table: pd.DataFrame, outcome: str = "impairment") -> pd.DataFrame:
    rows = [leave_one_region_out(table, outcome), later_years(table, outcome=outcome),
            heat_blind_control(table, outcome)]
    fitted = predict(fit(table), table, outcome)
    rows.insert(0, _score("in-sample", table[outcome].to_numpy(), fitted, float(table[outcome].mean())))
    return pd.DataFrame([row.as_dict() for row in rows])


def within_between(table: pd.DataFrame, outcome: str = "impairment") -> Dict[str, float]:
    """Least-squares slopes of logit(outcome) on the two heat-dose parents, for a check."""
    y = link("rate", table[outcome].clip(*RATE_CLIP).to_numpy())
    design = np.column_stack([np.ones(len(table)), table["dhw_dev"], table["dhw_region"]])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    return {"intercept": float(beta[0]), "within_region_per_c_week": float(beta[1]),
            "between_region_per_c_week": float(beta[2]), "n": int(len(table))}


def scenario_frame(table: pd.DataFrame, warming: pd.Series, reference_years: int = 10) -> pd.DataFrame:
    """One row per region: recent heat dose, the warming step, and the two scenarios.

    `warming` maps a region to the increase in heat dose, in Celsius-weeks, that
    its nearest Coral Reef Watch station's trend gives by 2050.
    """
    last = table.sort_values("year").groupby("region").tail(reference_years)
    out = last.groupby("region").agg(
        years=("year", "size"), last_year=("year", "max"), lat=("lat", "mean"), lon=("lon", "mean"),
        surveys=("surveys", "sum"), dhw_recent=("dhw", "mean"), dhw_region=("dhw_region", "first"),
        observed_impairment=("impairment", "mean"), observed_share_severe=("share_severe", "mean"),
    ).reset_index()
    out["warming_c_weeks"] = out["region"].map(warming).astype(float)
    out["dhw_dev_now"] = out["dhw_recent"] - out["dhw_region"]
    out["dhw_dev_2050"] = out["dhw_dev_now"] + out["warming_c_weeks"]
    return out


def expected(model: CausalModel, dhw_dev: Sequence[float], dhw_region: Sequence[float],
             outcome: str = "impairment", n: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Posterior mean and 5th-95th percentile of an outcome at given heat doses."""
    rows: List[Dict[str, float]] = []
    for dev, region in zip(dhw_dev, dhw_region):
        draws = model.sample(n, do={"dhw_dev": float(dev), "dhw_region": float(region)},
                             seed=seed)[outcome]
        rows.append({"mean": float(np.mean(draws)), "p05": float(np.percentile(draws, 5)),
                     "p50": float(np.percentile(draws, 50)), "p95": float(np.percentile(draws, 95))})
    return pd.DataFrame(rows)
