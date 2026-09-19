"""What a bleached reef costs the communities around it, as a flow network.

Each country gets a small EcoAnalyst network. A benefit enters at its own
source node, passes through the reef, and arrives at the community that holds
it:

    fishers  --> reef --> fishery   --> households   reef fishers
    spending --> reef --> tourism   --> households   reef tourism value
    people   --> reef --> shoreline --> households   people the reef shields
    species  --> reef --> habitat   --> reef_life    reef-associated species

The reef node is the only lossy component. Its loss rate is the share of coral
that is bleached, which the fitted model of `risk.py` sets from the heat dose
(the causal variable `impairment` is bound to `node:reef:waste_rate`). What the
reef loses is the part of the benefit exposed to bleaching.

The share of a benefit that depends on living coral is a configuration choice,
not a measurement, so it sits on the edge into the reef as an `efficiency` in
conversion mode: only that share reaches the reef, and the rest is reported as
converted and is never at risk. The default is 1.0, which treats every unit as
coral-dependent and is the largest number the data can support. Read it as an
upper bound.

Read the results as exposure, not as damage. "38,000 reef fishers exposed"
means the model puts 38,000 fishers behind the share of reef function that
bleaching impairs in that year. It does not say that those people lose their
living, and the flow model has no view on how fast a reef recovers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from ecoanalyst import CausalModel, EcosystemNetwork
from ecoanalyst.causal import path_cost, simulate

REEF = "reef"
HOUSEHOLDS = "households"
REEF_LIFE = "reef_life"
SOURCE_PREFIX = "benefit_"

#: service key -> (node id, community node, unit of the benefit)
SERVICES: Dict[str, tuple] = {
    "fishery": ("fishery", HOUSEHOLDS, "reef fishers"),
    "tourism": ("tourism", HOUSEHOLDS, "USD per year"),
    "shoreline": ("shoreline", HOUSEHOLDS, "people protected per year"),
    "habitat": ("habitat", REEF_LIFE, "reef-associated species"),
}


@dataclass
class CountryCase:
    """One country's benefits and the heat doses to compare."""

    country: str
    benefits: Dict[str, float]                 # service key -> quantity, in that service's unit
    dhw_dev_now: float
    dhw_dev_2050: float
    dhw_region: float
    dependence: Dict[str, float] = field(default_factory=dict)  # service key -> share of the benefit that needs live coral


def build_network(case: CountryCase) -> EcosystemNetwork:
    net = EcosystemNetwork(name=f"Reef services, {case.country}", network_type="ecosystem",
                           efficiency_mode="conversion")
    net.add_node("reef", "Reef", f"Coral reef, {case.country}", node_id=REEF,
                 properties={"capacity": {"value": 1.0, "unit": "reef"}, "waste_rate": 0.0})
    net.add_node("community", "Community", "Coastal households", node_id=HOUSEHOLDS,
                 properties={"waste_rate": 0.0})
    net.add_node("habitat", "Habitat", "Reef-associated species", node_id=REEF_LIFE,
                 properties={"waste_rate": 0.0})
    for key, (node_id, community, unit) in SERVICES.items():
        source = SOURCE_PREFIX + key
        net.add_node("source", "Benefit", f"{key.title()} benefit ({unit})", node_id=source,
                     properties={"waste_rate": 0.0})
        net.add_node("service", "Service", key.title(), node_id=node_id,
                     properties={"waste_rate": 0.0})
        # Only the coral-dependent share of the benefit reaches the reef.
        net.add_edge(source, REEF, "service",
                     {"waste_rate": 0.0, "efficiency": float(case.dependence.get(key, 1.0))})
        net.add_edge(REEF, node_id, "service", {"waste_rate": 0.0})
        net.add_edge(node_id, community, "service", {"waste_rate": 0.0})
    return net


def path_of(service: str) -> list:
    node_id, community, _unit = SERVICES[service]
    return [SOURCE_PREFIX + service, REEF, node_id, community]


def assess(case: CountryCase, model: CausalModel, n: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Benefit exposed to bleaching now and in 2050, per service, with a 5-95% band.

    Both runs share the seed, so the difference between them comes from the
    heat dose alone.
    """
    net = build_network(case)
    net.causal = model
    outcomes = {key: path_cost(path_of(key), input_quantity=float(case.benefits[key]))
                for key in SERVICES if case.benefits.get(key) not in (None, 0) and not pd.isna(case.benefits.get(key))}
    if not outcomes:
        return pd.DataFrame()
    runs = {}
    for label, dev in (("now", case.dhw_dev_now), ("2050", case.dhw_dev_2050)):
        runs[label] = simulate(net, outcomes, n=n, seed=seed,
                               do={"dhw_dev": float(dev), "dhw_region": float(case.dhw_region)})
    rows = []
    for key in outcomes:
        now, later = runs["now"].outcomes[key], runs["2050"].outcomes[key]
        extra = later - now
        rows.append({
            "country": case.country, "service": key, "unit": SERVICES[key][2],
            "benefit": float(case.benefits[key]), "dependence": float(case.dependence.get(key, 1.0)),
            "exposed_now": float(np.mean(now)), "exposed_now_p05": float(np.percentile(now, 5)),
            "exposed_now_p95": float(np.percentile(now, 95)),
            "exposed_2050": float(np.mean(later)), "exposed_2050_p05": float(np.percentile(later, 5)),
            "exposed_2050_p95": float(np.percentile(later, 95)),
            "additional": float(np.mean(extra)), "additional_p05": float(np.percentile(extra, 5)),
            "additional_p95": float(np.percentile(extra, 95)),
        })
    return pd.DataFrame(rows)


def assess_all(cases: Sequence[CountryCase], model: CausalModel, n: int = 2000,
               seed: int = 0) -> pd.DataFrame:
    frames = [assess(case, model, n=n, seed=seed) for case in cases]
    frames = [frame for frame in frames if not frame.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def cases_from_tables(countries: pd.DataFrame,
                      dependence: Optional[Mapping[str, float]] = None) -> list:
    """Cases from a country table that already carries its benefit columns.

    `services.attach` puts the benefit columns on the risk table; a country
    with no benefit value for a service simply has no path for it.
    """
    cases = []
    for row in countries.itertuples():
        values = {key: float(getattr(row, key, float("nan")) or float("nan")) for key in SERVICES
                  if hasattr(row, key)}
        cases.append(CountryCase(
            country=row.country, benefits=values, dhw_dev_now=row.dhw_dev_now,
            dhw_dev_2050=row.dhw_dev_2050, dhw_region=row.dhw_region,
            dependence=dict(dependence or {}),
        ))
    return cases
