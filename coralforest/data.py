"""Load the Reef Check survey table and apply the 2018 cleaning rules.

The cleaning mirrors original/How2TrainYourCoralBleachingAI.Rmd step for step.
R/reproduce_2018.R applies the same rules in R; tests/test_data.py checks that
both give 9,111 rows with 255 bleaching events.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
RDATA_PATH = REPO_ROOT / "data" / "ReefCheck974.rdata"

TARGET = "Bleaching"
NUMERIC = ["Year", "Depth"]
CATEGORICAL = [
    "Ocean", "Storms", "HumanImpact", "Siltation", "Dynamite",
    "Poison", "Sewage", "Industrial", "Commercial",
]
FEATURES = ["Ocean", "Year", "Depth", "Storms", "HumanImpact", "Siltation",
            "Dynamite", "Poison", "Sewage", "Industrial", "Commercial"]

# Level order used for the integer codes. Stressor levels follow their natural
# intensity order. Ocean has no order; its codes are only labels.
LEVELS = {
    "Ocean": ["Arabian Gulf", "Atlantic", "East Pacific", "Indian", "Pacific", "Red Sea"],
    "Storms": ["no", "yes"],
    "HumanImpact": ["none", "low", "moderate", "high"],
    "Siltation": ["never", "occasionally", "often", "always"],
    "Dynamite": ["none", "low", "moderate", "high"],
    "Poison": ["none", "low", "moderate", "high"],
    "Sewage": ["none", "low", "moderate", "high"],
    "Industrial": ["none", "low", "moderate", "high"],
    "Commercial": ["none", "low", "moderate", "high"],
}

# Blank-cell recodes from the 2018 notebook: blanks become "unknown" where the
# survey form already had an "unknown" level, otherwise the lowest level.
_BLANK_TO = {
    "Ocean": "unknown",
    "Storms": "unknown",
    "HumanImpact": "unknown",
    "Siltation": "never",
    "Dynamite": "none",
    "Poison": "unknown",
    "Sewage": "none",
    "Industrial": "none",
    "Commercial": "none",
}
_EXTRA = {"Storms": {"y": "yes"}, "Siltation": {"Occasionally": "occasionally"}}


def load_raw(path: Path | str = RDATA_PATH) -> pd.DataFrame:
    """Read the `ReefCheck` data frame from the .rdata file as plain strings."""
    import pyreadr

    frame = pyreadr.read_r(str(path))["ReefCheck"]
    out = pd.DataFrame(index=frame.index)
    for col in frame.columns:
        if col in NUMERIC:
            out[col] = frame[col].astype(float if col == "Depth" else int)
        else:
            out[col] = frame[col].astype(object).where(frame[col].notna(), "").astype(str)
    return out


def clean_2018(raw: pd.DataFrame) -> pd.DataFrame:
    """Apply the 2018 notebook's cleaning. Returns a new frame, index reset."""
    df = raw.copy()
    for col, blank in _BLANK_TO.items():
        values = df[col].replace("", blank)
        for old, new in _EXTRA.get(col, {}).items():
            values = values.replace(old, new)
        df[col] = values
    has_unknown = (df[CATEGORICAL + [TARGET]] == "unknown").any(axis=1)
    df = df.loc[~has_unknown]
    df = df.loc[df["Sewage"] != "k"]
    df = df.loc[df["Dynamite"] != "prior"]
    return df.reset_index(drop=True)


def load_clean(path: Path | str = RDATA_PATH) -> pd.DataFrame:
    return clean_2018(load_raw(path))


def encode(df: pd.DataFrame, features: list[str] | None = None) -> tuple[np.ndarray, list[str]]:
    """Integer-code the categorical columns in LEVELS order; keep numerics as is.

    R's randomForest splits a factor on any subset of its levels. scikit-learn
    trees split on thresholds, so each factor becomes one ordered integer column.
    This keeps the number of candidate variables per split equal to R (11), which
    is what `mtry` counts. Unseen levels raise, so a data change fails loudly.
    """
    features = list(features or FEATURES)
    cols = []
    for col in features:
        if col in LEVELS:
            codes = pd.Categorical(df[col], categories=LEVELS[col]).codes
            if (codes < 0).any():
                bad = sorted(set(df.loc[codes < 0, col]))
                raise ValueError(f"{col}: levels not in LEVELS: {bad}")
            cols.append(codes.astype(float))
        else:
            cols.append(df[col].to_numpy(dtype=float))
    return np.column_stack(cols), features


def target(df: pd.DataFrame) -> np.ndarray:
    """1 for observed bleaching, 0 otherwise."""
    return (df[TARGET] == "Yes").to_numpy(dtype=int)
