import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from coralforest.data import REPO_ROOT, encode, load_clean, target
from coralforest.forest import StratifiedForest

FIXTURES = REPO_ROOT / "tests" / "fixtures"
R_VOTES = FIXTURES / "r_reference_oob_votes.csv"
R_METRICS = FIXTURES / "r_reference_metrics.json"


def test_separable_toy_problem():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(600, 3))
    y = (X[:, 0] > 1.0).astype(int)
    forest = StratifiedForest(n_trees=50, sampsize=(100, 40), mtry=2, seed=1).fit(X, y)
    assert (forest.predict(X) == y).mean() > 0.95
    assert np.isfinite(forest.oob_votes_).all()


def test_needs_both_classes():
    with pytest.raises(ValueError):
        StratifiedForest(n_trees=2).fit(np.zeros((5, 2)), np.zeros(5, dtype=int))


@pytest.fixture(scope="module")
def forest_and_data():
    df = load_clean()
    X, _ = encode(df)
    y = target(df)
    return StratifiedForest(seed=2018).fit(X, y), y


def test_oob_votes_track_r_row_by_row(forest_and_data):
    # Rank agreement only: a changed class ratio (e.g. sampsize (300, 30)) still
    # correlates above 0.98. test_oob_rates_within_r_seed_spread catches that.
    forest, y = forest_and_data
    r = pd.read_csv(R_VOTES)
    assert np.array_equal(r["observed"].eq("Yes").to_numpy(dtype=int), y)
    assert np.corrcoef(forest.oob_votes_, r["oob_votes_yes"])[0, 1] > 0.98


def test_oob_rates_within_r_seed_spread(forest_and_data):
    forest, y = forest_and_data
    pred = (forest.oob_votes_ >= 0.5).astype(int)
    sens = ((pred == 1) & (y == 1)).sum() / (y == 1).sum()
    spec = ((pred == 0) & (y == 0)).sum() / (y == 0).sum()
    ref = json.loads(R_METRICS.read_text())
    # Integer-coded threshold splits vs R's factor-subset splits: allow a few points.
    assert abs(sens - ref["sensitivity_over_seeds"]["mean"]) < 0.05
    assert abs(spec - ref["specificity_over_seeds"]["mean"]) < 0.02
