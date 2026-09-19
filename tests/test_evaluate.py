import numpy as np
import pytest

from coralforest.data import load_clean
from coralforest.evaluate import (
    bleaching_years, expected_calibration_error, later_years, leave_one_ocean_out,
    leave_one_year_out, random_5fold, score,
)


def test_score_hand_computed():
    y = np.array([1, 1, 0, 0, 0, 0])
    p = np.array([0.9, 0.4, 0.6, 0.1, 0.2, 0.3])
    s = score(y, p, (p >= 0.5).astype(int))
    assert s["confusion"] == {"tn": 3, "fp": 1, "fn": 1, "tp": 1}
    assert s["sensitivity"] == pytest.approx(0.5)
    assert s["specificity"] == pytest.approx(0.75)
    assert s["accuracy_always_no"] == pytest.approx(4 / 6)
    assert s["cost_per_1000"] == pytest.approx(1000 * (3 * 1 + 1 * 1) / 6)
    assert s["cost_per_1000_always_no"] == pytest.approx(1000 * 3 * 2 / 6)
    assert s["roc_auc"] == pytest.approx(7 / 8)


def test_score_single_class_is_undefined_not_an_error():
    s = score(np.zeros(4, dtype=int), np.full(4, 0.1), np.zeros(4, dtype=int))
    assert s["roc_auc"] is None and s["sensitivity"] is None
    assert s["specificity"] == 1.0


def test_ece_perfect_and_worst():
    y = np.array([0, 0, 1, 1])
    assert expected_calibration_error(y, y.astype(float)) == pytest.approx(0.0)
    assert expected_calibration_error(y, 1.0 - y) == pytest.approx(1.0)


@pytest.fixture(scope="module")
def df():
    return load_clean()


@pytest.mark.parametrize("splitter", [random_5fold, leave_one_ocean_out, leave_one_year_out, later_years])
def test_splits_are_disjoint(df, splitter):
    for _, tr, te in splitter(df, 0):
        assert len(np.intersect1d(tr, te)) == 0
        assert len(te) > 0


def test_ocean_split_holds_out_whole_basins(df):
    for ocean, tr, te in leave_one_ocean_out(df, 0):
        assert set(df["Ocean"].iloc[te]) == {ocean}
        assert ocean not in set(df["Ocean"].iloc[tr])


def test_year_split_covers_bleaching_years_only(df):
    labels = [label for label, _, _ in leave_one_year_out(df, 0)]
    assert labels == [str(y) for y in bleaching_years(df)]


def test_later_years_split_starts_after_last_bleaching_year(df):
    (_, tr, te), = list(later_years(df, 0))
    last = bleaching_years(df)[-1]
    assert df["Year"].iloc[tr].max() == last
    assert df["Year"].iloc[te].min() > last
