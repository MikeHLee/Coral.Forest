import numpy as np

from coralforest.data import FEATURES, LEVELS, encode, load_clean, load_raw, target


def test_raw_shape():
    raw = load_raw()
    assert raw.shape == (12392, 12)
    assert (raw["Bleaching"] == "Yes").sum() == 361


def test_clean_matches_r_and_report():
    df = load_clean()
    # R/reproduce_2018.R and the 2018 report both give 9,111 rows and 255 events.
    assert len(df) == 9111
    assert target(df).sum() == 255
    assert not (df == "unknown").any().any()
    assert not (df["Sewage"] == "k").any()
    assert not (df["Dynamite"] == "prior").any()


def test_every_level_is_known():
    df = load_clean()
    for col, levels in LEVELS.items():
        assert set(df[col]) <= set(levels), col


def test_bleaching_years_pinned():
    # Pins the extract: a changed or re-exported file fails here first.
    df = load_clean()
    years = sorted(df.loc[target(df) == 1, "Year"].unique())
    assert years == [1998, 1999, 2000, 2001, 2002, 2003]


def test_encode_shape_and_order():
    df = load_clean()
    X, cols = encode(df)
    assert X.shape == (9111, 11)
    assert cols == FEATURES
    sew = cols.index("Sewage")
    assert np.array_equal(np.unique(X[:, sew]), np.arange(4))
