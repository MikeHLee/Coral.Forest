"""Score the 2018 forest and a gradient-boosting baseline under several splits.

    python -m coralforest.evaluate            # writes results/scores.json + results/scores.md

Splits
------
oob                   The 2018 method: out-of-bag votes on all rows (forest only).
random_5fold          Stratified 5-fold cross-validation, rows shuffled.
leave_one_ocean_out   Train on five ocean basins, test on the sixth. Ocean is
                      dropped as a feature because the test basin is unseen.
leave_one_year_out    Hold out, one at a time, each year that has at least one
                      bleaching record, and train on every other year.
later_years           Train on the years up to the last year with a bleaching
                      record and score all later years. Metrics that need both
                      classes are reported as None when a fold lacks one.

Models
------
rf_2018         coralforest.forest.StratifiedForest with the 2018 settings.
rf_2018_noyear  The same forest without the Year column.
gbm             scikit-learn HistGradientBoostingClassifier, Ocean as a native
                categorical feature, other factors as ordered codes.

Each model keeps its own decision rule: the forests use the majority vote; the
gradient-boosting model flags a site when p >= 0.25, the cost-minimising cut
for the 2018 objective that a missed bleaching event costs three false alarms.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .data import FEATURES, REPO_ROOT, encode, load_clean, target
from .forest import StratifiedForest

COST_FN, COST_FP = 3.0, 1.0
GBM_THRESHOLD = COST_FP / (COST_FP + COST_FN)


# ---------------------------------------------------------------- metrics ---

def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    which = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        mask = which == b
        if mask.any():
            ece += mask.mean() * abs(y[mask].mean() - p[mask].mean())
    return float(ece)


def score(y: np.ndarray, p: np.ndarray, flagged: np.ndarray) -> dict:
    """Threshold-free and decision metrics. Undefined values are None."""
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    flagged = np.asarray(flagged, dtype=int)
    n, pos = len(y), int(y.sum())
    neg = n - pos
    tp = int(((flagged == 1) & (y == 1)).sum())
    fp = int(((flagged == 1) & (y == 0)).sum())
    fn = int(((flagged == 0) & (y == 1)).sum())
    tn = int(((flagged == 0) & (y == 0)).sum())
    both = pos > 0 and neg > 0
    return {
        "n": n,
        "positives": pos,
        "roc_auc": float(roc_auc_score(y, p)) if both else None,
        "pr_auc": float(average_precision_score(y, p)) if both else None,
        "pr_auc_no_skill": pos / n if n else None,
        "brier": float(brier_score_loss(y, p, pos_label=1)) if n else None,
        "brier_no_skill": (pos / n) * (1 - pos / n) if n else None,
        "ece": expected_calibration_error(y, p) if n else None,
        "accuracy": (tp + tn) / n if n else None,
        "accuracy_always_no": neg / n if n else None,
        "sensitivity": tp / pos if pos else None,
        "specificity": tn / neg if neg else None,
        "cost_per_1000": 1000 * (COST_FN * fn + COST_FP * fp) / n if n else None,
        "cost_per_1000_always_no": 1000 * COST_FN * pos / n if n else None,
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


# ----------------------------------------------------------------- models ---

@dataclass
class Model:
    name: str
    features: list[str]
    fit_predict: Callable[[pd.DataFrame, pd.DataFrame, list[str], int], tuple[np.ndarray, np.ndarray]]


def _rf(train: pd.DataFrame, test: pd.DataFrame, features: list[str], seed: int):
    Xtr, _ = encode(train, features)
    Xte, _ = encode(test, features)
    forest = StratifiedForest(seed=seed).fit(Xtr, target(train))
    share = forest.vote_share(Xte)
    return share, (share >= 0.5).astype(int)


def _gbm(train: pd.DataFrame, test: pd.DataFrame, features: list[str], seed: int):
    Xtr, cols = encode(train, features)
    Xte, _ = encode(test, features)
    model = HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=300, max_leaf_nodes=15, l2_regularization=1.0,
        categorical_features=[c == "Ocean" for c in cols], early_stopping=False,
        random_state=seed,
    )
    model.fit(Xtr, target(train))
    p = model.predict_proba(Xte)[:, 1]
    return p, (p >= GBM_THRESHOLD).astype(int)


MODELS = [
    Model("rf_2018", FEATURES, _rf),
    Model("rf_2018_noyear", [f for f in FEATURES if f != "Year"], _rf),
    Model("gbm", FEATURES, _gbm),
]


# ----------------------------------------------------------------- splits ---

Split = Iterator[tuple[str, np.ndarray, np.ndarray]]  # (fold label, train idx, test idx)


def random_5fold(df: pd.DataFrame, seed: int) -> Split:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for k, (tr, te) in enumerate(skf.split(df, target(df))):
        yield f"fold{k}", tr, te


def leave_one_ocean_out(df: pd.DataFrame, seed: int) -> Split:
    for ocean in sorted(df["Ocean"].unique()):
        te = np.flatnonzero(df["Ocean"] == ocean)
        tr = np.flatnonzero(df["Ocean"] != ocean)
        yield ocean, tr, te


def bleaching_years(df: pd.DataFrame) -> list[int]:
    """Years with at least one bleaching record, ascending."""
    return sorted(int(y) for y in df.loc[target(df) == 1, "Year"].unique())


def leave_one_year_out(df: pd.DataFrame, seed: int) -> Split:
    for year in bleaching_years(df):
        te = np.flatnonzero(df["Year"] == year)
        tr = np.flatnonzero(df["Year"] != year)
        yield str(year), tr, te


def later_years(df: pd.DataFrame, seed: int) -> Split:
    last = bleaching_years(df)[-1]
    tr = np.flatnonzero(df["Year"] <= last)
    te = np.flatnonzero(df["Year"] > last)
    yield f"after {last}", tr, te


SPLITS = {
    "random_5fold": (random_5fold, set()),
    "leave_one_ocean_out": (leave_one_ocean_out, {"Ocean"}),
    "leave_one_year_out": (leave_one_year_out, set()),
    "later_years": (later_years, set()),
}


def run_split(df: pd.DataFrame, split: str, model: Model, seed: int) -> dict:
    splitter, drop = SPLITS[split]
    features = [f for f in model.features if f not in drop]
    y = target(df)
    p = np.full(len(df), np.nan)
    flagged = np.zeros(len(df), dtype=int)
    per_fold = {}
    for label, tr, te in splitter(df, seed):
        pt, ft = model.fit_predict(df.iloc[tr], df.iloc[te], features, seed)
        p[te], flagged[te] = pt, ft
        per_fold[label] = score(y[te], pt, ft)
    scored = ~np.isnan(p)
    pooled = score(y[scored], p[scored], flagged[scored])
    pooled["folds"] = {k: {m: v[m] for m in ("n", "positives", "roc_auc", "sensitivity", "specificity")}
                       for k, v in per_fold.items()}
    pooled["features"] = features
    return pooled


def run_oob(df: pd.DataFrame, seed: int, features: list[str]) -> dict:
    X, _ = encode(df, features)
    forest = StratifiedForest(seed=seed).fit(X, target(df))
    share = forest.oob_votes_
    out = score(target(df), share, (share >= 0.5).astype(int))
    out["features"] = features
    return out


def run_all(seed: int = 2018) -> dict:
    df = load_clean()
    y = target(df)
    results: dict = {
        "dataset": {
            "file": "data/ReefCheck974.rdata",
            "rows_clean": int(len(df)),
            "positives": int(y.sum()),
            "years": [int(df["Year"].min()), int(df["Year"].max())],
            "years_with_positives": bleaching_years(df),
        },
        "decision_rules": {
            "rf_2018": "vote share >= 0.5",
            "rf_2018_noyear": "vote share >= 0.5",
            "gbm": f"p >= {GBM_THRESHOLD:.2f} (missed event costs {COST_FN:g} false alarms)",
        },
        "seed": seed,
        "splits": {"oob": {
            "rf_2018": run_oob(df, seed, MODELS[0].features),
            "rf_2018_noyear": run_oob(df, seed, MODELS[1].features),
        }},
    }
    for split in SPLITS:
        results["splits"][split] = {m.name: run_split(df, split, m, seed) for m in MODELS}
    return results


def _fmt(v, pct: bool = False) -> str:
    if v is None:
        return "n/a"
    return f"{100 * v:.1f}%" if pct else f"{v:.3f}"


def to_markdown(results: dict) -> str:
    rows = [
        "| Split | Model | ROC AUC | PR AUC (no skill) | Brier (no skill) | Sensitivity | Specificity | Cost / 1,000 (always No) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for split, models in results["splits"].items():
        for name, s in models.items():
            rows.append(
                f"| {split} | {name} | {_fmt(s['roc_auc'])} | "
                f"{_fmt(s['pr_auc'])} ({_fmt(s['pr_auc_no_skill'])}) | "
                f"{_fmt(s['brier'])} ({_fmt(s['brier_no_skill'])}) | "
                f"{_fmt(s['sensitivity'], True)} | {_fmt(s['specificity'], True)} | "
                f"{s['cost_per_1000']:.0f} ({s['cost_per_1000_always_no']:.0f}) |"
            )
    return "\n".join(rows) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--seed", type=int, default=2018)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results")
    args = parser.parse_args(argv)
    results = run_all(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "scores.json").write_text(json.dumps(results, indent=2) + "\n")
    (args.out / "scores.md").write_text(to_markdown(results))
    print(to_markdown(results))


if __name__ == "__main__":
    main()
