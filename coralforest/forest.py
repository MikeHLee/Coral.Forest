"""A port of the 2018 R call

    randomForest(Bleaching ~ ., ntree = 500, mtry = 5,
                 nodesize = sqrt(12), sampsize = c(300, 100))

to scikit-learn decision trees.

Correspondence with R randomForest 4.x (classification):

- sampsize = c(300, 100): each tree draws 300 "No" rows and 100 "Yes" rows,
  with replacement, stratified by class. Rows not drawn are that tree's
  out-of-bag (OOB) rows.
- mtry = 5: max_features=5 candidate variables at each split.
- nodesize = 3.46: R stops splitting a node whose in-bag population is at or
  below nodesize, so a node splits only when it holds 4 or more draws. That is
  min_samples_split=4 here.
- Prediction is a majority vote of per-tree class labels; the "Yes" vote share
  is the score. With 300:100 sampling the vote share is not a probability of
  bleaching: the trees see "Yes" at 25% while the data has it at 2.8%.

One difference remains: R splits a factor on any subset of its levels, while
these trees split integer codes on a threshold (see data.encode).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.tree import DecisionTreeClassifier


@dataclass
class StratifiedForest:
    n_trees: int = 500
    sampsize: tuple[int, int] = (300, 100)  # (negatives, positives) per tree
    mtry: int = 5
    nodesize: float = 12 ** 0.5
    seed: int = 2018
    trees_: list = field(default_factory=list, repr=False)
    oob_votes_: np.ndarray | None = field(default=None, repr=False)
    oob_counts_: np.ndarray | None = field(default=None, repr=False)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "StratifiedForest":
        rng = np.random.default_rng(self.seed)
        neg = np.flatnonzero(y == 0)
        pos = np.flatnonzero(y == 1)
        if len(pos) == 0 or len(neg) == 0:
            raise ValueError("training data needs both classes")
        n = len(y)
        votes = np.zeros(n)
        counts = np.zeros(n)
        self.trees_ = []
        min_split = int(np.floor(self.nodesize)) + 1
        for _ in range(self.n_trees):
            idx = np.concatenate([
                rng.choice(neg, size=self.sampsize[0], replace=True),
                rng.choice(pos, size=self.sampsize[1], replace=True),
            ])
            tree = DecisionTreeClassifier(
                max_features=min(self.mtry, X.shape[1]),
                min_samples_split=min_split,
                random_state=int(rng.integers(2**31 - 1)),
            )
            tree.fit(X[idx], y[idx])
            self.trees_.append(tree)
            oob = np.ones(n, dtype=bool)
            oob[idx] = False
            if oob.any():
                votes[oob] += tree.predict(X[oob])
                counts[oob] += 1
        self.oob_votes_ = np.divide(votes, counts, out=np.full(n, np.nan), where=counts > 0)
        self.oob_counts_ = counts
        return self

    def vote_share(self, X: np.ndarray) -> np.ndarray:
        """Fraction of trees that vote "Yes" for each row."""
        if not self.trees_:
            raise RuntimeError("fit first")
        return np.mean([t.predict(X) for t in self.trees_], axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Majority vote, ties to "Yes" (R breaks ties at random)."""
        return (self.vote_share(X) >= 0.5).astype(int)
