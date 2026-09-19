# Coral.Forest

Coral.Forest started as a 2018 Modern Regression course project: a random forest that flags coral reefs at risk of bleaching from Reef Check survey data. This repository keeps the original notebook and report, and adds a reproducible rebuild in R and Python with evaluation splits that test the model on ocean basins and years it has not seen.

| Path | Contents |
|---|---|
| `original/` | The 2018 R notebook and PDF report, unchanged |
| `data/` | The Reef Check extract used in 2018, with a [data sheet](data/README.md) |
| `R/reproduce_2018.R` | The 2018 cleaning and model call, runnable from the repository root |
| `coralforest/` | Python package: data loading, a port of the R forest, the evaluation harness |
| `results/` | Metrics written by the R script and by `python -m coralforest.evaluate` |
| `tests/` | pytest suite, including checks that the Python port agrees with R |

## Run it

Python 3.10 or later:

```bash
python -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/python -m coralforest.evaluate
.venv/bin/pytest -q
```

The evaluation takes about 40 seconds on a laptop and writes `results/scores.json` and `results/scores.md`.

R, with the `randomForest` package installed:

```bash
Rscript R/reproduce_2018.R
```

Set `REPEATS=20` to fit 20 forests with consecutive seeds and report the spread.

## The 2018 model

The response is `Bleaching` (yes or no). The eleven predictors are ocean basin, year, depth, recent storms, and seven human-stressor ratings (overall human impact, siltation, dynamite fishing, poison fishing, sewage, industrial pollution, commercial fishing). After cleaning, 9,111 surveys remain and 255 of them (2.8%) record bleaching.

The model is `randomForest(Bleaching ~ ., ntree = 500, mtry = 5, nodesize = sqrt(12), sampsize = c(300, 100))`. Each tree draws 300 surveys without bleaching and 100 with bleaching, so the trees see bleaching far more often than it occurs. The 2018 goal was to treat a missed bleaching event as three times as costly as a false alarm. Performance was read from the out-of-bag (OOB) votes.

## Python port

`coralforest/forest.py` ports the 2018 `randomForest` call to scikit-learn trees: per-class sampling of 300 and 100 surveys per tree, five candidate predictors per split, the same minimum node size, and out-of-bag voting. The trees split integer-coded factor levels on thresholds, whereas R splits a factor on any subset of its levels; the module docstring lists the other correspondences. `tests/test_forest.py` checks that the port's out-of-bag votes track the R rerun row by row.

## Evaluation

`python -m coralforest.evaluate` scores three models under five splits:

| Split | What it tests |
|---|---|
| `oob` | The 2018 method: each survey is scored by the trees that did not draw it |
| `random_5fold` | Stratified 5-fold cross-validation with shuffled rows |
| `leave_one_ocean_out` | Train on five ocean basins, test on the sixth (`Ocean` is dropped as a predictor) |
| `leave_one_year_out` | Hold out, one at a time, each year that has at least one bleaching record |
| `later_years` | Train on the years up to the last year with a bleaching record, test on all later years |

The models are `rf_2018` (the Python port with the 2018 settings), `rf_2018_noyear` (the same forest without `Year`) and `gbm` (scikit-learn's histogram gradient boosting on the same predictors, flagging a site when its predicted probability is 0.25 or more, which is the cost-minimising cut when a miss costs three false alarms).

The harness writes every score to `results/scores.json` and a summary table to `results/scores.md`. Both files are regenerated on each run and are not committed. Pass `--seed N` to check another seed.

## License

The code is licensed under the GNU General Public License v3.0 (see `LICENSE`). The survey data were collected by Reef Check; cite Reef Check when you use them.
