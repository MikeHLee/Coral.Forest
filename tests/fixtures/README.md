# Test fixtures

`r_reference_metrics.json` and `r_reference_oob_votes.csv` are the outputs of

```bash
REPEATS=20 Rscript R/reproduce_2018.R
```

run with R 4.3.1 and randomForest 4.7-1.2 (seed 2018 for the votes), copied
from `results/`. `tests/test_forest.py` compares the Python port against them.
Regenerate them only when the R reference itself changes.
