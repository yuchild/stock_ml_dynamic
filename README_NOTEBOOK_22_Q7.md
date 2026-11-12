# Notebook 22 — q=7 balanced-target experiment

The experimental target uses seven *training-fitted* quantile bins, mapped in ascending order to `[2,2,0,0,1,1,1]`. Labels are 2=down, 0=middle, 1=up. `class_weight="balanced"` is used in Logistic Regression and Random Forest.

Legacy q=3 remains the default in shared fold APIs for saved-model compatibility; Notebook 22 explicitly opts into q=7. Existing model artifacts and prospective ledgers must **not** be reused with this target.

The up class covers ~3/7 of quantiles by construction, so the majority-class baseline is around 42.9%, not 33.3%. Weighted training does not make evaluation labels balanced. Compare accuracy, balanced accuracy, up precision/recall, and trade coverage; never interpret an increase in accuracy alone as evidence of better entry signals.

Run `python -m pytest -q tests/test_q7_target.py`, then execute `notebooks/22_nested_accuracy_optimization.ipynb`. Actual data-driven metrics require the local master-table artifacts and are not included in this package.
