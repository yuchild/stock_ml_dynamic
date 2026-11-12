# Notebook 12: nested chronological regularization validation

Run `notebooks/12_nested_regularization_validation.ipynb` in the existing VS Code kernel after Notebook 03 has saved the master table.

- The original three-class tertile target is retained, with cutoffs fitted separately within each inner and outer training split.
- Historical/market features retain 5-bar and 20-bar volatility; 10-bar volatility is omitted.
- Five expanding outer folds, each with a chronological inner validation split. Inner log-loss selects Logistic L2 regularization `C` from 0.03, 0.1, 0.3, 1.0.
- Compare tuned L2 against fixed-C L2 and a prior-class baseline, using identical outer test rows.
- Training is capped at 120,000 sampled rows per fit; test rows are not sampled.
- Fit times, ROC curves, fold-level accuracy, balanced accuracy, macro OVR ROC-AUC and log-loss are saved under `artifacts/regularization_diagnostics/1d`.
- The newest 20% of timestamps remains locked out. No holdout labels are evaluated.

The notebook requires the master table files in either `artifacts/master_tables/1d` or `notebooks/artifacts/master_tables/1d`.
