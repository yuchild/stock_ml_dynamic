# Notebook 22 — Nested accuracy-first optimization

Three expanding **outer** chronological folds, two expanding **inner** chronological folds. Inner validation chooses among a predeclared grid of six feature/model configurations; outer folds evaluate the selection process. Accuracy is primary; balanced accuracy breaks ties. Original last 20% historical holdout remains untouched. Prospective ledger remains untouched.

Open `notebooks/22_nested_accuracy_optimization.ipynb` in VS Code. Default training caps: logistic 25,000 rows, Random Forest 15,000 rows. This is a compute-conscious initial run; do not compare different training caps as if they were identical experiments. The notebook reads Notebook 03's existing master table and writes `artifacts/nested_accuracy/1d/` CSV reports and a manifest.

Note: training-only target thresholds and candle clustering are refit for each fold; features are causal rolling features. Each candidate uses the same complete-case rows. The nested outer folds evaluate selection, not any one fixed model; three folds provide limited statistical precision. Do not tune using outer scores or the historical holdout.
