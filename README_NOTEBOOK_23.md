# Notebook 23 — Class-1 Entry Precision

The seven-quantile target `[2,2,0,0,1,1,1]` remains unchanged. This notebook uses three expanding outer folds and two inner folds, selecting a model and class-1 entry threshold exclusively on inner-fold predictions. Only class 1 triggers a hypothetical long entry, and only when it is the model's highest-probability class and its probability clears the threshold.

The inner selection criterion is pooled class-1 precision with at least 5% signal coverage. Configurations include balanced and unweighted logistic regression and Random Forest, enabling a controlled comparison of the user's requested balanced training versus an unweighted diagnostic. The legacy prospective deployment is not changed.

Run `notebooks/23_class1_entry_precision.ipynb` from VS Code after Notebook 03 has produced the master parquet. Output CSVs and a manifest are written under `artifacts/entry_precision/1d`.

**Important limitations:** The cost-adjusted next-bar return is a simple illustrative one-bar proxy, not executable P&L; q=7 Up does not necessarily mean a profitable long trade. Thresholds and hyperparameters are exploratory and should not be repeatedly selected on outer folds. The historical holdout remains closed.
