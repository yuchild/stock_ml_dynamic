# Notebook 18 — Seed robustness and paired ensemble diagnostics

Notebook: `notebooks/18_seed_robustness_diagnostics.ipynb`.

This development-only experiment repeats the Notebook 17 Logistic L2, Random Forest, residual MLP, and fixed soft-voting models across several random seeds with chronological folds. It measures paired ROC-AUC deltas, run-to-run variability, runtime and residual neural training histories. It does not use the historical final holdout or prospective prediction ledger.

Start with `SMOKE_TEST=True` (2 seeds × 2 folds, 8 epochs). Change to `False` for 5 seeds × 5 folds and 35 epochs, which is much more expensive. Reports appear under `artifacts/robustness_diagnostics/1d/`.

Repeated seeds/folds are correlated. Reported deltas are descriptive, not statistical proof. The existing model version remains frozen for prospective monitoring.
