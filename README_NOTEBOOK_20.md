# Notebook 20 — Temporal robustness and paired diagnostics

Notebook 19 two-fold smoke test: calibrated temporal stacking mean macro ROC-AUC 0.571448 vs RF 0.566724; calibrated stack accuracy 0.385024 vs RF 0.392232. These are preliminary and **not** five-fold estimates.

## Run

1. Open `notebooks/20_temporal_robustness_diagnostics.ipynb` from your VS Code `.venv` kernel.
2. Run with `RUN_FULL_BENCHMARK=False` to analyze the existing Notebook 19 scores, or set `True` to rerun five chronological folds (computationally expensive).
3. Inspect `artifacts/temporal_stability/1d/paired_improvements.csv`, `fold_winners.csv`, `quality_summary.csv`, and `manifest.json`.

The benchmark invokes Notebook 19's existing inner refit / chronological meta and calibration procedure. The diagnostics do not load the final historical holdout or write to the prospective prediction ledger. Overlapping training windows mean fold-paired comparisons are descriptive, not formal hypothesis tests. Five folds are preferable to two, but not sufficient to demonstrate live trading value. Model versioning and prospective shadow testing are separate future work.
