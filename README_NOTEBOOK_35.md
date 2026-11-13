# Notebook 35: exact coverage and market regimes

1. Keep the `artifacts/` directory from your existing installation; do not overwrite it with the ZIP.
2. Run Notebook 34 first if `artifacts/notebook34_calibration_coverage/validation_predictions.parquet` does not exist.
3. Run `notebooks/35_exact_coverage_regime_paired_signals.ipynb` and share its saved outputs.

This notebook reuses saved outer-development validation predictions, does not refit models, and does not access the closed historical final holdout or prospective prediction ledger. Exact top-k is an ex-post ranking diagnostic only. Regimes are descriptive proxies derived from historical validation outcomes with a two-date lag. Date-block bootstrap is per fold.
