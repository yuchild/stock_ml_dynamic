# Notebook 36 — Chronological portfolio policy

Run Notebook 34 to produce `artifacts/notebook34_calibration_coverage/validation_predictions.parquet` and `policies.csv`, then run `notebooks/36_chronological_portfolio_policy.ipynb`. Notebook 36 uses fixed training-calibrated thresholds and applies 10-position per-date portfolio caps, cash drag, turnover costs and paired block-bootstrap comparisons. Sector caps require a complete point-in-time `symbol,sector` CSV; absent that input, no sector cap is applied. The final historical holdout and prospective ledger are not read.
