# Data audit — Notebook 05

This additive update adds `src/stock_ml_dynamic/data_audit.py`, `notebooks/05_data_leakage_audit.ipynb`, and `tests/test_data_audit.py`.

Run after Notebook 03 has saved `artifacts/master_tables/1d/{master_labeled.parquet,latest_inference.parquet,transformers.joblib,manifest.json}`. The notebook reports incomplete latest-bar symbols, missing feature columns, target alignment, chronological partition boundaries, and differences between the saved universe and live PostgreSQL.

The notebook is read-only. It cannot prove that the saved fitted transforms were originally trained only on historical data; source inspection shows the intended training cutoff logic. Future walk-forward experiments must refit quantile boundaries, KMeans, numeric scaler, and one-hot encoder separately for every fold.

To run tests: `python -m pytest -q tests/test_data_audit.py`.
