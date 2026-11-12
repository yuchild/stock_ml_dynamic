# Notebook 14 — Frozen classifier and prospective paper tracking

Run `notebooks/14_prospective_prediction_tracking.ipynb` **after** the latest daily OHLCV ingestion has completed. Select the `Python (stock_ml_dynamic)` Jupyter kernel. This notebook fetches fresh OHLCV data directly from the read-only PostgreSQL source, rather than using the historical master-table snapshot.

## First run

1. Run the repository tests: `python -m pytest -q`. Install the notebook dependencies first if needed: `python -m pip install -e '.[ml,notebook,dev]'`.
2. Run Notebook 14 with `TRAIN_NEW_MODEL = True` to create a **new prospective deployment version** of the previously selected Logistic L2 classifier (`C=0.10`). It refits the training-only target tertile boundaries, candle K-means, encoders, scalers and model on all outcomes that have already been observed. It never re-evaluates or changes the closed historical holdout.
3. Review the newest source bar for every symbol. **Only after verifying bars are complete and future outcomes are unavailable**, set `RECORD_PREDICTIONS = True` to commit paper predictions to the ledger.

## Subsequent daily runs

Set `TRAIN_NEW_MODEL = False` and leave `MODEL_PATH = None` to load the **oldest** frozen version by default. Use the same frozen version for the entire prospective trial. Refresh OHLCV data; generate new predictions; append once; reconcile earlier predictions only after the next actual bar arrives. Re-running the same model/ticker/timestamp does not append duplicates.

## Files

- `artifacts/prospective/1d/models/<version>/model.joblib` — fitted model and full preprocessing bundle
- `artifacts/prospective/1d/models/<version>/manifest.json` — immutable version metadata and SHA-256 checksum
- `artifacts/prospective/1d/prediction_ledger.parquet` — append-only prediction fields; actual outcomes are populated later

**Limitations:** This is a single-writer local file ledger, not a transactional database. Confirm completeness of market bars before recording; missing trading days or ingestion gaps can change which later bar is treated as the next observation. Do not backdate predictions or treat historical scores as prospective results. The equal-weight market proxy uses contemporaneous prices of the other eligible tickers. Joblib/pickle model files must only be loaded from trusted local sources. The output is paper evaluation, not trading instructions.
