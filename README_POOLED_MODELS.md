# Pooled model experiments (notebook 04)

Run notebook 03 first to generate `artifacts/master_tables/1d/` (master_labeled.parquet, latest_inference.parquet, transformers.joblib, manifest.json). Then run notebook 04 using the `Python (stock_ml_dynamic)` kernel in VS Code. Install with `python -m pip install -e '.[ml,notebook,dev]'`.

Notebook 04 fits a multiclass majority-prior baseline and HistGradientBoostingClassifier, plus a mean-return baseline and HistGradientBoostingRegressor. Training uses the saved chronological cutoff; test rows are strictly later than the cutoff. It uses saved training-only preprocessing, and predicts next-bar direction and adjusted close for inference-ready latest rows. Outputs are under `artifacts/pooled_runs/`.

These holdout metrics are exploratory. Do not repeatedly optimize against the same holdout or automatically deploy the best score. For multi-step price trajectories, train a separately validated horizon-aware model. Streamlit remains a results/deployment interface, not a training environment.
