# Notebook 06: Expanding-window walk-forward validation

Run `notebooks/03_universe_master_table.ipynb` first to generate `artifacts/master_tables/1d/`. Then run `notebooks/06_walk_forward_validation.ipynb` from the repository root using the VS Code `.venv` kernel.

The notebook re-fits quantile labels, K-means candle clustering, numeric scaling and categorical one-hot encoding in **each fold**. The saved global `direction` and `candle_type` columns are overwritten, not used as training labels. All 220 symbols share a timestamp cutoff; only labels observed by that cutoff enter training. The evaluation is exploratory; no models are promoted to deployment.

The VYLR diagnostic reads its raw adjusted closing prices from PostgreSQL and reports whether there are 30 valid prior observations. It does not fill missing data.

The output metrics are written to `artifacts/walk_forward/1d/`. The default is three expanding folds and a 30,000-row training-only sample for K-means (to keep the clustering analysis practical). Full model training uses all complete historical training rows. The notebook uses fixed k=4; elbow selection is not retuned per fold.

Run tests: `python -m pytest -q`.

Caution: The historical table from Notebook 03 already has a globally fitted `direction` and `candle_type`, but Notebook 06 recalculates them. A future improvement is to store the raw causal table independently, including rows that may not have received a label in Notebook 03, to guarantee complete fold coverage. Results do not imply tradeable predictive skill.
