# Stock ML Dynamic — notebook-first workflow

PostgreSQL is the read-only data source. All feature engineering, target selection, training and model evaluation take place in Jupyter notebooks. Streamlit displays market data and saved experiment results; it does not train models.

## Install and launch

```bash
cd ~/Documents/github/stock_ml_dynamic
source .venv/bin/activate
python -m pip install -e ".[ml,notebook,dev]"
python -m ipykernel install --user --name stock-ml-dynamic --display-name "Python (stock_ml_dynamic)"
python -m jupyter lab
```

Open `notebooks/01_feature_target_experiments.ipynb`, select `Python (stock_ml_dynamic)` and run its cells. The notebook uses the existing PostgreSQL socket connection. Start the dashboard separately with `python -m streamlit run app.py`.

## Repository roles

- `notebooks/`: interactive feature, target, training and evaluation experiments
- `src/stock_ml_dynamic/`: reusable feature calculations, specifications and training logic
- `artifacts/experiments/`: saved models and metadata (local, not committed)
- `app.py` and `pages/`: read-only visualization and saved model evaluation; no training UI

Saved model ranking is exploratory. A single holdout and repeated feature tuning are not a validated production selection process.

## All-stock master feature table (notebook-first)

Open `notebooks/03_universe_master_table.ipynb` in VS Code. Install notebook dependencies
with `python -m pip install -e '.[ml,notebook,dev]'`. It reads all tickers for one
interval from PostgreSQL, computes per-stock causal features and next-bar targets,
fits global training-only direction quantiles and K-means candle types, and fits a
training-only scaler + one-hot encoder. It saves labeled rows, latest inference
rows, fitted transforms and a manifest under `artifacts/master_tables/<interval>/`.

Do not fit preprocessing again on validation, test or inference data. Never
randomly shuffle time-series rows for training. `next_adj_close` is the next-bar
regression target; `adj_close` is the current price. The last row per stock is
excluded from labeled training data but preserved for next-bar inference.
