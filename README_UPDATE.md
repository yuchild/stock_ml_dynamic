# Candle Feature and Target Table v1

Copy `src/stock_ml_dynamic/candle_table.py`, `notebooks/02_candle_feature_target_table.ipynb`, and `tests/test_candle_table.py` into matching paths in your existing repository. No existing files are replaced.

In VS Code, open the new notebook, select `Python (stock_ml_dynamic)`, then run all cells. Requires pandas, numpy, scikit-learn, matplotlib, SQLAlchemy and your existing database driver. Install via `python -m pip install -e '.[ml,notebook,dev]'` from the repository root.

Data source: `raw.stock_prices`. One symbol and interval per notebook run. No database writes, no predictive-model training.

## Important modeling rules

- Quantile cutoffs are fitted on training next-bar returns only. `direction` is the *next* bar's realized return category.
- The candle clustering elbow curve and fitted scaler/K-means use training candles only; arbitrary cluster numbers do not inherently mean bullish or bearish.
- The adjusted-close z-score uses the previous 30 bars, not the current bar.
- Slope and concavity are derivatives of a quadratic interpolant through the most recent three equally spaced bars. Units are price per bar and price per bar squared, not percentages. For irregular timestamps, the fit still treats bars as evenly spaced.
- `adj_close` is the current price reference. `next_adj_close` is the one-bar-ahead regression target; `next_return` is also available for return regression.
- For a next-bar forecast made after a bar closes, use only completed candle features. Calendar features describe the completed candle's timestamp in America/New_York by default.
- Current code labels equal-range candles as zero proportions. Check invalid OHLC data before training.
- Quantile fitting and K-means must be re-fitted inside each walk-forward training fold. Never refit them on test/live observations.
- With daily data, hour_of_day may be constant and should be omitted when uninformative.
