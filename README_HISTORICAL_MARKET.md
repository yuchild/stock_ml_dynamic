# Notebook 08 — Historical and market-relative features

Run `notebooks/08_historical_market_baselines.ipynb` after Notebook 03 generated the saved master table.

- Filters ticker histories with fewer than 31 bars or invalid latest z-score.
- Adds lagged cumulative returns (1/2/3/5/10/20), rolling volatility, moving-average distance, rolling position, relative volume, normalized slope/concavity, peer market return, relative strength and peer momentum.
- Peer market proxy is contemporaneous leave-one-out equal-weight return of other eligible symbols (not SPY); requires >=10 peers.
- Uses 5 expanding chronological folds and fits quantile targets, KMeans candle type, encoders, scalers independently per fold.
- Compares same complete test rows for original and enhanced features. Models: prior baseline, L2 logistic, L1 logistic, Gaussian NB. No trees/NN.
- Writes fold metrics, ROC coordinates, figures and timings to repository-root `artifacts/historical_market_benchmark/1d`.

Limitations: No market benchmark outside the dataset; missing/partial intraday bars can impair cross-sectional alignment. Historical universe selection can cause survivorship bias. This is exploratory CV; keep final chronological holdout untouched for confirmatory evaluation.
