# Notebook 26 — Volume, flow, breadth and regime feature research

This is a **development-only**, predeclared **three-fold chronological ablation**. It does not modify Notebook 15's prospective ledger or evaluate the closed historical holdout.

## Experiment

Four feature families: (1) price-only historical+technical, (2) price+volume, (3) volume-only, (4) price+volume flow and market breadth. Six model configurations: balanced/unweighted Logistic L2, balanced/unweighted Random Forest, balanced Extra Trees and balanced histogram gradient boosting. This is **72 model fits** (3 folds × 4 families × 6 models), each scored at four *diagnostic* thresholds, or 288 fold/feature/model/threshold rows. Some fits may take substantial time.

Features include relative volume and z-scores over 5/10/20/60 bars, volume dispersion, log dollar-volume proxy, signed volume pressure, Chaikin-style close-location money flow, price/volume rolling correlation, volume-price impact, return×volume shock, peer advance breadth and peer volume-weighted advance. Rolling statistics only use current and earlier completed bars; relative-volume reference windows use *prior* bars.

All feature families use the same eligible observations. Missing volume is imputed using training-only medians, with missingness indicators. The notebook retains the q=7 labels `[2,2,0,0,1,1,1]`, and reports balanced vs unweighted model training explicitly. The reference model comparison is **not** a nested model-selection test: choosing a winner from these development-fold results requires a separate future prospective shadow test. The threshold .50 leaderboard is a predeclared reference, not a post hoc optimal trading threshold.

**Interpretation:** The target Up class covers 3/7 of the distribution; class-1 precision must be compared against the observed Up prevalence, not 33%. Also report entry coverage, mean gross next-bar return, and net return after assumed 10 bps per entry. These are *single-bar return proxies*, not a portfolio backtest. There is no liquidity, slippage, fill, capital, or position-sizing simulation.

**Causality and data warnings:** All current-bar OHLCV features are available only *after* the bar closes. Intraday predictions must not use them before close. Historical split-adjusted volume and adjusted-price/raw-price mixtures can distort volume signals. Market breadth uses the available eligible ticker universe, not a complete point-in-time S&P 500 constituent set; survivorship bias remains. Peer breadth uses same-day peers, so it must not be used before all peers' bars have completed. Cross-sectional peer features must be constructed only from the information available by the prediction timestamp. No news/NLP data is ingested here.

## Run

Open `notebooks/26_volume_flow_regime_research.ipynb` from the repository or notebooks directory. Run the tests `python -m pytest tests/test_volume_regime_research.py -q`. Run the notebook; reports are saved to `artifacts/volume_regime/1d/`. If it is too slow, first run with `CONFIGS` restricted to two model configurations, but report the restriction when sharing results.
