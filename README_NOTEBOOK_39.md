# Notebook 39: Economic Edge Discovery

A concentrated search across 11 regression configurations, three expanding chronological outer folds, three daily portfolio capacities, 1,000 date-matched random portfolio draws per fold/capacity, training-only feature screening, market-relative returns, win/loss economics, upper-tail robustness, and date-block paired confidence intervals.

**Run:** `notebooks/39_economic_edge_discovery.ipynb` from the repository root or `notebooks/` directory with the existing `.venv` kernel.

**Inputs:** existing `artifacts/master_tables/1d` files, created by Notebook 03. **Outputs:** `artifacts/edge_discovery/1d/*.csv`. **Costs:** 0 by user request. **Historical final holdout:** excluded by the 20% development split; no prospective ledger access.

**Interpretation:** the 2:1 target refers to the observed average winning return divided by the absolute average losing return, *not* a 2R stop/target execution simulation. The date-matched random null accounts for the daily market opportunity set but not for the entire model-search multiple-testing problem. `market_excess_daily` is a same-date benchmark-relative selection diagnostic, not an executable long-short backtest. Treat top-ranked results as discovery candidates pending independent validation.

**Compute:** 33 fitted models (11 per outer fold), 99 portfolio evaluations, 9 random-control distributions, 1,000 moving-block bootstrap repetitions for paired comparisons. Huber and ElasticNet may take time. Reduce `TRAIN_CAP` for a smoke run only; do not compare smoke results to full-history research.
