# Notebook 43 — asymmetric tail risk and expected shortfall proxies

Run `notebooks/43_asymmetric_tail_risk_and_expected_shortfall.ipynb` with the project `.venv` kernel after the usual master-table artifacts exist. Preserves existing `artifacts/` when upgrading.

83 numbered analyses across: strict baseline reconciliation, six tail thresholds/directions, logistic and HGB classifiers, seven conditional quantiles, 20+ ex-ante ranking strategies, chronological three-fold comparisons, probability reliability, drawdown, 2:1 payoff screening, 10-day paired bootstrap, tail dependence, and realized downside severity. Zero costs. All fit transformations use development training only; no historical final holdout or prospective ledger.

**Important:** Seven independently fitted quantile models may cross. Expected shortfall reported here is *realized* tail severity on validation portfolios, not a calibrated ex-ante ES forecast. Model search is exploratory and bootstrap intervals are not familywise-adjusted.
