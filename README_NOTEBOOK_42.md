# Notebook 42 — Tail events and asymmetric return modeling

Open `notebooks/42_tail_event_and_asymmetric_returns.ipynb` using the existing `.venv` kernel. It reconstructs the Notebook 41 Ridge α=100 baseline first, then trains +3%, +5%, +10% tail-event classifiers (logistic and histogram boosting), q10/q50/q90 conditional quantile regressors, and several deterministic ex-ante combinations. All preprocessing and fitting use the development training folds only. It evaluates 5/10/20-stock portfolios with zero transaction costs, paired block bootstraps, payoff ratios, calibration, drawdowns, and return concentration.

**Required:** preserve the existing `artifacts/master_tables/1d/` folder when extracting. Results export to `artifacts/tail_edge/1d/`. Notebook 39 and 41 historical metrics should not be compared as if their pipelines were identical; this notebook includes a hard assertion against Notebook 41's re-run Ridge baseline.

The notebook does not access the final historical holdout or the prospective prediction ledger. Historical outcomes are used for evaluation, not for ex-ante portfolio selection. This is exploratory research, not trading advice.
