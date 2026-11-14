# Notebook 44: Factor-matched alpha versus volatility

Run `notebooks/44_factor_matched_alpha_vs_volatility.ipynb` after preserving `artifacts/master_tables/1d/`. It reconstructs the frozen Ridge baseline using the Notebook 41 training pipeline, then compares selected stocks to same-date factor-matched random controls. It evaluates volatility and volume strata, fixed volatility-balanced selection, influential-stock exclusions, moving-block intervals, and audit outputs.

All results are development-only, zero-cost, exploratory and not multiple-testing adjusted. Historical final holdout and prospective ledger remain closed.

Outputs: `artifacts/factor_matched/1d/`. Matching can be computationally intensive (300 randomized portfolios per date at k=5, 150 at k=10/20).
