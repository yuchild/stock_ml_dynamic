# Notebook 38 — Zero-cost payoff ratio and broad model search

Run `notebooks/38_payoff_ratio_model_search.ipynb` after ensuring `artifacts/master_tables/1d` contains the four local master-table artifacts.

Eight trained models (24 fits), two controls, three position sizes, three chronological folds, 90 portfolio comparisons. The final historical 20% holdout is excluded. Costs are explicitly zero. The observed 2:1 payoff ratio is not a simulated 2R take-profit/1R stop. The oracle uses future returns and is diagnostic only.

Outputs are saved under `artifacts/payoff_search/1d/`. Existing `artifacts/` must be preserved when updating the repository.
