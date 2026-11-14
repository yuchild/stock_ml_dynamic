# Notebook 40 — Source-of-edge forensic laboratory

Open `notebooks/40_source_of_edge_forensics.ipynb` and run all cells using the project `.venv` kernel.

**Data inputs**: existing `artifacts/master_tables/1d/` (required), and Notebook 39's saved `artifacts/edge_discovery/1d/{trades,metrics,daily}.csv` (optional but strongly recommended for frozen-trade analysis). Notebook 39 is not retrained by Notebook 40.

**Coverage**: 3 expanding chronological development folds; Ridge α100 all features, each isolated feature group, each leave-one-group-out model, top 10/30/60 training-selected coefficients, plus up to 8 observable-factor-neutralized scores; each evaluated at top 5/10/20. Frozen Notebook 39 trades are audited for year/month/symbol concentration, largest winners and losers, equity/drawdowns, payoff ratio, and tails. Includes paired 10-day moving-block confidence intervals and a joint max-statistic multiple-comparison diagnostic. Results saved under `artifacts/edge_sources/1d/`.

**Caveats**: Feature-group assignment is an approximate keyword taxonomy; check uncategorized features and categorical dummy labels. The neutrality experiment removes linear exposure to observable same-day feature proxies, not to a fully specified risk-factor model. Historical sector-neutrality cannot be claimed without point-in-time sector mapping. The joint reality check is an exploratory diagnostic on reused development folds, not an independent confirmatory test. Zero transaction costs are an explicit research assumption. Close-to-close returns may not be executable. The final historical holdout and prospective ledger are not accessed.

**Validation**: `PYTHONPATH=src python -m pytest tests/test_edge_source_research.py -q`.
