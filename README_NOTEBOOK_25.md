# Notebook 25 — Expanded decision research

Open `notebooks/25_expanded_decision_research.ipynb` after extracting this repository into your local project.

Run all cells with the existing Python environment. Three outer chronological folds, two inner folds, 18 predeclared configurations, seven thresholds, q=7 labels `[2,2,0,0,1,1,1]`. Balanced classifiers are default; unweighted controls are clearly labeled.

Notebook 24 found 41.05% mean accuracy against a 41.96% majority baseline, and the entry policy had negative estimated mean returns. This notebook asks whether a broader, predeclared search produces consistent improvements.

Outputs: `artifacts/decision_research/1d/` (inner grid, outer policy evaluation, choices, decision summary, candidate stability, manifest). No holdout or prospective-ledger access.

Run `python -m pytest tests/test_decision_research.py -q` first if desired. Model training can be lengthy. Never promote a model solely from these three outer development folds.
