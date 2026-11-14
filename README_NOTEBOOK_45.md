# Notebook 45 — Point-in-time integrity and frozen-signal sensitivity

Run `notebooks/45_point_in_time_integrity_and_validation.ipynb` from VS Code with `.venv`. Retain `artifacts/master_tables/1d/`. This notebook does not open the historical holdout or modify the prospective ledger. It uses zero transaction costs.

**Limitations:** Source-level corporate-action accuracy, historical index membership, delisted names, and vendor revision histories require external datasets; they cannot be certified by internal consistency checks. All extreme-return masking is a retrospective sensitivity analysis, not a trading filter.
