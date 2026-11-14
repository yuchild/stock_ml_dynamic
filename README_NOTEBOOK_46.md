# Notebook 46 — Provenance and leakage validation

Run `notebooks/46_provenance_and_leakage_validation.ipynb` using the project venv. The notebook reads the existing master artifacts, records source fingerprints, inspects source code, checks target alignment, lists extreme-return events and generates an evidence gate. Local PostgreSQL reconciliation is opt-in via `RUN_LOCAL_DB_AUDIT=True`. No final holdout inspection or new model search. Results: `artifacts/provenance_validation/1d/`.
