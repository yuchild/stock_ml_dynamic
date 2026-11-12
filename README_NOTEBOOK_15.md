# Notebook 15 — Prospective operations and tracking

Notebook 15 is housekeeping only. It does not train ensembles or neural networks; those belong in Notebook 16.

1. Run `python -m pytest -q` and open `notebooks/15_prospective_operations.ipynb` in VS Code.
2. Leave `COMMIT_NEW_PREDICTIONS=False` for the first run. Inspect source freshness, the frozen model path, and any prior ledger outcomes. Confirm the source's daily timestamp convention and that the exchange session and ingestion are complete.
3. Only for a genuinely new, completed session with no known next bar, set `COMMIT_NEW_PREDICTIONS=True`, `OPERATOR_CONFIRMED=True`, and `CONFIRMED_SESSION_DATE='YYYY-MM-DD'`. Run again; duplicate keys are skipped by the ledger.

Outputs: `artifacts/prospective/1d/prediction_ledger.parquet` and timestamped `artifacts/prospective/1d/runs/<run_id>/` reports (source/ledger audits, prospective scores, per-class recall, ROC plots when sufficient outcomes exist).

**Safety limitations:** The source bar timestamp may be a session-date label, not a close timestamp; the explicit human confirmation is essential. The ledger is single-writer only, and reconciliation assumes the first later bar is the next interval. A holiday calendar and transactional database are not implemented. Do not treat retrospective or stale predictions as prospective performance. A model artifact must be loaded only from trusted local files.

### v14.1 fix
An empty prospective ledger is expected on the first run. Monitoring now returns empty, schema-defined score tables rather than raising `KeyError: 'actual_direction'`. No change to model training or prediction recording.
