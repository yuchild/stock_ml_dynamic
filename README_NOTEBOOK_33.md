# Notebook 33: chronological out-of-fold Random Forest stacking

Open `notebooks/33_chronological_oof_stacking.ipynb` with the repository `.venv` kernel.

Tests: `python -m pytest tests/test_oof_stacking_research.py -q`.

Compares full-history Random Forest with OOF probability meta-models (Logistic Regression and shallow RF) and an RF augmented with OOF probability columns. Five expanding OOF windows, three outer chronological folds, fixed q7 target. The final 20% holdout is never scored; no prospective ledger writes.

Expected compute: 3 outer folds × (5 OOF base RF + 1 full-history base RF + 1 augmented RF + 1 shallow RF meta) = 24 RF fits, plus three Logistic meta fits. 200 trees each except shallow meta RF also 200.
