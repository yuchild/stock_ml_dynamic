# Notebook 17 — Residual Neural Network and Temporal Ensembles

Development-only five-fold chronological comparison of Logistic L2, Random Forest, custom residual PyTorch MLP, fixed soft-voting, and temporal stacking. Output: `artifacts/architecture_ensemble/1d/`.

Start with `N_FOLDS=2`, `NEURAL_EPOCHS=8`, `INCLUDE_STACKING=False` for a smoke test. Full run: 5 folds, 35 epochs, stacking on. Stacking uses inner chronological OOF model predictions; outer-fold preprocessing is shared with inner folds, so inner validation preprocessing is not fully nested. Do not use for production selection without addressing this limitation.

Run `python -m pytest -q tests/test_architecture_ensemble.py`. CUDA available automatically for custom neural model. Notebook does not access PostgreSQL directly, reopen the final holdout, or write to prospective ledger.
