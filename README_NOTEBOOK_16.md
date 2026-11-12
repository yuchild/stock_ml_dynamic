# Notebook 16 — nonlinear ensemble and custom PyTorch classifier benchmark

Run `notebooks/16_nonlinear_ensembles_custom_neural.ipynb` after Notebook 03 master artifacts exist. Install `pip install -e ".[ml,notebook,neural,dev]"`. Five chronological development folds; final 20% holdout and prospective ledger remain untouched.

Fixed candidates: prior baseline, Logistic L2 C=0.1, Random Forest (160 trees), Extra Trees (160 trees), histogram gradient boosting (150 rounds), and custom PyTorch two-hidden-layer MLP (128/64, dropout .2, AdamW, 35 epochs max, patience 6). Feature and target transforms are fitted separately within each outer fold. Neural early stopping uses a chronological slice of outer-training rows. Results under `artifacts/nonlinear_benchmark/1d/`.

First smoke test: set N_FOLDS=2, NEURAL_EPOCHS=8; then restore defaults. Do not use final holdout to select a winner.
