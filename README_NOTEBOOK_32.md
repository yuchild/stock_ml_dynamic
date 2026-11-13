# Notebook 32: Chronological Four-Deck Sequential Random Forest Stacking

This is an experimental implementation of the proposed four-deck training regimen. It **does not randomly shuffle dates**: random shuffling would allow later market regimes and correlated observations to influence predictions for earlier dates. Instead, four consecutive nonoverlapping global time decks are constructed within each outer-fold training window.

- Stage 1: fit on deck 1; predict probabilities on decks 2–4 and the outer validation window.
- Stage 2: fit on deck 2 using original predictors and stage-1 out-of-sample probabilities; predict on decks 3–4 and validation.
- Stage 3: fit on deck 3 using stage-1 and stage-2 out-of-sample probabilities; predict on deck 4 and validation.
- Stage 4: fit on deck 4 using stage-1/2/3 probabilities; evaluate on a subsequent, unseen chronological outer fold.

Each model is **freshly trained**, with independent, training-only 0.5%/99.5% winsorization and median imputation. The added columns `model1_p0`–`model3_p2` are **probabilities**, not in-sample fitted labels. The hard-label columns `model1_pred` etc. are also created for inspection, but the next model uses the probability columns.

The original Q7 target labels (2 down, 0 middle, 1 up) are fixed from deck 1's training returns within each outer fold. We compare the cascade against a deck-4-only RF and a conventional all-training RF. These are development-fold comparisons; the previously closed final 20% holdout is **not** used. A future latest-bar inference workflow should be separately implemented and audited before any operational use.

The current code uses 200 trees per model, leaf size 8, max depth 12, and training cap 60,000 rows; set notebook constants to increase workload. Three outer folds require 18 RF fits (4 sequential + 2 baselines each). The study is deliberately diagnostic, not an automated hyperparameter selection process.

Run `python -m pytest tests/test_sequential_deck_stacking.py -q` from the repository root. Notebook path: `notebooks/32_sequential_four_deck_stacking.ipynb`.
