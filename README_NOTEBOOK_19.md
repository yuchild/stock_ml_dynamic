# Notebook 19 — Probability calibration and leakage-safe temporal stacking

This notebook is **development-only**. It excludes the final 20% historical holdout and never reads or writes the prospective prediction ledger.

## Motivation
Notebook 18's two-seed/two-fold smoke run found mean ROC-AUC 0.566598 for Logistic L2 + Random Forest voting versus 0.565947 for Random Forest. The paired advantage was +0.000651, positive in only 2/4 paired runs. This is not convincing evidence of improvement.

## Experiment
For every outer expanding chronological fold, refit *all* transformations, including candle clustering and target quantiles, within each inner training window. Use the first two inner chronological validation windows for stacking meta-model training, and a subsequent, disjoint inner validation window for temperature calibration. The outer fold's test data is never used to select weights, fit preprocessing, fit stacking, or calibrate.

Compare Logistic L2, Random Forest, uncalibrated 50/50 voting, temperature-calibrated voting, temporal stacking, and temperature-calibrated temporal stacking. Report macro OVR ROC-AUC, accuracy, balanced accuracy, log loss, and confidence ECE. Temperature scaling changes probability sharpness, not argmax decisions or multiclass rank necessarily; evaluate calibration and discrimination separately.

## Run
Open `notebooks/19_temporal_calibration_stacking.ipynb` in VS Code. Start with `SMOKE_TEST=True` (two outer folds, lower row caps), then use `False` for the full five-fold benchmark. This experiment is CPU-heavy because inner preprocessing is refit repeatedly.

Output: `artifacts/temporal_calibration/1d/` (metrics, summary, chronological manifest). No model is deployed by this notebook.

## Caveat
Historical folds and shared stocks are correlated; do not interpret fold-level deltas as independent significance tests. Avoid reopening the previously evaluated final holdout for model selection. Prospective performance remains the decisive next evaluation.
