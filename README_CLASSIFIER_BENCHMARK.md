# Notebook 07: Classification-only model comparison

Run `notebooks/07_classifier_feature_selection.ipynb` after Notebook 03 has saved `artifacts/master_tables/1d`.

The notebook excludes entire ticker/interval histories with fewer than 31 bars or an invalid latest 30-prior-bar z-score. The exclusion happens in the classifier benchmark dataset; it does not rewrite the existing persisted master table, which can be regenerated later with this eligibility rule.

Five expanding-window chronological folds, train-only quantiles, candle KMeans, one-hot/scaling, feature reduction and classifiers. Model families: prior baseline, L1/L2 logistic regression, random forest, Gaussian Naive Bayes, histogram gradient boosting, MLP neural network. Feature schemes: all features, PCA 95% variance, L1 coefficient selection, Ridge top-half coefficient ranking. L2 shrinkage is not automatic elimination.

**Compute:** notebook starts with `FAST_MODE=True` for a 40-run smoke test; set `False` for all 140 configurations. Random forest uses 100 trees, maximum depth 12 and all CPU cores. MLP uses scikit-learn CPU implementation and early stopping; it is a genuine feedforward neural network, not a PyTorch GPU implementation. Up to 120,000 training rows are sampled per fold, while test rows are never sampled. Preprocessing time and per-model fit/predict time are recorded. Progress is printed and metrics checkpointed after each model.

Results saved to repository-root `artifacts/classifier_benchmark/1d/`: `fold_metrics.csv`, `fold_metrics.parquet`, `roc_curves.parquet`, `feature_selection.csv`, `universe_eligibility.csv`, `run_config.json`, `accuracy_comparison.png`, `roc_curves.png`.

**Evaluation:** multiclass macro one-vs-rest AUC; per-fold accuracy, balanced accuracy, log-loss, fit and prediction seconds. Do not select a winner on these folds and report its CV score as unbiased final performance; hold out a later untouched period for final confirmation.
