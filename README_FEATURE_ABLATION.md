# Notebook 09: feature-group ablation and locked holdout

Run notebooks/09_feature_ablation_holdout.ipynb after Notebook 03 has saved master artifacts. The notebook finds root or notebooks/artifacts automatically.

Five expanding folds are constructed **only** from the oldest 80% of distinct timestamps. The newest 20% is untouched. For each fold, target thresholds, K-means, original scaling/encoding, and enhanced-feature scaling are fitted using training data only. Each feature scheme and model uses the same complete observations for paired comparison.

Feature schemes: original, all expanded, and removal of momentum, volatility, trend/position, volume, or peer-market features. Models: prior-class baseline, logistic L1, logistic L2. No regressors, trees or neural networks.

Outputs: artifacts/feature_ablation/1d/development_fold_metrics.csv, development_fold_metrics.parquet, development_roc_curves.parquet, ablation_accuracy.png, ablation_roc.png, holdout_lock.json.

**Important:** Select a single configuration from development CV before evaluating the final holdout in a separate later notebook. Repeated tuning against the same development folds can still overfit model selection. The peer market proxy uses contemporaneous same-timestamp returns; verify peers are observable by prediction time. Eligibility based on latest history creates retrospective survivorship bias. The saved master table's latest period should be checked for any incomplete bars.
