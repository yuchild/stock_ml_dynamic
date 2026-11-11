# Notebook 10 — Volatility diagnostics

Purpose: quantify which realized-volatility lookback windows (5, 10, 20 bars) improve next-bar direction classification, without changing the training-fold return-tertile labels.

Run `python -m pytest -q`, then execute `notebooks/10_volatility_window_diagnostics.ipynb` in the repository's existing virtual environment after Notebook 03 has generated the master artifacts.

Five chronological expanding folds are restricted to the oldest 80% of timestamps. The most recent 20% is **never evaluated** by this experiment. For each fold, targets, KMeans, scaling, and one-hot encoding are trained using past observations only. All eight feature configurations share identical complete observations and sampled training indices. Volatility regime boundaries are fitted from training-fold 20-bar volatility. The benchmark records accuracy, balanced accuracy, multiclass macro OVR ROC-AUC, log-loss, ROC curves, volatility-regime scores, class prevalence, and training/prediction seconds.

Outputs: `artifacts/volatility_diagnostics/1d/` at the repository root. No random forests, neural networks, regressors, or alternative labels are trained.

**Limitations:** Current targets are fixed raw-return tertiles; improvements from volatility may reflect return-magnitude regimes rather than directional predictability. The peer market return assumes all same-timestamp peers have completed bars; no live data readiness check is performed. Do not interpret CV selection scores as final untouched-test performance. Running a different target definition requires a separate, carefully designed experiment.
