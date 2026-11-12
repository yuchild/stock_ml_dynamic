# Notebook 21 — Accuracy-first feature engineering

This is a development-only experiment. It does not touch the final historical holdout or prospective ledger.

New causal OHLCV features: rolling return standard deviation (5/10/20/40/60), downside volatility, rolling mean return, volatility-adjusted return, rolling skewness, lagged-normalized return z-scores, drawdowns, and prior-volume-relative volume.

The experiment compares existing historical features against volatility-only and all expanded features on **the same complete-case observations** using three classifiers and expanding chronological folds. Accuracy is the primary ranking criterion; balanced accuracy, ROC-AUC, log-loss and selective prediction accuracy are supporting diagnostics. Threshold selection is exploratory and requires a separate nested validation process before deployment.

First run smoke mode, then full five-fold mode. Larger feature windows reduce the sample size; comparisons are paired, but are not directly comparable to previous experiments that used different samples. The notebook does not promise a profitable strategy.
