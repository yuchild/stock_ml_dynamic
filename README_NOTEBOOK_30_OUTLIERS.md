# Notebook 30 v29.2: robust feature-space EDA

Run `notebooks/30_feature_space_360_eda.ipynb` from the project virtual environment. Install `seaborn` if missing.

- Train-only quantile winsorization of engineered numerical predictors (default 0.5th/99.5th percentile).
- Original target, next returns, OHLCV, and sector/stock return correlations are preserved.
- Five-bin raw vs winsorized charts for 24 most affected predictors and five-bin histograms for every transformed numeric variable.
- Before/after L1 and L2 multinomial logistic regression and Ridge return regression on the same chronological split.
- Not a new holdout evaluation or proof of improved forecasting. Strong outliers can contain important information; compare both variants under future nested CV.
