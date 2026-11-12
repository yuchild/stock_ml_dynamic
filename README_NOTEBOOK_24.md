# Notebook 24 — Comprehensive model-search decision dashboard

This notebook responds to the need for fewer, more substantial experiments and an interpretable model-selection framework.

## Research questions

1. Can any predeclared candidate beat the majority-class accuracy baseline across three unseen development periods?
2. Can a separately selected class-1-only entry rule improve on the unconditional Up frequency at **at least 5% coverage per inner fold**?
3. Does the simple next-bar return proxy remain positive after an illustrative 10-bp cost assumption?
4. Do expanded volatility/volume features help relative to the baseline feature set?

## Search

Ten predefined candidate configurations: Logistic L2 (baseline/all/volatility, plus unweighted diagnostic), Random Forest (two leaf sizes plus volatility), Extra Trees, and histogram gradient boosting (all/volatility). Default balanced training; histogram boosting receives balanced sample weights. Six class-1 thresholds from 0.35 to 0.60.

Three expanding **outer** folds, each containing two expanding **inner** folds. All fitted quantile boundaries, candle clustering, and scaling are trained within each fold. Outer outcomes are used for reporting only. The existing 20% historical final holdout is excluded.

Expected 60 inner model fits plus at most six outer fits. Expect a lengthy CPU-intensive run; the neural GPU is not needed. Do not interrupt while an outer fold is running. The experiment uses existing master artifacts and does not change database contents.

## Run

1. Extract this complete repository into the existing project.
2. Run `python -m pytest tests/test_comprehensive_search.py -q` in the project virtual environment.
3. Open `notebooks/24_comprehensive_model_search.ipynb` and Run All.

Outputs: `artifacts/comprehensive_search/1d/{inner_grid.csv,outer_policy_evaluation.csv,inner_policy_choices.csv,inner_model_rankings.csv,inner_threshold_rankings.csv,manifest.json}`.

## Limitations

Class 1 is a relative-return septile category, not a guaranteed positive net return. Accuracy and precision are distinct from profitability. Cross-sectional samples are correlated, so do not treat all stock-day observations as independent statistical evidence. No realistic fills, spreads, slippage, capital allocation, position overlap, or corporate-action execution handling are modeled. Results cannot be treated as investment advice or a live trading signal. Keep the existing prospective ledger and old frozen holdout untouched.
