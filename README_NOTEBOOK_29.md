# Notebook 29 — Deep Random Forest Research

Start with `notebooks/29_deep_random_forest_research.ipynb`. This notebook uses the existing master artifacts in `artifacts/` and does not regenerate the master table. The executed Notebook 28 motivated its robustness checks: the entry policy produced 3,187 / 44 / 321 signals across outer folds and negative mean returns in the first fold.

Default experiment: 36 Random Forest configurations × 2 feature sets × 2 sampling methods × 2 inner folds × 3 outer folds = **864 inner fits**; up to 6 outer refits. 200 trees per fit, 60k row cap. Adjust `CONFIGS`, `TRAIN_CAP`, `N_ESTIMATORS`, `SAMPLING_MODES` to control runtime. Models fit with `n_jobs=-1`, so memory/CPU load may be significant; monitor RAM and temperatures. Threshold testing reuses predictions, so does not multiply model fits.

The three outer folds and nested inner folds are chronological, with the existing 20% development/holdout split. Never tune further based on outer outcomes and claim unbiased generalization. Cost sensitivity and day-aggregated returns are diagnostics, not an executable trading strategy. All outputs saved under `artifacts/rf_deep_research/1d`. No historical final holdout or prospective prediction ledger modifications.
