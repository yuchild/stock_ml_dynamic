# Notebook 13 — Final holdout evaluation

The frozen model is multinomial LogisticRegression L2, `C=0.10`, with the original next-bar training-only return tertile labels and expanded historical/peer-market features excluding `realized_vol_10` (keeping 5/20 bar volatility). The newest 20% of globally ordered timestamps is held out of development.

Run `notebooks/13_final_holdout_evaluation.ipynb` once after Notebook 12. Outputs are in `artifacts/final_holdout/1d/`. The evaluator refuses to overwrite existing final metrics by default. No hyperparameter selection or feature selection uses final holdout outcomes.

Important: the benchmark uses up to 120,000 randomly sampled training observations and all complete holdout rows. This is a model evaluation, not a backtest or trading-performance assessment. Contemporaneous peer-market features require the same-timestamp peer bars to be available when inference occurs. Class labels are relative to the training return distribution, not literal zero-return movements.
