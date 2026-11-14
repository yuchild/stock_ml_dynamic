# Notebook 37 — Target alignment

Open `notebooks/37_accuracy_vs_profitability_targets.ipynb` and run all cells after ensuring `artifacts/master_tables/1d/` contains the four master-table artifacts. The notebook uses three chronological development folds, excluding the final historical 20% holdout. It compares three-class direction RF, next-bar return regression, and binary net-positive classification with fixed predeclared thresholds and an equal-capital daily 10-slot research policy. Results are written to `artifacts/target_alignment/1d/`.

**Caveats:** Close-to-close adjusted returns may not be directly tradable; fixed costs do not represent actual fills; labels are fitted within folds; models are not optimized on validation; the experiment does not establish a live trading edge. Keep locally generated `artifacts/` when extracting the ZIP.
