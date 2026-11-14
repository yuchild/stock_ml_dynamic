import pandas as pd

def test_decision_merge_preserves_both_fold_counts():
    leader = pd.DataFrame({"model": ["ridge_all"], "k": [5], "positive_folds": [3], "market_excess": [.001], "profit_factor": [1.2]})
    paired_summary = pd.DataFrame({"model": ["ridge_all"], "positive_folds": [1]})
    decision = leader.loc[leader["k"].eq(5)].merge(paired_summary.rename(columns={"positive_folds": "positive_lift_folds"}), on="model", how="left", validate="one_to_one")
    assert decision["positive_folds"].iloc[0] == 3
    assert decision["positive_lift_folds"].iloc[0] == 1
    assert bool((decision["positive_folds"].eq(3) & decision["market_excess"].gt(0) & decision["profit_factor"].gt(1)).iloc[0])
