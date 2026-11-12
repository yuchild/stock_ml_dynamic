import numpy as np
import pandas as pd
import pytest
from stock_ml_dynamic.decision_research import CANDIDATES_25,THRESHOLDS_25,summarize_policies,candidate_stability

def test_candidate_names_unique():
    assert len(CANDIDATES_25)==18
    assert len({c.name for c in CANDIDATES_25})==18
    assert any(not c.balanced for c in CANDIDATES_25)
    assert any(c.balanced for c in CANDIDATES_25)

def test_thresholds_sorted():
    assert tuple(sorted(set(THRESHOLDS_25)))==THRESHOLDS_25

def test_gate_rejects_bad_economics():
    rows=[dict(policy='entry',outer_fold=i,accuracy=.4,majority_baseline_accuracy=.42,precision_lift=.05,coverage=.1,mean_entry_net_return=-.001) for i in range(1,4)]
    result=summarize_policies(pd.DataFrame(rows))
    assert not bool(result.iloc[0].entry_gate_pass)

def test_gate_requires_all_folds_coverage():
    rows=[dict(policy='entry',outer_fold=i,accuracy=.4,majority_baseline_accuracy=.42,precision_lift=.05,coverage=(.01 if i==3 else .1),mean_entry_net_return=.002) for i in range(1,4)]
    assert not bool(summarize_policies(pd.DataFrame(rows)).iloc[0].entry_gate_pass)

def test_candidate_stability_deduplicates_threshold_rows():
    rows=[]
    for fold in [1,2]:
        for threshold in [.4,.5]:
            rows.append(dict(outer_fold=1,inner_fold=fold,candidate='x',threshold=threshold,accuracy=.3+fold*.1,roc_auc=.55,balanced_accuracy=.35,fit_seconds=2.))
    result=candidate_stability(pd.DataFrame(rows))
    assert int(result.iloc[0].folds)==2
    assert np.isclose(result.iloc[0].mean_accuracy,.45)

def test_empty_data_rejected():
    with pytest.raises(ValueError):summarize_policies(pd.DataFrame())
    with pytest.raises(ValueError):candidate_stability(pd.DataFrame())
