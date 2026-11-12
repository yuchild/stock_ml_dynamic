import pandas as pd
import pytest
from stock_ml_dynamic.temporal_stability import validate_scores, paired_fold_deltas, summarize_pairs, fold_winners


def sample():
    rows=[]
    for fold in (1,2,3):
        for model,auc,ll in [('random_forest',.55,.99),('temporal_stack_calibrated',.56,.98)]:
            rows.append(dict(fold=fold,model=model,roc_auc=auc,accuracy=.4,balanced_accuracy=.39,log_loss=ll,ece=.02))
    return pd.DataFrame(rows)


def test_positive_improvements_and_loss_direction():
    d=paired_fold_deltas(sample(),'temporal_stack_calibrated')
    assert len(d)==3
    assert (d.improvement_roc_auc>0).all()
    assert (d.improvement_log_loss>0).all()


def test_pair_summary_counts():
    s=summarize_pairs(sample())
    assert s.iloc[0].positive_folds_roc_auc==3
    assert s.iloc[0].n_folds==3


def test_mismatched_folds_rejected():
    data=sample().drop(index=0)
    with pytest.raises(ValueError,match='fold sets differ'):
        paired_fold_deltas(data,'temporal_stack_calibrated')


def test_duplicate_rows_rejected():
    data=pd.concat([sample(),sample().iloc[[0]]],ignore_index=True)
    with pytest.raises(ValueError,match='Duplicate'):
        validate_scores(data)


def test_fold_winners():
    w=fold_winners(sample())
    assert len(w)==3
    assert set(w.model)=={'temporal_stack_calibrated'}
