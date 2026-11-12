import pandas as pd
import pytest
from stock_ml_dynamic.robustness_diagnostics import paired_differences,summarize_repeats

def sample():
    return pd.DataFrame([{'seed':seed,'fold':fold,'model':model,'roc_auc':auc,'accuracy':.4,'balanced_accuracy':.38,'log_loss':1.08,'fit_seconds':1.,'predict_seconds':.1} for seed in (42,52) for fold in (1,2) for model,auc in [('random_forest',.56),('soft_vote_lr_rf',.57),('logistic_l2',.55)]])

def test_paired_differences():
    out=paired_differences(sample()); row=out.set_index('model').loc['soft_vote_lr_rf'];assert row.n_paired_runs==4;assert row.mean_delta==pytest.approx(.01)

def test_duplicate_rejected():
    data=sample();with_duplicate=pd.concat([data,data.iloc[:1]],ignore_index=True)
    with pytest.raises(ValueError,match='Duplicate'):paired_differences(with_duplicate)

def test_summary():
    summary=summarize_repeats(sample());assert summary.iloc[0]['model']=='soft_vote_lr_rf'

def test_missing_reference():
    with pytest.raises(ValueError,match='Reference'):paired_differences(sample(),reference='missing')
