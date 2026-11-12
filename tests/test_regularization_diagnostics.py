import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from stock_ml_dynamic.regularization_diagnostics import inner_fold_for_outer, sample_rows, metrics, summary_regularization
from stock_ml_dynamic.walk_forward import Fold

def test_inner_fold_precedes_outer():
    times=pd.date_range('2025-01-01',periods=150,freq='D',tz='UTC')
    df=pd.DataFrame({'bar_timestamp':times})
    outer=Fold(1,times[119],times[140])
    inner=inner_fold_for_outer(df,outer)
    assert inner.train_end < inner.test_end == outer.train_end

def test_invalid_inner_fraction():
    with pytest.raises(ValueError):inner_fold_for_outer(pd.DataFrame(),Fold(1,None,None),fraction=.9)

def test_sample_rows_aligned_and_deterministic():
    x=np.arange(300).reshape(100,3);y=np.arange(100)
    a,b=sample_rows(x,y,20,42)
    c,d=sample_rows(x,y,20,42)
    assert len(a)==len(b)==20 and np.array_equal(a,c) and np.array_equal(b,d)
    assert np.array_equal(a[:,0]//3,b)

def test_metrics_three_class():
    x=np.arange(30).reshape(10,3)
    y=np.array([0,1,2,0,1,2,0,1,2,0])
    m=DummyClassifier(strategy='prior').fit(x,y)
    report,prob=metrics(y,m,x)
    assert prob.shape==(10,3) and np.isfinite(report['log_loss'])

def test_summary():
    frame=pd.DataFrame([{'model':'a','accuracy':.4,'balanced_accuracy':.38,'roc_auc':.55,'log_loss':1.,'fit_seconds':2.},
                        {'model':'a','accuracy':.5,'balanced_accuracy':.48,'roc_auc':.56,'log_loss':.9,'fit_seconds':3.}])
    assert summary_regularization(frame).iloc[0]['total_fit_seconds']==5
