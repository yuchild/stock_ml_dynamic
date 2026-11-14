import numpy as np
import pandas as pd
from stock_ml_dynamic.tail_edge_42 import _event,_safe_auc,event_calibration,tail_concentration,compare_to_ridge

def test_event_threshold_is_strict():
    assert _event(np.array([.03,.031,-.1]),.03).tolist()==[0,1,0]

def test_auc_constant_labels_is_nan():
    assert np.isnan(_safe_auc(np.array([0,0,0]),np.array([.1,.2,.3])))

def test_calibration_fixed_bins():
    f=pd.DataFrame({'fold':[1,1,1],'next_return':[.06,-.02,.07],'up_5pct_logistic':[.8,.1,.7],'up_5pct_hgb':[.9,.2,.6]})
    x=event_calibration(f,bins=10)
    assert x.n.sum()==6
    assert (x.observed.between(0,1)).all()

def test_tail_concentration_is_diagnostic():
    f=pd.DataFrame({'fold':[1]*4,'model':['ridge_all']*4,'k':[5]*4,'next_return':[.1,.02,-.01,-.02]})
    x=tail_concentration(f,top_fractions=(.25,))
    assert len(x)==1 and np.isclose(x.iloc[0].top_sum,.1)

def test_paired_comparison_preserves_k():
    dates=pd.date_range('2023-01-01',periods=25)
    x=pd.DataFrame([{'fold':1,'model':m,'k':5,'bar_timestamp':d,'daily_return':v} for d in dates for m,v in [('ridge_all',.01),('challenger',.02)]])
    r=compare_to_ridge(x,reps=50)
    assert len(r)==1 and r.iloc[0]['k']==5 and np.isclose(r.iloc[0].mean_lift,.01)
