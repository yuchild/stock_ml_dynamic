import numpy as np
import pandas as pd
from stock_ml_dynamic.asymmetric_edge_43 import safe_auc,fit_binary,temporal_calibrate,event_reliability,paired_comparison,concentration

def test_binary_constant_class():
    x=np.ones((8,2));z=np.ones((3,2))
    for kind in ('logistic','hgb'):
        assert np.array_equal(fit_binary(x,np.zeros(8,dtype=int),z,kind,42),np.zeros(3))

def test_temporal_calibration_constant():
    assert np.allclose(temporal_calibrate([.1,.2],[0,0],[.4,.5]),0)

def test_reliability_has_both_tails():
    x=pd.DataFrame({'fold':[1,1],'next_return':[.06,-.06]})
    for t in (3,5,10):
        for side in ('up','down'):
            for kind in ('logistic','hgb'):
                x[f'{side}_{t}_{kind}']=[.2,.8]
    a=event_reliability(x)
    assert {'up_5_hgb','down_5_hgb'}.issubset(set(a.model))

def test_concentration_and_paired():
    dates=pd.date_range('2024-01-01',periods=25,tz='UTC')
    daily=pd.DataFrame([dict(fold=1,model=model,k=5,bar_timestamp=d,daily_return=r) for model,r in [('ridge_all',.01),('challenger',.02)] for d in dates])
    p=paired_comparison(daily,reps=30)
    assert p.iloc[0]['mean_lift']>.009
    t=pd.DataFrame([dict(fold=1,model='ridge_all',k=5,next_return=r) for r in [.1,-.03,.02]])
    assert len(concentration(t))==1
