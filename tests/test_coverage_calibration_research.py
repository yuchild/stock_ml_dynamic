import numpy as np
import pandas as pd
from stock_ml_dynamic.coverage_calibration_research import (
    chronological_calibration_split, calibrate_up, threshold_for_coverage,
    policy_metrics, reliability_table,
)

def example(n=600):
    d=pd.date_range('2024-01-01',periods=n,freq='D',tz='UTC')
    return pd.DataFrame({'bar_timestamp':d,'next_bar_timestamp':d+pd.Timedelta(days=1),
                         'direction':np.arange(n)%3,'next_return':np.arange(n)/10000})

def test_split_purges_label_overlap():
    a,b=chronological_calibration_split(example())
    assert a.next_bar_timestamp.max()<b.bar_timestamp.min()
    assert len(a)+len(b)<600

def test_threshold_training_only():
    x=np.arange(100)/100
    t=threshold_for_coverage(x,.10)
    assert .88<t<.91
    assert np.mean(x>=t)==.10

def test_platt_probability_and_finite():
    x=np.linspace(.05,.95,120)
    y=(np.arange(120)%3==1).astype(int)
    z,w=calibrate_up(x,y,np.array([.1,.5,.9]))
    assert z.shape==(3,) and w.shape==(120,)
    assert np.isfinite(z).all() and ((z>=0)&(z<=1)).all()

def test_metrics_and_reliability():
    f=example()
    scores=np.linspace(.01,.99,len(f))
    m=policy_metrics(f,scores,.90)
    assert m['entry_count']>0
    assert 0<m['coverage']<.2
    bins=reliability_table(f,scores,fold=1,model='m',subset='raw')
    assert sum(x['n'] for x in bins)==len(f)
