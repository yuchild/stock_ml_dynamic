import numpy as np
import pandas as pd
import pytest
from stock_ml_dynamic.volume_regime_research import add_volume_regime_features, _metrics, _model

def sample():
    times=pd.date_range('2025-01-01',periods=85,tz='UTC')
    rows=[]
    for symbol,offset in [('AAA',0),('BBB',3)]:
        for i,t in enumerate(times):
            p=100+offset+i*.2
            rows.append(dict(symbol=symbol,interval_code='1d',bar_timestamp=t,adj_close=p,
                close_price=p,high_price=p+1,low_price=p-1,volume=1000+i*10))
    return pd.DataFrame(rows)

def test_no_future_leakage():
    a=sample();b=a.copy();b.loc[b.bar_timestamp==b.bar_timestamp.max(),'volume']=10000000
    x,n=add_volume_regime_features(a);y,_=add_volume_regime_features(b)
    earlier=x.bar_timestamp<x.bar_timestamp.max()
    pd.testing.assert_frame_equal(x.loc[earlier,n],y.loc[earlier,n])

def test_volume_features_exist_and_are_finite_late():
    d,n=add_volume_regime_features(sample())
    assert 'volume_relative_20' in n and 'peer_advance_fraction' in n
    assert np.isfinite(d.loc[d.symbol=='AAA','volume_relative_20'].iloc[-1])

def test_missing_volume_allowed():
    a=sample();a.loc[a.symbol=='AAA','volume']=np.nan
    d,n=add_volume_regime_features(a)
    assert d.loc[d.symbol=='AAA','volume_data_available'].eq(0).all()

def test_duplicate_rejected():
    a=sample();a=pd.concat([a,a.iloc[[0]]],ignore_index=True)
    with pytest.raises(ValueError,match='Duplicate'):add_volume_regime_features(a)

def test_balancing_available():
    assert _model('forest',True,42).class_weight=='balanced'
    assert _model('forest',False,42).class_weight is None

def test_entry_metric_and_baseline():
    y=np.array([1,1,0,2]);p=np.array([[.1,.8,.1],[.1,.7,.2],[.6,.3,.1],[.1,.2,.7]])
    m=_metrics(y,p,np.array([.02,.01,-.01,-.02]),threshold=.5)
    assert m['accuracy']==1 and m['entry_count']==2 and m['entry_precision']==1
