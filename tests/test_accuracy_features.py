import numpy as np
import pandas as pd
import pytest
from stock_ml_dynamic.accuracy_features import add_accuracy_features,selective_metrics

def sample():
    return pd.DataFrame({'symbol':['AAA']*90,'interval_code':['1d']*90,'bar_timestamp':pd.date_range('2025-01-01',periods=90,tz='UTC'),'adj_close':np.linspace(100,140,90)+np.sin(np.arange(90)),'volume':np.arange(90)+100})

def test_causality_future_changes_do_not_modify_past():
    x=sample();a,n=add_accuracy_features(x)
    x.loc[80:,'adj_close']*=3;x.loc[80:,'volume']*=4
    b,_=add_accuracy_features(x)
    pd.testing.assert_frame_equal(a.loc[:79,n],b.loc[:79,n])

def test_duplicate_bars_rejected():
    x=sample();x=pd.concat([x,x.iloc[[0]]]);
    with pytest.raises(ValueError,match='Duplicate'):add_accuracy_features(x)

def test_selective_coverage_and_accuracy():
    p=np.array([[.1,.8,.1],[.7,.2,.1],[.4,.3,.3]])
    s=selective_metrics([1,0,2],p,thresholds=(.6,))
    assert s.iloc[0].n_selected==2 and s.iloc[0].selective_accuracy==1

def test_invalid_probabilities():
    with pytest.raises(ValueError):selective_metrics([1],[[.2,.2,.2]])
