import numpy as np
import pandas as pd
import pytest
from stock_ml_dynamic.temporal_calibration import temperature_scale,fit_temperature,ece,chronological_blocks
from stock_ml_dynamic.walk_forward import Fold

def test_temperature_normalization():
    p=np.array([[.2,.3,.5],[.4,.2,.4]])
    q=temperature_scale(p,1.5)
    assert np.allclose(q.sum(axis=1),1)
    assert np.all(q>0)

def test_identity_temperature():
    p=np.array([[.2,.3,.5],[.4,.2,.4]])
    assert np.allclose(temperature_scale(p,1),p)

def test_fit_temperature_finite():
    p=np.tile(np.array([[.2,.3,.5],[.5,.3,.2],[.3,.5,.2]]),(30,1))
    y=np.tile([2,0,1],30)
    assert .3<=fit_temperature(p,y)<=5

def test_ece_bounds():
    p=np.array([[.2,.3,.5],[.7,.2,.1]])
    assert 0<=ece(p,[2,0])<=1

def test_chronological_blocks():
    dates=pd.date_range('2020-01-01',periods=120,tz='UTC')
    df=pd.DataFrame({'bar_timestamp':dates,'next_bar_timestamp':dates+pd.Timedelta(days=1)})
    folds=chronological_blocks(df,Fold(1,dates[110],dates[119]))
    assert len(folds)==3
    assert all(f.train_end<f.test_end<=dates[110] for f in folds)
    assert folds[0].test_end==folds[1].train_end

def test_invalid_temperature():
    with pytest.raises(ValueError):temperature_scale(np.ones((2,3)),0)
