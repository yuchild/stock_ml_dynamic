import numpy as np
import pandas as pd
from stock_ml_dynamic.candle_table import DirectionQuantiles
from stock_ml_dynamic.nested_accuracy import _model

def test_septile_mapping_and_outside_edges():
    x = pd.Series(np.arange(700,dtype=float))
    target=DirectionQuantiles(q=7).fit(x)
    assert target.labels_ == [2,2,0,0,1,1,1]
    y=target.transform(pd.Series([10,110,210,310,410,510,610,-10,1000]))
    assert y.tolist()==[2,2,0,0,1,1,1,2,1]

def test_duplicate_edges_rejected():
    import pytest
    with pytest.raises(ValueError,match='unique bins'):
        DirectionQuantiles(q=7).fit(pd.Series([0]*100+[1]*100))

def test_balanced_estimators():
    assert _model('logistic',.1,42).class_weight=='balanced'
    assert _model('forest',20,42).class_weight=='balanced'

def test_legacy_target_unchanged():
    q=DirectionQuantiles().fit(pd.Series(np.arange(300,dtype=float)))
    assert q.labels_ == [2,0,1]
