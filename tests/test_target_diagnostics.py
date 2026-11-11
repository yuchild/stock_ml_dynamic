import numpy as np
import pandas as pd
from stock_ml_dynamic.target_diagnostics import make_labels, evaluate_predictions, summary_targets


def test_tertiles_training_only():
    tr,te,kt,ke,meta=make_labels('tertiles',[-3.,-2.,-1.,1.,2.,3.],[100.,-100.],np.ones(6),np.ones(2))
    assert kt.all() and ke.all()
    assert list(te)==[1,2]
    assert meta['cutoffs'][0]<meta['cutoffs'][1]


def test_volatility_target_uses_current_vol():
    tr,te,_,_,meta=make_labels('volatility_adjusted',[0.,.02,-.02],[.006,-.006,0.],
                               [.01,.01,.01],[.01,.01,.01],volatility_multiplier=.5)
    assert list(te)==[1,2,0]
    assert meta['volatility_window']==20


def test_binary_zero_exclusion():
    tr,te,kt,ke,_=make_labels('binary_direction',[-.01,0.,.01],[0.,-.02,.02],np.ones(3),np.ones(3))
    assert list(kt)==[True,False,True]
    assert list(ke)==[False,True,True]
    assert list(te[ke])==[0,1]


def test_metrics_binary():
    score=evaluate_predictions(np.array([0,1,0,1]),np.array([0,1,0,1]),
        np.array([[.9,.1],[.1,.9],[.8,.2],[.2,.8]]),[0,1])
    assert score['accuracy']==1 and score['roc_auc']==1


def test_invalid_target():
    import pytest
    with pytest.raises(ValueError):make_labels('unknown',[0,1],[0], [1,1],[1])
