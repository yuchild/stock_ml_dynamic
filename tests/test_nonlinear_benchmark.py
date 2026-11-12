import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from stock_ml_dynamic.nonlinear_benchmark import aligned_proba, score_probabilities, chronological_neural_split, build_estimators

def test_aligned_probabilities():
    m=DummyClassifier(strategy='prior').fit(np.zeros((6,2)),[0,2,2,0,2,0])
    p=aligned_proba(m,np.zeros((4,2)))
    assert p.shape==(4,3)
    assert np.all(p[:,1]==0)
    assert np.allclose(p.sum(axis=1),1)

def test_invalid_probabilities():
    with pytest.raises(ValueError):score_probabilities([0,1],np.array([[.2,.3,.4],[.2,.3,.5]]))

def test_scoring():
    y=np.array([0,1,2,0,1,2])
    p=np.eye(3)[y]*.9+.1/3
    assert score_probabilities(y,p)['accuracy']==1.0

def test_chronological_split():
    dates=pd.date_range('2024-01-01',periods=60,tz='UTC')
    df=pd.DataFrame({'next_bar_timestamp':np.repeat(dates,5)})
    mask,cutoff=chronological_neural_split(df)
    assert df.loc[mask,'next_bar_timestamp'].max()<=cutoff
    assert df.loc[~mask,'next_bar_timestamp'].min()>cutoff

def test_models_present():
    assert {'random_forest','extra_trees','hist_gradient_boosting','logistic_l2_fixed','prior_baseline'}<=set(build_estimators())
