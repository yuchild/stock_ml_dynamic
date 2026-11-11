import numpy as np
import pandas as pd
import pytest
from stock_ml_dynamic.feature_ablation import feature_schemes, development_holdout_split, GROUPS, _model

def test_feature_schemes_exclude_only_named_group():
    cols=[c for g in GROUPS.values() for c in g]
    schemes=feature_schemes(cols)
    assert schemes['original']==[]
    assert schemes['all_expanded']==cols
    for name,group in GROUPS.items():
        assert set(schemes['without_'+name])==set(cols)-set(group)

def test_holdout_boundary_is_chronological_and_unscored():
    times=pd.date_range('2020-01-01',periods=200,freq='D',tz='UTC')
    df=pd.DataFrame({'bar_timestamp':times,'symbol':'A'})
    dev,boundary=development_holdout_split(df,.20)
    assert len(dev)==160
    assert dev.bar_timestamp.max()==boundary
    assert (df.loc[~df.index.isin(dev.index),'bar_timestamp']>boundary).all()

def test_holdout_rejects_bad_fraction():
    df=pd.DataFrame({'bar_timestamp':pd.date_range('2020-01-01',periods=200,freq='D',tz='UTC')})
    with pytest.raises(ValueError): development_holdout_split(df,.9)

def test_model_choices_are_classification_only():
    for name in ('prior_baseline','logistic_l1','logistic_l2'):
        assert hasattr(_model(name),'fit')
    with pytest.raises(ValueError): _model('random_forest')
