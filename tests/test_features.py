import numpy as np
import pandas as pd
import pytest
from stock_ml_dynamic.features import FEATURE_COLUMNS, build_features
from stock_ml_dynamic.health import audit_history


def sample(n=100):
    c = np.linspace(100, 120, n)
    return pd.DataFrame({"time":pd.date_range("2025-01-01",periods=n,freq="D"),"open":c,"high":c+1,"low":c-1,"close":c,"volume":np.full(n,1000)})

def test_future_targets_and_features():
    df=build_features(sample(), horizon=5)
    assert df.target_forward_return.tail(5).isna().all()
    assert df.target_up.tail(5).isna().all()
    assert set(FEATURE_COLUMNS).issubset(df.columns)
    assert df.loc[50,"target_forward_return"] == pytest.approx( df.loc[55,"close"]/df.loc[50,"close"]-1)

def test_causal_features_do_not_change_when_future_changes():
    a=sample()
    b=a.copy()
    b.loc[80:,"close"] *= 3
    left=build_features(a)
    right=build_features(b)
    pd.testing.assert_frame_equal(left.loc[:70, FEATURE_COLUMNS],right.loc[:70, FEATURE_COLUMNS])

def test_duplicate_rejected():
    df=sample()
    df.loc[1,"time"]=df.loc[0,"time"]
    with pytest.raises(ValueError):
        build_features(df)

def test_health_invalid_ohlc():
    df=sample()
    df.loc[0,"high"]=df.loc[0,"low"]-1
    assert audit_history(df)["invalid_ohlc"] == 1
