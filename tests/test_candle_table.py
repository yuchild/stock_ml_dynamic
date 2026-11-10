import numpy as np
import pandas as pd
from stock_ml_dynamic.candle_table import build_candle_table, apply_training_transformers

def sample(n=140):
    x=np.arange(n,dtype=float)
    c=100+x*.07+np.sin(x)*2
    return pd.DataFrame({'bar_timestamp':pd.date_range('2025-01-01',periods=n,freq='D',tz='UTC'),'open_price':c-.2,'high_price':c+1,'low_price':c-1,'close_price':c,'adj_close_price':c,'volume':1000})

def test_candle_sums_and_alignment():
    t=build_candle_table(sample())
    assert np.allclose(t[['top_pct','body_pct','bottom_pct']].sum(axis=1),1)
    assert np.isclose(t['next_return'].iloc[40],t['adj_close_pctc'].iloc[41])
    assert pd.isna(t['next_return'].iloc[-1])

def test_future_changes_do_not_change_past_features():
    a=sample();b=a.copy();b.loc[100:,'adj_close_price']*=2
    x=build_candle_table(a);y=build_candle_table(b)
    for c in ['top_pct','body_pct','bottom_pct','gap','adj_z','slope','concavity']:
        pd.testing.assert_series_equal(x[c].iloc[:100],y[c].iloc[:100])

def test_fit_transform_and_last_target():
    t=build_candle_table(sample())
    out,q,k=apply_training_transformers(t,100,k_values=range(2,5))
    assert pd.isna(out['direction'].iloc[-1])
    assert set(out['direction'].dropna().unique()).issubset({0,1,2})
    assert k.chosen_k_ in {2,3,4}
