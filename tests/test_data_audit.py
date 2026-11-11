import numpy as np
import pandas as pd
from stock_ml_dynamic.candle_table import build_candle_table, DirectionQuantiles


def test_next_bar_target_alignment_and_last_bar_unlabeled():
    n = 45
    close = np.arange(100, 100+n, dtype=float)
    df = pd.DataFrame({'bar_timestamp':pd.date_range('2026-01-01',periods=n,freq='D',tz='UTC'),
                       'open_price':close-.2,'close_price':close,'high_price':close+1,
                       'low_price':close-1,'adj_close_price':close,'volume':1000})
    result = build_candle_table(df)
    assert np.isclose(result.loc[0,'next_return'],close[1]/close[0]-1)
    assert result.loc[n-1,'next_adj_close'] != result.loc[n-1,'next_adj_close']
    assert pd.isna(result.loc[n-1,'next_return'])
    assert np.isclose(result.loc[30,'adj_z'],(close[30]-close[:30].mean())/close[:30].std(ddof=1))


def test_quantile_training_edges_are_fixed_for_future_extremes():
    q = DirectionQuantiles().fit(pd.Series(np.linspace(-.1,.1,99)))
    original = q.edges_.copy()
    labels = q.transform(pd.Series([-100.,100.]))
    assert labels.tolist() == [2,1]
    assert np.array_equal(original,q.edges_)
