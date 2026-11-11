import numpy as np
import pandas as pd
from stock_ml_dynamic.historical_market import add_historical_market_features


def synthetic(n=50, stocks=12):
    rows=[]
    for j in range(stocks):
        for i in range(n):
            price=100+j+i*.2
            rows.append(dict(symbol=f'S{j}',interval_code='1d',bar_timestamp=pd.Timestamp('2026-01-01',tz='UTC')+pd.Timedelta(days=int(i)),adj_close=price,volume=100+i,slope=0.2,concavity=0))
    return pd.DataFrame(rows)


def test_features_only_use_past_and_present():
    raw=synthetic()
    a,names=add_historical_market_features(raw)
    changed=raw.copy()
    changed.loc[changed.bar_timestamp>=pd.Timestamp('2026-02-10',tz='UTC'),'adj_close'] *= 4
    b,_=add_historical_market_features(changed)
    cutoff=pd.Timestamp('2026-02-09',tz='UTC')
    left=a.loc[a.bar_timestamp<=cutoff,names].reset_index(drop=True)
    right=b.loc[b.bar_timestamp<=cutoff,names].reset_index(drop=True)
    pd.testing.assert_frame_equal(left,right)


def test_leave_one_out_market_proxy():
    raw=synthetic()
    features,_=add_historical_market_features(raw)
    r=features[(features.symbol=='S0') & (features.bar_timestamp==pd.Timestamp('2026-01-15',tz='UTC'))].iloc[0]
    expected=np.mean([0.2/(100+j+13*.2) for j in range(1,12)])
    assert np.isclose(r.peer_market_return,expected)


def test_initial_history_is_missing():
    features,_=add_historical_market_features(synthetic())
    assert features.loc[features.symbol.eq('S0')].iloc[0].return_20bar != features.loc[features.symbol.eq('S0')].iloc[0].return_20bar
    assert np.isfinite(features.loc[features.symbol.eq('S0')].iloc[-1].return_20bar)


def test_duplicate_bars_rejected():
    raw=synthetic()
    try:
        add_historical_market_features(pd.concat([raw,raw.iloc[[0]]],ignore_index=True))
    except ValueError as e:
        assert 'Duplicate' in str(e)
    else:
        raise AssertionError('Expected duplicate rejection')
