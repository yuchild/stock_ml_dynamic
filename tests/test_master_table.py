import numpy as np
import pandas as pd
from stock_ml_dynamic.master_table import build_universe_features, assemble_master_table, model_ready


def prices(n=110):
    rows=[]
    for symbol,shift in [('AAA',0),('BBB',20),('CCC',40)]:
        for i in range(n):
            close=100+shift+i*.12+np.sin(i/4)*2
            rows.append(dict(symbol=symbol, interval_code='1d',bar_timestamp=pd.Timestamp('2025-01-01',tz='UTC')+pd.Timedelta(days=i),
                open_price=close-.2,high_price=close+1,low_price=close-1,close_price=close,
                adj_close_price=close,volume=10000))
    return pd.DataFrame(rows)


def test_one_future_target_missing_per_stock():
    features=build_universe_features(prices())
    result=assemble_master_table(features, k_values=range(2,5))
    assert len(result.inference)==3
    assert result.inference['direction'].isna().all()
    assert result.master['direction'].notna().all()
    assert result.master.groupby('symbol').size().eq(109).all()
    assert result.inference['next_adj_close'].isna().all()
    X, rows=model_ready(result)
    assert len(X)==len(rows)>0
    assert np.isfinite(X.to_numpy()).all()
    Xi,ri=model_ready(result,'inference')
    assert len(Xi)==len(ri)==3
    assert list(X.columns)==list(Xi.columns)


def test_no_cross_symbol_target():
    features=build_universe_features(prices())
    for _,g in features.groupby('symbol'):
        assert g.iloc[-1]['next_adj_close']!=g.iloc[-1]['next_adj_close']
        assert np.isclose(g.iloc[0]['next_adj_close'],g.iloc[1]['adj_close'])


def test_causal_features_unaffected_by_future():
    p=prices()
    a=build_universe_features(p)
    p.loc[(p.symbol=='AAA') & (p.bar_timestamp==p[p.symbol=='AAA'].bar_timestamp.max()),'adj_close_price']=999999
    b=build_universe_features(p)
    cols=['top_pct','body_pct','bottom_pct','gap','adj_z','slope','concavity']
    aa=a[a.symbol=='AAA'].iloc[:-1]
    bb=b[b.symbol=='AAA'].iloc[:-1]
    pd.testing.assert_frame_equal(aa[cols].reset_index(drop=True),bb[cols].reset_index(drop=True))


def test_multi_horizon_regression_targets_do_not_cross_stocks():
    from stock_ml_dynamic.master_table import add_forward_price_targets
    features=build_universe_features(prices())
    targets=add_forward_price_targets(features, horizons=[1,3,15])
    for _,group in targets.groupby('symbol'):
        assert group['adj_close_t+15'].tail(15).isna().all()
        assert np.isclose(group.iloc[0]['adj_close_t+3'],group.iloc[3]['adj_close'])
