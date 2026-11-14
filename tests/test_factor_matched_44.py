import numpy as np
import pandas as pd
from stock_ml_dynamic.factor_matched_44 import add_factor_bins, matched_random_experiment, within_bin_directionality, stratified_topk

def example():
    rng=np.random.default_rng(42)
    rows=[]
    for fold in (1,2,3):
        for day in range(25):
            for i in range(24):
                rows.append(dict(fold=fold,bar_timestamp=pd.Timestamp('2024-01-01',tz='UTC')+pd.Timedelta(days=day+fold*30),symbol=f'S{i}',score=float(rng.normal()),next_return=float(rng.normal(0,.02)),volatility_20=float(i),relative_volume_20=float(i%8),volume=float(i*100+1),market_momentum_5=float(i%3)))
    return pd.DataFrame(rows)

def test_factor_bins_are_observable():
    x,fac=add_factor_bins(example())
    assert 'volatility_20' in fac
    assert x['bin_volatility_20'].between(0,2).all()

def test_matching_deterministic_and_no_future_leak():
    x=example()
    a,summary,_,_=matched_random_experiment(x,k=5,reps=8,seed=9)
    b,summary2,_,_=matched_random_experiment(x,k=5,reps=8,seed=9)
    pd.testing.assert_frame_equal(a,b)
    pd.testing.assert_frame_equal(summary,summary2)
    shuffled=x.copy();shuffled['next_return']=shuffled['next_return']+1
    c,_,_,_=matched_random_experiment(shuffled,k=5,reps=8,seed=9)
    assert np.allclose(c.matched_mean-a.matched_mean,1)

def test_directionality_and_balance():
    x=example()
    d=within_bin_directionality(x)
    assert len(d)>0
    b=stratified_topk(x,k=5)
    assert b.groupby(['fold','bar_timestamp']).size().eq(5).all()
