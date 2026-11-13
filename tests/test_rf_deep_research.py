import numpy as np
import pandas as pd
from stock_ml_dynamic.rf_deep_research import candidate_grid,training_indices,entry_metrics,choose,cost_sensitivity,daily_aggregated_returns

def test_grid_unique():
    c=candidate_grid();assert len(c)==36 and len({x['name'] for x in c})==36

def test_sampling_recent():
    assert training_indices(pd.DataFrame(index=range(200)),100,42,'recent').tolist()==list(range(100,200))

def test_sampling_uniform_deterministic():
    x=pd.DataFrame(index=range(200));assert np.array_equal(training_indices(x,100,42,'uniform'),training_indices(x,100,42,'uniform'))

def test_entry_mask_requires_argmax():
    p=np.array([[.1,.6,.3],[.1,.4,.5],[.2,.7,.1]])
    v=entry_metrics(np.array([1,1,0]),p,np.array([.02,.03,-.01]),.4,10)
    assert v['entry_count']==2 and np.isclose(v['mean_net'],.004)

def test_inner_entry_requires_coverage_each_fold():
    rows=[dict(config='x',features='price',sampling='uniform',threshold=.5,inner_fold=i,
        accuracy=.5,auc=.6,logloss=1.,precision=.6,mean_net=.01,coverage=c,entry_count=n)
        for i,(c,n) in enumerate([(.04,200),(.001,4)],1)]
    selected,_=choose(rows,'entry',150,.025);assert selected is None

def test_costs():
    t=pd.DataFrame([dict(objective='entry',fold=1,gross_return=.01,bar_timestamp='2026-01-01',net_return=.009)])
    c=cost_sensitivity(t,(0,10));assert np.isclose(c.iloc[1].mean_net,.009)
    d=daily_aggregated_returns(t);assert len(d)==1 and d.iloc[0].trades==1
