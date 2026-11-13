import numpy as np
import pandas as pd
import pytest
from stock_ml_dynamic.deployable_policy_research import select_portfolio,summarize,bootstrap_daily


def fixture():
    dates=pd.date_range('2025-01-01',periods=12,tz='UTC')
    rows=[]
    for model in ('full_history_rf','oof_meta_logistic','oof_meta_rf'):
        for date in dates:
            for i in range(4):
                rows.append(dict(fold=1,model=model,bar_timestamp=date,symbol=f'S{i}',
                    sector='A' if i<2 else 'B',up_probability=.8-i*.1,
                    direction=1 if i%2==0 else 2,next_return=.02-i*.005))
    p=pd.DataFrame(rows)
    thresholds=pd.DataFrame([dict(fold=1,model=m,threshold=.5) for m in p.model.unique()])
    return p,thresholds


def test_fixed_threshold_and_position_cap():
    p,t=fixture()
    trades,daily=select_portfolio(p,t,max_positions=2)
    assert (daily.entries==2).all()
    assert set(trades.symbol)=={'S0','S1'}
    assert (daily.eligible==4).all()


def test_sector_cap_and_missing_mapping():
    p,t=fixture()
    trades,_=select_portfolio(p,t,max_positions=3,max_per_sector=1)
    assert set(trades.symbol)=={'S0','S2'}
    with pytest.raises(ValueError,match='sector'):
        select_portfolio(p.drop(columns='sector'),t,max_per_sector=1)


def test_no_validation_outcome_used_for_selection():
    p,t=fixture()
    a,_=select_portfolio(p,t,max_positions=2)
    p['direction']=2;p['next_return']=-.4
    b,_=select_portfolio(p,t,max_positions=2)
    assert a[['fold','model','bar_timestamp','symbol']].equals(b[['fold','model','bar_timestamp','symbol']])


def test_summary_and_bootstrap():
    p,t=fixture()
    trades,daily=select_portfolio(p,t,max_positions=2)
    scores=summarize(trades,daily)
    assert len(scores)==3
    ci=bootstrap_daily(daily,reps=20,block_days=3)
    assert len(ci)==2
    assert np.isfinite(ci[['ci_low','ci_high']].to_numpy()).all()
