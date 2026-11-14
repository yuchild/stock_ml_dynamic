import numpy as np
import pandas as pd
from stock_ml_dynamic.edge_source_research import feature_groups, trade_diagnostics, reality_check

def test_groups_partition():
    names=['market_breadth','volume_z','momentum_5','atr','close','misc']
    groups=feature_groups(names)
    assert sorted(i for ids in groups.values() for i in ids)==list(range(len(names)))

def test_frozen_diagnostics():
    df=pd.DataFrame({'model':['ridge_a100']*4,'k':[5]*4,'fold':[1]*4,'bar_timestamp':pd.date_range('2025-01-01',periods=4,tz='UTC'),'symbol':['A','B','C','D'],'next_return':[.02,-.01,.03,-.02]})
    d=trade_diagnostics(df)
    assert len(d['top_winners'])==4
    assert len(d['by_symbol'])==4

def test_reality_check():
    dates=pd.date_range('2024-01-01',periods=40,tz='UTC')
    d=pd.DataFrame([{'fold':f,'bar_timestamp':date,'model':m,'k':5,'daily_return':.001 if m=='ridge_all' else .002} for f in (1,2,3) for date in dates for m in ('ridge_all','challenger')])
    out=reality_check(d,reps=30)
    assert len(out)==1 and 0<=out.familywise_p.iloc[0]<=1
