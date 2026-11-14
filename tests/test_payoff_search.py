import numpy as np
import pandas as pd
from stock_ml_dynamic.payoff_search import payoff_stats,daily_top_k,date_block_ci

def test_payoff_2_to_1():
    p=payoff_stats([.02,.02,-.01,-.01,-.01])
    assert np.isclose(p['payoff_ratio'],2)
    assert np.isclose(p['win_rate'],.4)
    assert p['meets_2_to_1']
    assert np.isclose(p['expectancy'],.002)

def test_zero_and_empty():
    assert np.isnan(payoff_stats([])['expectancy'])
    assert payoff_stats([0,0])['win_rate']==0

def test_top_k_deterministic():
    df=pd.DataFrame({'bar_timestamp':['2025-01-01']*3+['2025-01-02']*3,'symbol':['A','B','C']*2,'next_return':[.1,-.1,.2]*2})
    result=daily_top_k(df,[.1,.3,.2,.4,.1,.2],2,'test',1)
    assert result[result.selected].groupby('bar_timestamp').size().tolist()==[2,2]
    assert set(result[result.selected].query('bar_timestamp=="2025-01-01"').symbol)=={'B','C'}

def test_ci_deterministic():
    x=np.linspace(-.1,.1,100)
    assert date_block_ci(x,seed=1)==date_block_ci(x,seed=1)
