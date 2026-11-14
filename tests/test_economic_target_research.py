import numpy as np
import pandas as pd
from stock_ml_dynamic.economic_target_research import prepare_extras,choose_positions

def test_train_only_bounds():
    tr=pd.DataFrame({'x':[1.,2.,3.,4.,5.]})
    te=pd.DataFrame({'x':[10000.]})
    a,b=prepare_extras(tr,te,['x'])
    assert b[0,0]<6

def test_fixed_slots_and_cost():
    x=pd.DataFrame({'bar_timestamp':pd.to_datetime(['2024-01-01']*3,utc=True),
                    'symbol':['A','B','C'],'next_return':[.03,-.02,.01]})
    picks,daily=choose_positions(x,[.8,.9,.7],[True,True,True],2,.001)
    assert picks.selected.sum()==2
    assert np.isclose(daily.portfolio_net.iloc[0],(.03-.001-.02-.001)/2)

def test_empty_cash_day():
    x=pd.DataFrame({'bar_timestamp':pd.to_datetime(['2024-01-01'],utc=True),
                    'symbol':['A'],'next_return':[.05]})
    picks,daily=choose_positions(x,[.9],[False],10,.001)
    assert daily.portfolio_net.iloc[0]==0 and daily.utilization.iloc[0]==0
