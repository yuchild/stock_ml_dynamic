import numpy as np
import pandas as pd
from stock_ml_dynamic.edge_discovery import random_portfolio_null,tail_sensitivity,_candidate_specs

def test_random_null_constant_daily_returns():
    f=pd.DataFrame({'bar_timestamp':np.repeat(['2026-01-01','2026-01-02'],10),'next_return':np.repeat([.02,-.01],10)})
    x=random_portfolio_null(f,k=3,reps=20)
    assert np.allclose(x,.005)

def test_random_null_reproducible():
    f=pd.DataFrame({'bar_timestamp':np.repeat(['a','b'],10),'next_return':np.arange(20)/100})
    assert np.array_equal(random_portfolio_null(f,reps=10),random_portfolio_null(f,reps=10))

def test_tail_removal_reduces_extreme_winner():
    x=np.r_[np.full(100,.01),10.]
    out=tail_sensitivity(x,trim=(0,.01))
    assert out.iloc[1].expectancy<out.iloc[0].expectancy

def test_model_search_diversity():
    specs=_candidate_specs(42,10)
    assert len(specs)>=10 and len({x[0] for x in specs})==len(specs)
