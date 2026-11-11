import numpy as np
import pandas as pd
from stock_ml_dynamic.walk_forward import expanding_folds, fold_data, evaluate_fold, diagnose_latest_z


def sample_data(n=130):
    rng=np.random.default_rng(12)
    dates=pd.date_range('2024-01-01',periods=n,freq='D',tz='UTC')
    parts=[]
    for sym in ('AAA','BBB'):
        p=100*np.exp(np.cumsum(rng.normal(0,.02,n)))
        df=pd.DataFrame({'symbol':sym,'interval_code':'1d','bar_timestamp':dates,'adj_close':p,
            'top_pct':rng.uniform(.1,.4,n),'body_pct':rng.uniform(.1,.5,n),'bottom_pct':rng.uniform(.1,.4,n),
            'gap':rng.normal(0,.05,n),'adj_z':rng.normal(size=n),'slope':rng.normal(size=n),'concavity':rng.normal(size=n),
            'month_of_year':dates.month,'day_of_month':dates.day,'day_of_week':dates.dayofweek,'hour_of_day':dates.hour})
        df['next_bar_timestamp']=df['bar_timestamp'].shift(-1)
        df['next_adj_close']=df['adj_close'].shift(-1)
        df['next_return']=df['next_adj_close']/df['adj_close']-1
        parts.append(df)
    return pd.concat(parts,ignore_index=True)


def test_folds_no_overlap():
    df=sample_data()
    folds=expanding_folds(df,n_folds=2)
    assert folds[0].test_end < folds[1].test_end
    assert folds[0].test_end == folds[1].train_end or folds[0].test_end < folds[1].train_end


def test_fold_transformers_are_training_only():
    df=sample_data()
    fold=expanding_folds(df,n_folds=2)[0]
    train,test,xtr,xte,bundle=fold_data(df,fold,selected_k=3)
    assert train.next_bar_timestamp.max() <= fold.train_end
    assert test.bar_timestamp.min() > fold.train_end
    assert test.next_bar_timestamp.max() <= fold.test_end
    assert xtr.shape[1]==xte.shape[1]
    assert set(train.direction.unique()).issubset({0,1,2})
    assert bundle['clusters'].chosen_k_==3
    # A change to future returns cannot affect fitted training quantiles.
    changed=df.copy()
    changed.loc[changed.bar_timestamp>fold.train_end,'next_return']=.99
    _,_,_,_,bundle2=fold_data(changed,fold,selected_k=3)
    np.testing.assert_array_equal(bundle['quantiles'].edges_,bundle2['quantiles'].edges_)


def test_evaluate_fold():
    df=sample_data()
    fold=expanding_folds(df,n_folds=2)[0]
    train,test,xtr,xte,_=fold_data(df,fold,selected_k=3)
    result=evaluate_fold(train,test,xtr,xte,fold_number=1)
    assert len(result)==4
    assert set(result.task)=={'classification','regression'}


def test_diagnose_missing():
    dates=pd.date_range('2024-01-01',periods=31,freq='D',tz='UTC')
    df=pd.DataFrame({'symbol':'VYLR','bar_timestamp':dates,'adj_close_price':[np.nan]+list(range(100,130))})
    assert diagnose_latest_z(df)['latest_z_missing']
