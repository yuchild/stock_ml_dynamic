import numpy as np
import pandas as pd
from stock_ml_dynamic.classifier_benchmark import eligible_universe, _selector, _probabilities
from sklearn.dummy import DummyClassifier


def test_short_history_excludes_entire_symbol():
    df = pd.DataFrame([{'symbol': s, 'interval_code':'1d', 'bar_timestamp':pd.Timestamp('2026-01-01',tz='UTC')+pd.Timedelta(days=i), 'adj_z':float(i)} for s,n in [('SHORT',29),('LONG',40)] for i in range(n)])
    result, report = eligible_universe(df)
    assert set(result.symbol)=={'LONG'}
    assert report.loc[report.symbol.eq('SHORT'),'eligible'].iloc[0] == False


def test_invalid_latest_z_excludes_symbol():
    df = pd.DataFrame({'symbol':['A']*35,'interval_code':['1d']*35,'bar_timestamp':pd.date_range('2026-01-01',periods=35,tz='UTC'),'adj_z':[0.2]*34+[np.nan]})
    result, report = eligible_universe(df)
    assert result.empty


def test_feature_schemes_fit_and_transform():
    rng=np.random.default_rng(42)
    x=rng.normal(size=(120,12)); y=np.tile([0,1,2],40)
    for name in ('all','pca_95','lasso_l1','ridge_l2','anova_50'):
        tr,te,n,_ = _selector(name,x,y,x[:9]); assert tr.shape==(120,n); assert te.shape==(9,n)


def test_probability_alignment():
    x=np.arange(12).reshape(-1,1); y=np.tile([2,0,1],4)
    m=DummyClassifier(strategy='prior').fit(x,y)
    p=_probabilities(m,x[:2]); assert p.shape==(2,3); assert np.allclose(p.sum(axis=1),1)
