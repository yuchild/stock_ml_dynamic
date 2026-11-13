import numpy as np
import pandas as pd
import pytest
from stock_ml_dynamic.oof_stacking_research import oof_windows, fit_meta, meta_features, aligned_probs

def sample_frame():
    dates=pd.date_range('2024-01-01',periods=100,freq='D',tz='UTC')
    return pd.DataFrame({'bar_timestamp':dates,'next_bar_timestamp':dates+pd.Timedelta(days=1)})

def test_windows_are_strictly_forward():
    f=sample_frame()
    seen=[]
    for a,b in oof_windows(f,n_windows=5):
        assert f.next_bar_timestamp.iloc[a].max()<f.bar_timestamp.iloc[b].min()
        seen.extend(b.tolist())
    assert len(seen)==len(set(seen))

def test_oof_warmup_unscored():
    f=sample_frame()
    b=np.concatenate([b for _,b in oof_windows(f)])
    assert min(b)>0

def test_meta_models_are_probabilities():
    rng=np.random.default_rng(42)
    p=rng.dirichlet([2,2,2],size=300)
    y=np.tile([0,1,2],100)
    for kind in ['logistic','rf']:
        m=fit_meta(p,y,kind)
        q=aligned_probs(m,meta_features(p[:12]))
        assert q.shape==(12,3)
        np.testing.assert_allclose(q.sum(axis=1),1,atol=1e-9)

def test_bad_meta_rejected():
    with pytest.raises(ValueError):fit_meta(np.ones((30,3))/3,np.zeros(30),'logistic')
