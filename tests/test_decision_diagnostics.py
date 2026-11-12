import numpy as np
import pandas as pd
from stock_ml_dynamic.decision_diagnostics import _aligned,_report,_deciles,_economic_binary,_safe_auc
from sklearn.linear_model import LogisticRegression

def test_report_no_entries():
    y=np.array([0,1,2,1]);p=np.array([[.9,.05,.05],[.8,.1,.1],[.1,.1,.8],[.7,.2,.1]])
    out=_report(y,p,np.zeros(4),fold=1,model='test')
    assert out['entry_count']==0 and np.isnan(out['entry_precision'])

def test_deciles_preserve_counts():
    n=103;y=np.arange(n)%3;p=np.column_stack([np.full(n,.3),np.linspace(.1,.6,n),np.full(n,.1)])
    p=p/p.sum(axis=1,keepdims=True)
    out=_deciles(pd.DataFrame(index=range(n)),y,p,np.zeros(n),1,'test',10)
    assert len(out)==10 and sum(row['n'] for row in out)==n

def test_aligned_probabilities():
    x=np.arange(90).reshape(-1,1);y=np.tile([0,1,2],30)
    m=LogisticRegression().fit(x,y);p=_aligned(m,x)
    assert p.shape==(90,3) and np.allclose(p.sum(axis=1),1)

def test_economic_target_baseline():
    rng=np.random.default_rng(42);x=rng.normal(size=(100,2));r=rng.normal(0,.02,100)
    train=pd.DataFrame({'next_return':r[:70]});test=pd.DataFrame({'next_return':r[70:]})
    out=_economic_binary(train,test,x[:70],x[70:],42,10)
    assert 0<=out['positive_base']<=1 and 0<=out['binary_accuracy']<=1

def test_auc_missing_class():
    assert np.isnan(_safe_auc(np.array([1,1]),np.array([[.1,.8,.1],[.2,.7,.1]])))
