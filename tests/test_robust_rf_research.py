import numpy as np
import pandas as pd
from stock_ml_dynamic.robust_rf_research import robust_matrices,select_inner,TREATMENTS

def frames():
    rng=np.random.default_rng(3)
    tr=pd.DataFrame({'a':rng.normal(size=300),'b':rng.normal(size=300)})
    va=pd.DataFrame({'a':[9999.,-9999.,0.],'b':[1.,2.,3.]})
    return tr,va

def test_clipping_bounds_from_train_only():
    tr,va=frames()
    _,z,_,_,bounds=robust_matrices(tr,va,['a','b'],TREATMENTS['clip_1'],cap=300)
    assert np.isfinite(z).all()
    assert z[0,0]==bounds.loc['a','upper']
    assert z[1,0]==bounds.loc['a','lower']
    assert bounds.loc['a','upper']<10

def test_raw_retains_outlier():
    tr,va=frames()
    _,z,_,_,bounds=robust_matrices(tr,va,['a','b'],None,cap=300)
    assert bounds is None and z[0,0]==9999.

def test_inner_selection_rejects_low_coverage():
    rows=[]
    for i in (1,2):
        rows.append(dict(features='price',treatment='raw',leaf='leaf8',threshold=.5,inner_fold=i,
                         accuracy=.42,auc=.56,mean_net=.1,entry_count=1,coverage=.001))
    chosen,_=select_inner(rows)
    assert chosen['entry'] is None
    assert chosen['accuracy']['treatment']=='raw'
