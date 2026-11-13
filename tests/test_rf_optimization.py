import numpy as np
import pandas as pd
from stock_ml_dynamic.rf_optimization import score_entries,choose_candidate,sample_indices,prepare_features

def test_sample_is_deterministic_and_sorted():
    assert np.array_equal(sample_indices(1000,100,7),sample_indices(1000,100,7))
    assert len(np.unique(sample_indices(1000,100,7)))==100

def test_no_entries_are_explicit():
    x=score_entries(np.array([1,0]),np.array([[.2,.3,.5],[.2,.3,.5]]),[.02,-.01],.5)
    assert x['entry_count']==0 and np.isnan(x['precision'])

def test_cost_deducted_once():
    x=score_entries(np.array([1,0]),np.array([[.1,.8,.1],[.1,.7,.2]]),[.03,-.01],.6,10)
    assert x['entry_count']==2 and np.isclose(x['mean_net'],.009)

def test_training_imputer_not_test_fitted():
    tr=pd.DataFrame({'x':[1.,3.,np.nan]});te=pd.DataFrame({'x':[1000.,np.nan]})
    a,b=prepare_features(tr,te,['x'])
    assert np.isclose(b[1,0],2)

def test_choice_requires_min_entries_for_entry():
    rows=[]
    for fold in (1,2):
        for config,prec,n in [('low',.9,2),('robust',.6,120)]:
            rows.append(dict(fold=fold,config=config,features='price',threshold=.5,accuracy=.42,
                             auc=.55,precision=prec,coverage=.1,mean_net=.001,entry_count=n))
    choice,_=choose_candidate(rows,min_entries=100,min_coverage=.02,objective='entry')
    assert choice['config']=='robust'
