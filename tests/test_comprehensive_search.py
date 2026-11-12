import numpy as np
import pandas as pd
from stock_ml_dynamic.comprehensive_search import Candidate, build_estimator, classification_metrics, summarize_inner

def test_balanced_estimators():
    for family in ('logistic','forest','extra_trees'):
        m=build_estimator(Candidate('a',family,'all',.1 if family=='logistic' else 20))
        assert m.class_weight=='balanced'

def test_histogram_balanced_by_fit_weights():
    c=Candidate('h','hist_boost','all',15)
    assert c.balanced and build_estimator(c).__class__.__name__=='HistGradientBoostingClassifier'

def test_class_one_precision():
    y=np.array([1,1,2,0])
    p=np.array([[.1,.8,.1],[.1,.7,.2],[.1,.8,.1],[.8,.1,.1]])
    result=classification_metrics(y,p)
    assert np.isclose(result['up_precision'],2/3)
    assert result['n']==4

def test_inner_selection_no_outer_scores():
    rows=[]
    for name,acc,wins,entries in [('a',.5,30,50),('b',.6,40,50)]:
        for fold in (1,2):
            rows.append(dict(candidate=name,threshold=.5,n=100,entries=entries,up_wins=wins,
                             up_base_wins=42,net_return_sum=1.,meets_min_coverage=True,
                             accuracy=acc,balanced_accuracy=acc,roc_auc=.55,fit_seconds=1.))
    accuracy,entry,_,_=summarize_inner(rows,min_coverage=.05,min_entries=10)
    assert accuracy=='b'
    assert entry==('b',.5)

def test_entry_abstains_when_no_precision_lift():
    rows=[dict(candidate='a',threshold=.5,n=100,entries=20,up_wins=5,up_base_wins=42,
               net_return_sum=-1.,meets_min_coverage=True,accuracy=.4,balanced_accuracy=.4,roc_auc=.55,fit_seconds=1.)]
    _,entry,_,_=summarize_inner(rows,min_coverage=.05,min_entries=10)
    assert entry is None
