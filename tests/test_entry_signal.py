import numpy as np
import pandas as pd
from stock_ml_dynamic.entry_signal import entry_metrics,choose_entry_rule,make_estimator

def test_only_up_argmax_enters():
    y=np.array([1,1,0,2]);p=np.array([[.1,.8,.1],[.55,.4,.05],[.1,.6,.3],[.1,.2,.7]])
    a=entry_metrics(y,p,.35,min_coverage=.2)
    assert a['entries']==2 and a['precision_up']==.5

def test_net_costs_applied_to_entries_only():
    y=np.array([1,0]);p=np.array([[.1,.8,.1],[.7,.2,.1]])
    a=entry_metrics(y,p,.5,cost_bps=10,next_returns=[.02,-.1])
    assert np.isclose(a['mean_entry_net_return'],.019)

def test_rule_uses_pooled_counts():
    rows=[dict(config='a',threshold=.5,n=100,entries=10,precision_up=.9,base_up_rate=.4),
          dict(config='a',threshold=.5,n=100,entries=90,precision_up=.5,base_up_rate=.4),
          dict(config='b',threshold=.5,n=200,entries=100,precision_up=.6,base_up_rate=.4)]
    rule,_=choose_entry_rule(rows,min_coverage=.05)
    assert rule==('b',.5)

def test_no_eligible_rule_abstains():
    rule,_=choose_entry_rule([dict(config='a',threshold=.9,n=100,entries=1,precision_up=1.,base_up_rate=.4)],min_coverage=.05)
    assert rule is None

def test_class_weight_variants():
    assert make_estimator('forest_balanced').class_weight=='balanced'
    assert make_estimator('forest_unweighted').class_weight is None
    assert make_estimator('logistic_balanced').class_weight=='balanced'
