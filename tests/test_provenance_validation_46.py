import numpy as np
import pandas as pd
from stock_ml_dynamic.provenance_validation_46 import master_alignment,compare_raw_to_master,decision_gate

def example():
    return pd.DataFrame({'symbol':['A','A','A'],'bar_timestamp':pd.to_datetime(['2020-01-01','2020-01-02','2020-01-03'],utc=True),'next_bar_timestamp':pd.to_datetime(['2020-01-02','2020-01-03','2020-01-04'],utc=True),'next_return':[.01,.02,.03],'adj_close':[1,1.01,1.03]})

def test_alignment_ok():
    a=master_alignment(example());assert a.failures.sum()==0

def test_alignment_detects_wrong_next():
    x=example();x.loc[0,'next_bar_timestamp']=pd.Timestamp('2020-01-03',tz='UTC');assert master_alignment(x).failures.sum()==1

def test_raw_comparison():
    x=example();raw=x[['symbol','bar_timestamp','adj_close']].rename(columns={'adj_close':'adj_close_price'});a=compare_raw_to_master(x,raw);assert a.price_mismatches_gt_1e_8.iloc[0]==0

def test_gate_unverified():
    gate=decision_gate(master_alignment(example()),False);assert gate.loc[gate.check.eq('historical_universe'),'status'].iloc[0]=='UNVERIFIED'
