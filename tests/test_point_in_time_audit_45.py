import numpy as np
import pandas as pd
from stock_ml_dynamic.point_in_time_audit_45 import master_integrity,frozen_sensitivity,score_integrity

def test_master_identity_and_order():
    m=pd.DataFrame({'symbol':['A'],'interval_code':['1d'],'bar_timestamp':pd.to_datetime(['2024-01-01'],utc=True),'next_bar_timestamp':pd.to_datetime(['2024-01-02'],utc=True),'adj_close':[100.],'next_adj_close':[110.],'next_return':[.10]})
    checks=master_integrity(m,pd.DataFrame({'symbol':['A']}),{}).set_index('check').value
    assert checks['target_identity_over_1e-8']==0
    assert checks['nonmonotonic_target_timestamps']==0

def test_bad_target_flagged():
    m=pd.DataFrame({'symbol':['A'],'interval_code':['1d'],'bar_timestamp':pd.to_datetime(['2024-01-01'],utc=True),'next_bar_timestamp':pd.to_datetime(['2024-01-02'],utc=True),'adj_close':[100.],'next_adj_close':[110.],'next_return':[.05]})
    checks=master_integrity(m,pd.DataFrame({'symbol':['A']}),{}).set_index('check').value
    assert checks['target_identity_over_1e-8']==1

def test_sensitivity_never_changes_selection():
    x=pd.DataFrame({'fold':[1]*4,'bar_timestamp':pd.to_datetime(['2024-01-01']*2+['2024-01-02']*2,utc=True),'symbol':['A','B','A','B'],'score':[2.,1.,2.,1.],'next_return':[.5,-.2,-.3,.1]})
    result,_=frozen_sensitivity(x,ks=(1,),thresholds=(.2,))
    original=result[result.scenario.eq('original')].mean_daily.iloc[0]
    winsor=result[result.scenario.eq('winsorize_both_0.2')].mean_daily.iloc[0]
    assert np.isclose(original,.1)
    assert np.isclose(winsor,0.)

def test_duplicate_scores_flagged():
    x=pd.DataFrame({'fold':[1,1],'bar_timestamp':pd.to_datetime(['2024-01-01']*2,utc=True),'symbol':['A','A'],'score':[1.,2.],'next_return':[.1,.2]})
    assert score_integrity(x).duplicate_stock_dates.iloc[0]==1
