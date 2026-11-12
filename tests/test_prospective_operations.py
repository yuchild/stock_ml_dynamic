import numpy as np
import pandas as pd
import pytest
from stock_ml_dynamic.prospective_operations import source_audit, recording_gate, ledger_audit, evaluation_report

def sample_prices():
    rows=[]
    for symbol in ['A','B']:
        for i in range(32):
            rows.append(dict(symbol=symbol,interval_code='1d',bar_timestamp=pd.Timestamp('2026-11-10',tz='UTC')-pd.Timedelta(days=31-i),adj_close_price=100+i,open_price=100+i,high_price=101+i,low_price=99+i,close_price=100+i,volume=1000))
    return pd.DataFrame(rows)

def sample_predictions():
    return pd.DataFrame([dict(symbol=s,interval_code='1d',bar_timestamp=pd.Timestamp('2026-11-10',tz='UTC'),model_version='v1',prob_0=.3,prob_1=.4,prob_2=.3,predicted_direction=1,actual_direction=pd.NA) for s in ['A','B']])

def test_source_audit_counts_and_no_future():
    frame,report=source_audit(sample_prices(),now='2026-11-11T00:00:00Z')
    assert report['symbols']==2 and frame.eligible_history.all()
    p=sample_prices();p.loc[0,'bar_timestamp']=pd.Timestamp('2026-11-12',tz='UTC')
    with pytest.raises(ValueError,match='Future'):source_audit(p,now='2026-11-11T00:00:00Z')

def test_recording_requires_explicit_confirmation():
    p=sample_predictions();raw=sample_prices();bundle={'model_version':'v1','training_end':'2026-11-10T00:00:00Z'}
    with pytest.raises(PermissionError):recording_gate(p,raw,bundle,confirmed_session_date='2026-11-10',now='2026-11-11T00:00:00Z')
    ok=recording_gate(p,raw,bundle,confirmed_session_date='2026-11-10',operator_confirmed=True,now='2026-11-11T00:00:00Z')
    assert ok['predictions']==2
    with pytest.raises(ValueError,match='dates differ'):
        recording_gate(p,raw,bundle,confirmed_session_date='2026-11-09',operator_confirmed=True,now='2026-11-11T00:00:00Z')

def test_gate_rejects_newer_observed_bar():
    p=sample_predictions();raw=sample_prices();extra=raw.iloc[-1:].copy();extra['bar_timestamp']=pd.Timestamp('2026-11-11',tz='UTC')
    with pytest.raises(ValueError,match='Newer observed'):
        recording_gate(p,pd.concat([raw,extra]),{'model_version':'v1','training_end':'2026-11-10T00:00:00Z'},confirmed_session_date='2026-11-10',operator_confirmed=True,now='2026-11-12T00:00:00Z')

def test_ledger_integrity_and_pending():
    p=sample_predictions();summary,versions=ledger_audit(p)
    assert summary['pending']==2 and summary['resolved']==0
    q=p.copy();q.loc[0,'prob_1']=.9
    with pytest.raises(ValueError,match='invalid_probabilities'):ledger_audit(q)

def test_evaluation_empty():
    scores,details=evaluation_report(sample_predictions())
    assert scores.empty and details.empty


def test_evaluation_empty_ledger_without_columns():
    scores,details=evaluation_report(pd.DataFrame())
    assert scores.empty and details.empty
    assert list(scores.columns)==['model_version','resolved_rows','accuracy','balanced_accuracy','roc_auc_ovr_macro','log_loss']
    assert list(details.columns)==['model_version','class_id','support','recall','predicted_count']
    summary,by_version=ledger_audit(pd.DataFrame())
    assert summary['rows']==0 and summary['resolved']==0
    assert by_version.empty
