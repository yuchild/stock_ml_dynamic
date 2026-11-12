import numpy as np
import pandas as pd
import pytest
from stock_ml_dynamic.prospective import append_predictions, reconcile_predictions, prospective_scores, predict_latest


def sample_prediction():
    return pd.DataFrame([{
        'model_version':'v1','interval_code':'1d','symbol':'AAA',
        'bar_timestamp':pd.Timestamp('2026-11-10',tz='UTC'),
        'adj_close':100.0,'prob_0':0.3,'prob_1':0.5,'prob_2':0.2,
        'predicted_direction':1,'target_lower_edge':-0.01,'target_upper_edge':0.01,
        'outcome_timestamp':pd.NaT,'outcome_adj_close':np.nan,
        'outcome_return':np.nan,'actual_direction':pd.NA,'resolved_at_utc':pd.NA,
        'recorded_at_utc':'2026-11-10T23:00:00+00:00'}])


def test_append_is_idempotent(tmp_path):
    path=tmp_path/'ledger.parquet'
    row=sample_prediction()
    assert append_predictions(row,path)==(1,1)
    assert append_predictions(row,path)==(0,1)
    assert len(pd.read_parquet(path))==1


def test_reconcile_only_after_future_bar(tmp_path):
    path=tmp_path/'ledger.parquet'
    append_predictions(sample_prediction(),path)
    bars=pd.DataFrame([
        {'symbol':'AAA','interval_code':'1d','bar_timestamp':pd.Timestamp('2026-11-10',tz='UTC'),'adj_close_price':100.0},
        {'symbol':'AAA','interval_code':'1d','bar_timestamp':pd.Timestamp('2026-11-11',tz='UTC'),'adj_close_price':102.0},
        {'symbol':'AAA','interval_code':'1d','bar_timestamp':pd.Timestamp('2026-11-12',tz='UTC'),'adj_close_price':80.0},
    ])
    current, changed = reconcile_predictions(path,bars.iloc[:1])
    assert changed == 0
    resolved, changed = reconcile_predictions(path,bars)
    assert changed == 1
    assert resolved.iloc[0]['actual_direction'] == 1
    assert resolved.iloc[0]['outcome_adj_close'] == 102
    again, changed = reconcile_predictions(path,bars)
    assert changed == 0
    assert again.iloc[0]['prob_1'] == 0.5


def test_scores_unresolved_and_missing_classes():
    assert prospective_scores(sample_prediction()).empty
    row=sample_prediction()
    row['actual_direction']=1
    scores=prospective_scores(row)
    assert scores.iloc[0]['accuracy']==1
    assert np.isnan(scores.iloc[0]['roc_auc_ovr_macro'])


def test_duplicate_batch_rejected(tmp_path):
    with pytest.raises(ValueError,match='Duplicate'):
        append_predictions(pd.concat([sample_prediction(),sample_prediction()]),tmp_path/'l.parquet')
