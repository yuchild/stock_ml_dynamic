import numpy as np
import pandas as pd
from stock_ml_dynamic.sequential_deck_stacking import chronological_decks, train_stage, augment, one_fold


def make_frame(n_dates=90,n_tickers=4):
    rng=np.random.default_rng(17)
    dates=pd.date_range('2023-01-01',periods=n_dates+1,freq='D',tz='UTC')
    records=[]
    for j in range(n_dates):
        for t in range(n_tickers):
            records.append(dict(symbol=f'S{t}',interval_code='1d',bar_timestamp=dates[j],
                                next_bar_timestamp=dates[j+1],next_return=rng.normal(0,.02),
                                f1=rng.normal(),f2=rng.lognormal()))
    return pd.DataFrame(records)


def test_chronological_decks_no_date_split():
    d=chronological_decks(make_frame())
    assert len(d)==4
    assert sum(map(len,d))==360
    assert all(a.next_bar_timestamp.max()<=b.bar_timestamp.min() for a,b in zip(d,d[1:]))


def test_upstream_predictions_are_out_of_sample():
    frame=make_frame(110)
    d=chronological_decks(frame)
    model=train_stage(d[0].assign(direction=np.tile([0,1,2,1],len(d[0])//4)),['f1','f2'],trees=8,leaf=2,cap=500,seed=2)
    x=augment(d[1], [model])
    assert {'model1_p0','model1_p1','model1_p2','model1_pred'}<=set(x.columns)
    np.testing.assert_allclose(x[['model1_p0','model1_p1','model1_p2']].sum(axis=1),1)


def test_four_stage_pipeline_and_baselines():
    frame=make_frame(110)
    tr=frame[frame.bar_timestamp<pd.Timestamp('2023-04-01',tz='UTC')]
    te=frame[frame.bar_timestamp>=pd.Timestamp('2023-04-01',tz='UTC')]
    report,audit,stages=one_fold(tr,te,['f1','f2'],trees=8,leaf=2,cap=500,seed=3)
    assert len(stages)==4
    assert list(audit.features)==[2,5,8,11]
    assert set(report.model)=={'sequential_four_deck','deck4_only_rf','all_training_rf'}
    assert (report.n==len(te)).all()
