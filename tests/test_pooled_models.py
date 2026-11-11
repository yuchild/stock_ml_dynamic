import numpy as np
import pandas as pd
import pytest
from stock_ml_dynamic.pooled_models import chronological_masks, classification_experiment, regression_experiment, predict_latest


def fixture_data():
    n=120
    t=pd.date_range('2025-01-01', periods=n+1, freq='D', tz='UTC')
    rows=pd.DataFrame({'bar_timestamp':t[:-1], 'next_bar_timestamp':t[1:],
                       'direction':np.tile([2,0,1],40), 'next_return':np.tile([-.02,0,.02],40),
                       'adj_close':100., 'next_adj_close':100*(1+np.tile([-.02,0,.02],40))})
    x=pd.DataFrame({'a':np.sin(np.arange(n)/4), 'b':np.arange(n)%4},index=rows.index)
    return x,rows,t[80]


def test_chronological_split_has_no_target_overlap():
    x,rows,cutoff=fixture_data()
    tr,te=chronological_masks(rows,cutoff)
    assert rows.loc[tr,'next_bar_timestamp'].max()<=cutoff
    assert rows.loc[te,'bar_timestamp'].min()>cutoff
    assert not (tr&te).any()


def test_classifier_and_regressor_execute():
    x,rows,cutoff=fixture_data()
    models,metrics,_=classification_experiment(x,rows,cutoff)
    assert set(models)=={'majority_baseline','hist_gradient_boosting'}
    assert (metrics['train_rows']>0).all() and (metrics['test_rows']>0).all()
    assert metrics['roc_auc'].notna().all()
    models,metrics,_=regression_experiment(x,rows,cutoff)
    assert 'price_mae' in metrics and (metrics['return_mae']>=0).all()


def test_probability_columns_follow_class_labels():
    x,rows,cutoff=fixture_data()
    models,_,_=classification_experiment(x,rows,cutoff)
    latest=rows.iloc[-2:].copy(); latest['symbol']=['AAPL','MSFT'];latest['interval_code']='1d'
    preds=predict_latest(models,x.iloc[-2:],latest,'classification')
    for name,model in models.items():
        for cls in model.classes_:
            assert f'{name}_prob_{int(cls)}' in preds


def test_multistep_split_requires_target_timestamps():
    _,rows,cutoff=fixture_data()
    with pytest.raises(ValueError,match='horizon-specific'):
        chronological_masks(rows,cutoff,horizon=15)
