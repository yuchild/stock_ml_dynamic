import numpy as np
import pandas as pd
from stock_ml_dynamic.volatility_diagnostics import volatility_schemes, _score, summary_volatility


def test_schemes_hold_nonvolatility_constant():
    extra=['return_1bar','realized_vol_5','realized_vol_10','realized_vol_20','peer_market_return']
    schemes=volatility_schemes(extra)
    assert len(schemes)==8
    assert schemes['no_volatility']==['return_1bar','peer_market_return']
    assert schemes['all_volatility']==extra
    assert schemes['vol_10_only']==['return_1bar','peer_market_return','realized_vol_10']
    assert all('return_1bar' in cols and 'peer_market_return' in cols for cols in schemes.values())


def test_scores_perfect_three_class_predictions():
    y=np.array([0,1,2,0,1,2])
    p=np.eye(3)[y]
    result=_score(y,y,p)
    assert result['accuracy']==1.0
    assert result['balanced_accuracy']==1.0
    assert result['roc_auc_ovr_macro']==1.0


def test_summary_grouped():
    data=pd.DataFrame([dict(scheme='all_volatility',model='logistic_l2',accuracy=.4,
      balanced_accuracy=.38,roc_auc_ovr_macro=.56,log_loss=1.1,fit_seconds=2),
      dict(scheme='all_volatility',model='logistic_l2',accuracy=.42,
      balanced_accuracy=.4,roc_auc_ovr_macro=.57,log_loss=1.09,fit_seconds=3)])
    summary=summary_volatility(data)
    assert abs(summary.iloc[0].accuracy-.41)<1e-9
    assert summary.iloc[0].fit_seconds==5
