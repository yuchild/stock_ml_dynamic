import numpy as np
import pandas as pd
import pytest
from stock_ml_dynamic.experiment import ExperimentSpec
from stock_ml_dynamic.experiments import build_experiment_frame
from stock_ml_dynamic.experiment_training import train_experiment, comparison_rows


def prices(n=450):
    rng = np.random.default_rng(7)
    close = 100 * np.exp(np.cumsum(rng.normal(0, .015, n)))
    return pd.DataFrame({'time': pd.date_range('2024-01-01', periods=n), 'open': close*.998,
                         'high': close*1.01, 'low': close*.99, 'close': close,
                         'volume': rng.integers(100, 1000, n)})


def test_spec_changes_fingerprint():
    assert ExperimentSpec().fingerprint() != ExperimentSpec(horizon=3).fingerprint()
    assert ExperimentSpec(features=('return_1',)).fingerprint() != ExperimentSpec().fingerprint()


def test_target_threshold_and_unknown_last_rows():
    spec = ExperimentSpec(horizon=3, target_type='threshold', threshold=.01)
    df = build_experiment_frame(prices(), spec)
    assert df.target.tail(3).isna().all()
    assert (df.target.dropna().isin([0, 1])).all()
    idx = 100
    assert df.target.iloc[idx] == int(df.target_forward_return.iloc[idx] > .01)


def test_feature_parameters_change_values():
    a = build_experiment_frame(prices(), ExperimentSpec())
    b = build_experiment_frame(prices(), ExperimentSpec(sma_short=30))
    assert a.sma_10_ratio.iloc[-20] != b.sma_10_ratio.iloc[-20]


def test_future_changes_do_not_change_past_features():
    a = prices()
    b = a.copy()
    b.loc[300:, 'close'] *= 4
    spec = ExperimentSpec()
    x = build_experiment_frame(a, spec)
    y = build_experiment_frame(b, spec)
    pd.testing.assert_frame_equal(x.loc[:250, list(spec.features)], y.loc[:250, list(spec.features)])


def test_training_and_regression_metrics():
    for kind in ('direction', 'return'):
        spec = ExperimentSpec(target_type=kind, features=('return_1', 'return_5', 'rsi_14'))
        models, meta = train_experiment(prices(), spec, 'TEST', '1d')
        assert len(models) == 2
        assert meta['spec_hash'] == spec.fingerprint()
        assert ('roc_auc' if kind == 'direction' else 'mae') in meta['metrics']['hist_gradient_boosting']


def test_invalid_feature_rejected():
    with pytest.raises(ValueError):
        ExperimentSpec(features=('target_forward_return',)).validate()
