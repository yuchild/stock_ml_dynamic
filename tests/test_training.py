import numpy as np
import pandas as pd
import pytest
from stock_ml_dynamic.training import TrainingConfig, prepare_split, train_classifier


def sample_prices(n=420):
    rng = np.random.default_rng(42)
    close = 100 * np.exp(np.cumsum(rng.normal(0, .012, n)))
    return pd.DataFrame({"time": pd.date_range("2023-01-01", periods=n, freq="D"), "open": close * .999, "high": close * 1.01, "low": close * .99, "close": close, "volume": rng.integers(100, 500, n)})


def test_purged_train_labels_do_not_touch_test():
    train, test, info = prepare_split(sample_prices(), TrainingConfig(horizon=5))
    assert train.index.max() + 5 < test.index.min()
    assert info["purged_bars"] == 5


def test_training_metrics_and_baseline():
    pytest.importorskip("sklearn")
    models, meta = train_classifier(sample_prices())
    assert set(models) == {"majority_baseline", "hist_gradient_boosting"}
    assert 0 <= meta["metrics"]["hist_gradient_boosting"]["accuracy"] <= 1
    assert meta["train_rows"] > 100


def test_too_few_bars_rejected():
    with pytest.raises(ValueError, match="Insufficient"):
        prepare_split(sample_prices(100), TrainingConfig())
