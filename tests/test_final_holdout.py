import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from stock_ml_dynamic.final_holdout import holdout_boundaries, FROZEN_C, EXCLUDED_EXTRA

def test_holdout_boundaries_are_chronological():
    dates = pd.date_range('2024-01-01', periods=120, freq='D', tz='UTC')
    frame = pd.DataFrame({'bar_timestamp': dates, 'next_bar_timestamp': dates + pd.Timedelta(days=1)})
    boundary, last = holdout_boundaries(frame)
    assert boundary == dates[95]
    assert last == dates[-1] + pd.Timedelta(days=1)
    assert (frame.bar_timestamp > boundary).sum() == 24

def test_frozen_configuration():
    assert FROZEN_C == 0.1
    assert EXCLUDED_EXTRA == ('realized_vol_10',)

def test_holdout_boundaries_reject_no_outcomes():
    dates = pd.date_range('2024-01-01', periods=120, freq='D', tz='UTC')
    frame = pd.DataFrame({'bar_timestamp': dates, 'next_bar_timestamp': [pd.NaT] * 120})
    with pytest.raises(ValueError):
        holdout_boundaries(frame)
