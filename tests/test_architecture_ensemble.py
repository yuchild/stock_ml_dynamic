import numpy as np
import pytest
from stock_ml_dynamic.architecture_ensemble import blend_probabilities,normalize_probabilities,chronological_meta_indices
import pandas as pd

def test_blend_normalizes():
    p=np.array([[.2,.3,.5],[.5,.3,.2]])
    q=np.array([[.4,.4,.2],[.1,.2,.7]])
    out=blend_probabilities([p,q],[1,1]);assert np.allclose(out.sum(axis=1),1)
    assert np.allclose(out,(p+q)/2)

def test_invalid_blend():
    with pytest.raises(ValueError):blend_probabilities([np.ones((2,3))],[-1])
    with pytest.raises(ValueError):normalize_probabilities(np.array([[float('nan'),.3,.7]]))

def test_temporal_meta_split():
    dates=pd.date_range('2024-01-01',periods=200,tz='UTC')
    rows=pd.DataFrame({'next_bar_timestamp':dates})
    mask,cut=chronological_meta_indices(rows)
    assert mask.sum()==160 and (~mask).sum()==40
    assert rows.loc[mask,'next_bar_timestamp'].max()<rows.loc[~mask,'next_bar_timestamp'].min()

def test_residual_forward():
    torch=pytest.importorskip('torch')
    from stock_ml_dynamic.architecture_ensemble import ResidualMLP
    net=ResidualMLP(11,hidden=16).net
    assert net(torch.randn(4,11)).shape==(4,3)
