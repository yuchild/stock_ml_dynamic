import numpy as np
import pandas as pd
from stock_ml_dynamic.outlier_eda import fit_outlier_bounds, apply_outlier_bounds

def test_train_only_bounds_and_targets_unchanged():
    df=pd.DataFrame({'f':list(range(120))+[100000.], 'next_return':list(range(121)), 'direction':[1]*121})
    train=pd.Series([True]*120+[False])
    b=fit_outlier_bounds(df,['f'],train,lower=.05,upper=.95)
    assert b.loc['f','upper']<120
    clean,report=apply_outlier_bounds(df,b)
    assert clean.f.iloc[-1]==b.loc['f','upper']
    assert clean.next_return.equals(df.next_return)
    assert clean.direction.equals(df.direction)
    assert report.loc['f','n_high']>0

def test_constant_and_sparse_skipped():
    df=pd.DataFrame({'constant':[0.]*150,'sparse':[np.nan]*140+list(range(10))})
    b=fit_outlier_bounds(df,['constant','sparse'],pd.Series([True]*150))
    assert (b.status!='clipped').all()

def test_invalid_quantiles():
    import pytest
    with pytest.raises(ValueError):fit_outlier_bounds(pd.DataFrame({'x':[1]}),['x'],[True],.9,.1)
