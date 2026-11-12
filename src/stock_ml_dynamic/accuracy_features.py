"""Causal OHLCV features for Notebook 21. Never uses labels or future bars."""
from __future__ import annotations
import numpy as np
import pandas as pd

WINDOWS=(5,10,20,40,60)

def add_accuracy_features(frame: pd.DataFrame):
    df=frame.copy()
    df['bar_timestamp']=pd.to_datetime(df['bar_timestamp'],utc=True)
    df=df.sort_values(['symbol','interval_code','bar_timestamp']).reset_index(drop=True)
    keys=['symbol','interval_code']
    if df.duplicated(keys+['bar_timestamp']).any():raise ValueError('Duplicate price bars')
    close=pd.to_numeric(df['adj_close'],errors='coerce')
    if close.dropna().le(0).any():raise ValueError('Nonpositive prices')
    df['_close']=close
    grp=df.groupby(keys,sort=False)
    df['_ret']=grp['_close'].pct_change(fill_method=None)
    names=[]
    for w in WINDOWS:
        retgrp=df.groupby(keys,sort=False)['_ret']
        std=retgrp.transform(lambda s:s.rolling(w,min_periods=w).std())
        mean=retgrp.transform(lambda s:s.rolling(w,min_periods=w).mean())
        for name,series in [(f'volatility_{w}',std),(f'avg_return_{w}',mean),
                            (f'sharpe_like_{w}',mean/std.replace(0,np.nan)),
                            (f'downside_vol_{w}',retgrp.transform(lambda s:s.clip(upper=0).pow(2).rolling(w,min_periods=w).mean().pow(.5)))]:
            df[name]=series;names.append(name)
    for w in (5,20,60):
        r=df.groupby(keys,sort=False)['_ret']
        df[f'return_skew_{w}']=r.transform(lambda s:s.rolling(w,min_periods=w).skew())
        names.append(f'return_skew_{w}')
        df[f'return_z_{w}']=(df['_ret']-r.transform(lambda s:s.shift(1).rolling(w,min_periods=w).mean()))/r.transform(lambda s:s.shift(1).rolling(w,min_periods=w).std()).replace(0,np.nan)
        names.append(f'return_z_{w}')
    for w in (10,20,60):
        highs=grp['_close'].transform(lambda s:s.rolling(w,min_periods=w).max())
        df[f'drawdown_{w}']=df['_close']/highs-1
        names.append(f'drawdown_{w}')
    if 'volume' in df:
        v=pd.to_numeric(df['volume'],errors='coerce').where(lambda x:x>=0)
        df['_v']=v
        for w in (5,20,60):
            vg=df.groupby(keys,sort=False)['_v']
            prior=vg.transform(lambda s:s.shift(1).rolling(w,min_periods=w).mean())
            df[f'volume_ratio_{w}']=v/prior.replace(0,np.nan)
            names.append(f'volume_ratio_{w}')
        df.drop(columns=['_v'],inplace=True)
    df.drop(columns=['_close','_ret'],inplace=True)
    return df,names


def selective_metrics(y,probs,thresholds=(.35,.4,.45,.5,.55,.6)):
    """Diagnostic only: thresholds are not to be optimized on evaluation folds."""
    y=np.asarray(y,dtype=int);p=np.asarray(probs,dtype=float)
    if p.ndim!=2 or p.shape[1]!=3 or len(y)!=len(p):raise ValueError('Expected N x 3 probabilities')
    if not np.isfinite(p).all() or not np.allclose(p.sum(axis=1),1,atol=1e-5):raise ValueError('Invalid probabilities')
    confidence=p.max(axis=1);pred=p.argmax(axis=1)
    rows=[]
    for threshold in thresholds:
        mask=confidence>=threshold
        rows.append({'threshold':float(threshold),'coverage':float(mask.mean()),'n_selected':int(mask.sum()),
                     'selective_accuracy':float((pred[mask]==y[mask]).mean()) if mask.any() else np.nan,
                     'up_down_fraction':float(np.isin(pred[mask],[1,2]).mean()) if mask.any() else np.nan})
    return pd.DataFrame(rows)
