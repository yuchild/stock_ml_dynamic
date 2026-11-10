"""Configurable features and targets; preserve legacy defaults in features.py."""
import numpy as np
import pandas as pd
from .experiment import ExperimentSpec


def build_experiment_frame(prices: pd.DataFrame, spec: ExperimentSpec) -> pd.DataFrame:
    spec.validate()
    required = {'time', 'open', 'high', 'low', 'close', 'volume'}
    if required - set(prices.columns):
        raise ValueError(f'Missing price columns: {sorted(required - set(prices.columns))}')
    df = prices.copy().sort_values('time').reset_index(drop=True)
    if df.time.duplicated().any():
        raise ValueError('Duplicate timestamps')
    for col in ('open', 'high', 'low', 'close', 'volume'):
        df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)
    close = df.close
    ret = close.pct_change(fill_method=None)
    for n in (1, 5, 20):
        df[f'return_{n}'] = close.pct_change(n, fill_method=None)
    for n, label in ((spec.sma_short, 'sma_10_ratio'), (spec.sma_medium, 'sma_20_ratio'), (spec.sma_long, 'sma_50_ratio')):
        df[label] = close / close.rolling(n, min_periods=n).mean() - 1
    df['volatility_20'] = ret.rolling(spec.volatility_window, min_periods=spec.volatility_window).std()
    df['volume_ratio_20'] = df.volume / df.volume.rolling(spec.volume_window, min_periods=spec.volume_window).mean()
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/spec.rsi_window, adjust=False, min_periods=spec.rsi_window).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/spec.rsi_window, adjust=False, min_periods=spec.rsi_window).mean()
    rs = gain / loss.replace(0, np.nan)
    df['rsi_14'] = 100 - 100 / (1 + rs)
    df.loc[(loss == 0) & (gain > 0), 'rsi_14'] = 100
    df.loc[(loss == 0) & (gain == 0), 'rsi_14'] = 50
    ema12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
    df['macd_ratio'] = (ema12 - ema26) / close
    df['range_ratio'] = (df.high - df.low) / close
    df['body_ratio'] = (df.close - df.open) / df.open
    df['target_forward_return'] = close.shift(-spec.horizon) / close - 1
    df['target'] = np.nan
    valid = df.target_forward_return.notna()
    if spec.target_type == 'return':
        df.loc[valid, 'target'] = df.loc[valid, 'target_forward_return']
    else:
        threshold = 0.0 if spec.target_type == 'direction' else spec.threshold
        df.loc[valid, 'target'] = (df.loc[valid, 'target_forward_return'] > threshold).astype(int)
    return df.replace([np.inf, -np.inf], np.nan)
