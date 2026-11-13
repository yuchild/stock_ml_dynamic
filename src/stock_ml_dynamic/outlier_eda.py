"""Leakage-aware winsorization diagnostics for Notebook 30.

Never clip targets, prices, returns used for P&L, or categorical variables.
Fit bounds on chronological training rows only; apply unchanged to later rows.
"""
import numpy as np
import pandas as pd


def fit_outlier_bounds(frame, columns, train_mask, lower=.005, upper=.995,
                       min_finite=100, min_unique=10):
    if not 0 <= lower < upper <= 1:
        raise ValueError('Expected 0 <= lower < upper <= 1')
    train_mask = pd.Series(train_mask, index=frame.index).fillna(False).astype(bool)
    rows = []
    for col in columns:
        if col not in frame:
            raise KeyError(col)
        x = pd.to_numeric(frame.loc[train_mask, col], errors='coerce').replace([np.inf, -np.inf], np.nan).dropna()
        if len(x) < min_finite or x.nunique() < min_unique:
            rows.append(dict(feature=col, lower=np.nan, upper=np.nan, n_train=len(x), status='skipped_insufficient_variation'))
            continue
        lo, hi = x.quantile([lower, upper])
        if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
            rows.append(dict(feature=col, lower=np.nan, upper=np.nan, n_train=len(x), status='skipped_invalid_bounds'))
            continue
        rows.append(dict(feature=col, lower=float(lo), upper=float(hi), n_train=len(x), status='clipped'))
    return pd.DataFrame(rows).set_index('feature')


def apply_outlier_bounds(frame, bounds):
    result = frame.copy()
    report = []
    for col, row in bounds.iterrows():
        if row['status'] != 'clipped':
            report.append(dict(feature=col, n_low=0, n_high=0, clipped_pct=0., status=row['status']))
            continue
        x = pd.to_numeric(result[col], errors='coerce').replace([np.inf, -np.inf], np.nan)
        low = int((x < row['lower']).sum())
        high = int((x > row['upper']).sum())
        result[col] = x.clip(lower=row['lower'], upper=row['upper'])
        report.append(dict(feature=col, n_low=low, n_high=high,
                           clipped_pct=100 * (low + high) / max(1, x.notna().sum()), status='clipped'))
    return result, pd.DataFrame(report).set_index('feature')


def compare_numeric_profiles(before, after, columns):
    records = []
    for col in columns:
        a = pd.to_numeric(before[col], errors='coerce').replace([np.inf, -np.inf], np.nan)
        b = pd.to_numeric(after[col], errors='coerce').replace([np.inf, -np.inf], np.nan)
        records.append(dict(feature=col, raw_skew=a.skew(), cleaned_skew=b.skew(),
                            raw_std=a.std(), cleaned_std=b.std(),
                            raw_median=a.median(), cleaned_median=b.median(),
                            raw_missing_pct=100*a.isna().mean(), cleaned_missing_pct=100*b.isna().mean()))
    return pd.DataFrame(records).set_index('feature')
