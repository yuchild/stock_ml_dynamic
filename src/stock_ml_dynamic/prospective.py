"""Frozen next-bar classifier: fitted artifact and append-only paper prediction ledger.

Never uses outcomes after an as-of timestamp to construct a prediction. Predictions
are immutable; later observed bars can be used only to resolve previously recorded
predictions. The training label boundaries are saved with the model version.
"""
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os
import time
import uuid
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, log_loss
from .candle_table import DirectionQuantiles, CandleClusters, NUMERIC_FEATURES, CATEGORICAL_FEATURES
from .classifier_benchmark import eligible_universe
from .historical_market import add_historical_market_features
from .final_holdout import FROZEN_C, EXCLUDED_EXTRA

CLASS_IDS = (0, 1, 2)
KEY = ['model_version', 'interval_code', 'symbol', 'bar_timestamp']


def prepare_universe(raw_prices):
    """Build causal features from current raw OHLCV, including latest unlabeled bars."""
    from .master_table import build_universe_features
    if raw_prices.empty:
        raise ValueError('No OHLCV prices supplied')
    frame = build_universe_features(raw_prices)
    frame['next_bar_timestamp'] = frame.groupby(['symbol', 'interval_code'], sort=False)['bar_timestamp'].shift(-1)
    eligible, eligibility = eligible_universe(frame)
    expanded, extra = add_historical_market_features(eligible)
    return expanded, [c for c in extra if c not in EXCLUDED_EXTRA], eligibility


def _finite(frame, columns):
    return frame[columns].replace([np.inf, -np.inf], np.nan).notna().all(axis=1)


def train_frozen_model(frame, extra, *, max_train_rows=120000, cluster_fit_limit=15000,
                       selected_k=4, random_state=42):
    """Fit the preselected configuration using only fully observed next-bar labels.

    This is a *new prospective deployment model*, not the untouched holdout
    evaluation model. Training on formerly held-out history is now permitted
    because hyperparameters are frozen and only future predictions are scored.
    """
    t0 = time.perf_counter()
    df = frame.sort_values(['symbol', 'interval_code', 'bar_timestamp']).reset_index(drop=True).copy()
    df['bar_timestamp'] = pd.to_datetime(df['bar_timestamp'], utc=True)
    df['next_bar_timestamp'] = pd.to_datetime(df['next_bar_timestamp'], utc=True)
    labeled = df['next_bar_timestamp'].notna() & np.isfinite(pd.to_numeric(df['next_return'], errors='coerce'))
    if labeled.sum() < 100:
        raise ValueError('Insufficient observed training outcomes')
    quantiles = DirectionQuantiles().fit(df.loc[labeled, 'next_return'])
    cluster_rows = df.loc[labeled]
    if len(cluster_rows) > cluster_fit_limit:
        cluster_rows = cluster_rows.sample(cluster_fit_limit, random_state=random_state)
    clusters = CandleClusters().fit(cluster_rows, selected_k=selected_k, k_values=range(2, 9))
    df['candle_type'] = clusters.transform(df)
    df['direction'] = quantiles.transform(df['next_return'])
    base_cols = list(NUMERIC_FEATURES) + list(CATEGORICAL_FEATURES)
    fit_mask = labeled & _finite(df, base_cols + list(extra)) & df['direction'].notna()
    if fit_mask.sum() < 100:
        raise ValueError('Insufficient complete training features')
    prep = ColumnTransformer([
        ('numeric', StandardScaler(), list(NUMERIC_FEATURES)),
        ('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=False), list(CATEGORICAL_FEATURES)),
    ], remainder='drop')
    prep.fit(df.loc[fit_mask, base_cols])
    scaler = StandardScaler().fit(df.loc[fit_mask, extra])
    x = np.column_stack([prep.transform(df.loc[fit_mask, base_cols]), scaler.transform(df.loc[fit_mask, extra])])
    y = df.loc[fit_mask, 'direction'].to_numpy(dtype=int)
    if set(np.unique(y)) != set(CLASS_IDS):
        raise ValueError('All three classes required')
    if max_train_rows and len(y) > max_train_rows:
        ids = np.sort(np.random.default_rng(random_state).choice(len(y), max_train_rows, replace=False))
        x, y = x[ids], y[ids]
    model = LogisticRegression(C=FROZEN_C, max_iter=600, solver='lbfgs')
    model.fit(x, y)
    training_end = df.loc[fit_mask, 'next_bar_timestamp'].max()
    version = 'logistic_l2_' + pd.Timestamp(training_end).strftime('%Y%m%dT%H%M%SZ')
    bundle = {'model': model, 'quantiles': quantiles, 'clusters': clusters,
              'preprocessing': prep, 'extra_scaler': scaler,
              'base_columns': base_cols, 'extra_columns': list(extra),
              'model_version': version, 'training_end': str(training_end),
              'C': FROZEN_C, 'train_rows': int(len(y)),
              'training_seconds': time.perf_counter()-t0,
              'selected_k': selected_k, 'random_state': random_state,
              'target': 'next-return tertiles; edges fitted on observed training outcomes'}
    return bundle


def save_model(bundle, root):
    """Save a versioned, immutable joblib bundle and a JSON manifest."""
    directory = Path(root) / bundle['model_version']
    directory.mkdir(parents=True, exist_ok=True)
    file = directory / 'model.joblib'
    if file.exists():
        raise FileExistsError(f'Model version already exists: {file}')
    tmp = directory / ('.model_' + uuid.uuid4().hex + '.tmp')
    joblib.dump(bundle, tmp)
    os.replace(tmp, file)
    manifest = {k: v for k, v in bundle.items() if k in ('model_version','training_end','C','train_rows','training_seconds','selected_k','target','base_columns','extra_columns','random_state')}
    manifest['sha256'] = hashlib.sha256(file.read_bytes()).hexdigest()
    (directory / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    return file


def load_model(path):
    """Only load locally trusted model artifacts; joblib uses pickle."""
    path = Path(path)
    bundle = joblib.load(path)
    expected = path.parent / 'manifest.json'
    if expected.exists():
        manifest = json.loads(expected.read_text())
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest['sha256']:
            raise ValueError('Model artifact checksum mismatch')
    return bundle


def predict_latest(frame, bundle):
    """Predict only the latest observed bar for each eligible ticker.

    Requires a freshly rebuilt causal feature table. Never fits or updates
    the saved transformers on inference rows.
    """
    df = frame.sort_values(['symbol','interval_code','bar_timestamp']).groupby(['symbol','interval_code'], sort=False).tail(1).copy()
    df['candle_type'] = bundle['clusters'].transform(df)
    columns = bundle['base_columns'] + bundle['extra_columns']
    valid = _finite(df, columns)
    df = df.loc[valid].copy()
    if df.empty:
        raise ValueError('No inference-ready stocks')
    xb = bundle['preprocessing'].transform(df[bundle['base_columns']])
    xe = bundle['extra_scaler'].transform(df[bundle['extra_columns']])
    p = bundle['model'].predict_proba(np.column_stack([xb, xe]))
    out = df[['symbol','interval_code','bar_timestamp','adj_close']].reset_index(drop=True).copy()
    out['model_version'] = bundle['model_version']
    for j, cls in enumerate(bundle['model'].classes_):
        out[f'prob_{int(cls)}'] = p[:,j]
    out['predicted_direction'] = bundle['model'].predict(np.column_stack([xb, xe])).astype(int)
    out['recorded_at_utc'] = pd.Timestamp.now(tz='UTC').isoformat()
    out['training_end'] = bundle['training_end']
    out['target_lower_edge'] = float(bundle['quantiles'].edges_[1])
    out['target_upper_edge'] = float(bundle['quantiles'].edges_[-2])
    out['outcome_timestamp'] = pd.NaT
    out['outcome_adj_close'] = np.nan
    out['outcome_return'] = np.nan
    out['actual_direction'] = pd.Series(pd.NA, index=out.index, dtype='Int64')
    out['resolved_at_utc'] = pd.Series(pd.NA, index=out.index, dtype='string')
    return out


def append_predictions(new_rows, ledger_path):
    """Append new as-of predictions; existing keys cannot be changed.

    Use a single writer for the ledger. The atomic replacement protects against
    partial writes, but is not a distributed multi-process lock.
    """
    path = Path(ledger_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = new_rows.copy()
    new['bar_timestamp'] = pd.to_datetime(new['bar_timestamp'], utc=True)
    if new.duplicated(KEY).any():
        raise ValueError('Duplicate prediction keys in new batch')
    if path.exists():
        existing = pd.read_parquet(path)
        existing['bar_timestamp'] = pd.to_datetime(existing['bar_timestamp'], utc=True)
        unique = pd.MultiIndex.from_frame(new[KEY]).isin(pd.MultiIndex.from_frame(existing[KEY]))
        new = new.loc[~unique].copy()
        result = pd.concat([existing, new], ignore_index=True)
    else:
        result = new
    if result.duplicated(KEY).any():
        raise ValueError('Duplicate keys in prediction ledger')
    tmp = path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.tmp')
    result.to_parquet(tmp, index=False)
    os.replace(tmp, path)
    return len(new), len(result)


def reconcile_predictions(ledger_path, raw_prices):
    """Resolve predictions against first strictly later *observed* bar per ticker.

    Never re-predict, overwrite probabilities, or use the next return in
    features. The first subsequent observed bar is treated as next interval;
    missing periods are flagged by the timestamps for human review.
    """
    path = Path(ledger_path)
    if not path.exists():
        raise FileNotFoundError(path)
    ledger = pd.read_parquet(path)
    if ledger.empty:
        return ledger, 0
    prices = raw_prices[['symbol','interval_code','bar_timestamp','adj_close_price']].copy()
    prices['bar_timestamp'] = pd.to_datetime(prices['bar_timestamp'], utc=True)
    prices = prices.sort_values(['symbol','interval_code','bar_timestamp'])
    ledger['bar_timestamp'] = pd.to_datetime(ledger['bar_timestamp'], utc=True)
    unresolved = ledger['outcome_timestamp'].isna()
    changed = 0
    for idx in ledger.index[unresolved]:
        row = ledger.loc[idx]
        later = prices.loc[(prices['symbol'] == row['symbol']) &
                           (prices['interval_code'] == row['interval_code']) &
                           (prices['bar_timestamp'] > row['bar_timestamp'])]
        if later.empty:
            continue
        bar = later.iloc[0]
        future_price = float(bar['adj_close_price'])
        old_price = float(row['adj_close'])
        if not np.isfinite(future_price) or future_price <= 0 or not np.isfinite(old_price) or old_price <= 0:
            continue
        ret = future_price / old_price - 1.0
        if ret <= float(row['target_lower_edge']):
            actual = 2
        elif ret <= float(row['target_upper_edge']):
            actual = 0
        else:
            actual = 1
        ledger.at[idx, 'outcome_timestamp'] = bar['bar_timestamp']
        ledger.at[idx, 'outcome_adj_close'] = future_price
        ledger.at[idx, 'outcome_return'] = ret
        ledger.at[idx, 'actual_direction'] = actual
        ledger.at[idx, 'resolved_at_utc'] = pd.Timestamp.now(tz='UTC').isoformat()
        changed += 1
    if changed:
        tmp = path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.tmp')
        ledger.to_parquet(tmp, index=False)
        os.replace(tmp, path)
    return ledger, changed


def prospective_scores(ledger, min_rows=30):
    """Only resolved predictions; no in-sample performance claims."""
    if ledger.empty:
        return pd.DataFrame(columns=['model_version','resolved_rows','accuracy',
                                     'balanced_accuracy','roc_auc_ovr_macro','log_loss'])
    resolved = ledger.loc[ledger['actual_direction'].notna()].copy()
    reports = []
    for version, rows in resolved.groupby('model_version'):
        y = rows['actual_direction'].astype(int).to_numpy()
        pred = rows['predicted_direction'].astype(int).to_numpy()
        p = rows[[f'prob_{i}' for i in CLASS_IDS]].to_numpy(dtype=float)
        record = {'model_version': version, 'resolved_rows': len(rows),
                  'accuracy': float(accuracy_score(y,pred)),
                  'balanced_accuracy': float(balanced_accuracy_score(y,pred)) if len(np.unique(y)) > 1 else np.nan,
                  'roc_auc_ovr_macro': np.nan, 'log_loss': np.nan}
        if len(rows) >= min_rows and set(np.unique(y)) == set(CLASS_IDS):
            record['roc_auc_ovr_macro'] = float(roc_auc_score(y,p,labels=list(CLASS_IDS),multi_class='ovr',average='macro'))
            record['log_loss'] = float(log_loss(y,p,labels=list(CLASS_IDS)))
        reports.append(record)
    return pd.DataFrame(reports, columns=['model_version','resolved_rows','accuracy','balanced_accuracy','roc_auc_ovr_macro','log_loss'])
