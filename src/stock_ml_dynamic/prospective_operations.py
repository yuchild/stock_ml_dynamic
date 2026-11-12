"""Operational gates and reporting for single-writer prospective paper evaluation.

No model training here. Never infer a completed session from a midnight daily-bar
stamp: operator confirmation is required before committing new predictions.
"""
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import uuid
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score, log_loss, roc_auc_score
from .prospective import append_predictions, reconcile_predictions, prospective_scores, predict_latest, CLASS_IDS


def source_audit(raw_prices, *, interval='1d', min_bars=31, max_lag_days=4, now=None):
    """Inspect data; do not assume exchange holiday calendars or bar completeness."""
    now = pd.Timestamp.now(tz='UTC') if now is None else pd.Timestamp(now)
    if now.tzinfo is None:
        now = now.tz_localize('UTC')
    else:
        now = now.tz_convert('UTC')
    needed = {'symbol','interval_code','bar_timestamp','adj_close_price','open_price','high_price','low_price','close_price','volume'}
    if missing := needed - set(raw_prices.columns):
        raise ValueError(f'Missing OHLCV columns: {sorted(missing)}')
    df = raw_prices.loc[raw_prices.interval_code.eq(interval)].copy()
    if df.empty:
        raise ValueError(f'No bars for {interval}')
    df['bar_timestamp'] = pd.to_datetime(df.bar_timestamp, utc=True)
    if df.duplicated(['symbol','interval_code','bar_timestamp']).any():
        raise ValueError('Duplicate OHLCV symbol/interval/timestamp rows')
    if (df.bar_timestamp > now).any():
        raise ValueError('Future-dated source bars')
    counts = df.groupby('symbol').size().rename('bar_count')
    latest = df.sort_values('bar_timestamp').groupby('symbol').tail(1).set_index('symbol')
    summary = pd.DataFrame({'bar_count':counts, 'latest_bar':latest.bar_timestamp,
                            'latest_adj_close':latest.adj_close_price})
    summary['eligible_history'] = summary.bar_count >= min_bars
    summary['lag_days'] = (now - summary.latest_bar).dt.total_seconds() / 86400
    summary['stale'] = summary.lag_days > max_lag_days
    summary['invalid_latest_price'] = (~np.isfinite(pd.to_numeric(summary.latest_adj_close,errors='coerce'))) | (summary.latest_adj_close <= 0)
    summary['latest_is_modal'] = summary.latest_bar.eq(summary.latest_bar.mode().iloc[0])
    latest_rows = df.loc[df.bar_timestamp.eq(df.bar_timestamp.max())]
    report = {'interval':interval,'as_of_utc':now.isoformat(), 'symbols':int(len(summary)),
              'history_eligible_symbols':int(summary.eligible_history.sum()),
              'modal_latest_bar':str(summary.latest_bar.mode().iloc[0]),
              'symbols_at_modal_latest':int(summary.latest_is_modal.sum()),
              'stale_symbols':int(summary.stale.sum()),
              'future_bars':0,'duplicate_bars':0,
              'max_source_bar':str(df.bar_timestamp.max()),
              'symbols_at_global_latest':int(latest_rows.symbol.nunique())}
    return summary.reset_index().sort_values('symbol'), report


def recording_gate(predictions, raw_prices, bundle, *, confirmed_session_date=None,
                   operator_confirmed=False, min_coverage=0.95, max_recording_delay_days=4, now=None):
    """Fail closed unless a human confirms the exchange session is final.

    Session date is compared in America/New_York, but timestamps from the source
    may be midnight UTC date labels, so this does NOT validate market close.
    """
    if not operator_confirmed:
        raise PermissionError('Set operator_confirmed=True only after confirming source bars are final')
    if confirmed_session_date is None:
        raise ValueError('Explicit confirmed_session_date (YYYY-MM-DD) is required')
    if not 0 < min_coverage <= 1:
        raise ValueError('min_coverage must be in (0,1]')
    date = pd.Timestamp(confirmed_session_date).date()
    now = pd.Timestamp.now(tz='UTC') if now is None else pd.Timestamp(now)
    now = now.tz_localize('UTC') if now.tzinfo is None else now.tz_convert('UTC')
    if date > now.tz_convert('America/New_York').date():
        raise ValueError('Cannot confirm a future New York session')
    if (now.tz_convert('America/New_York').date()-date).days > max_recording_delay_days:
        raise ValueError('Stale session: do not backfill retrospective predictions as prospective')
    if predictions.empty or predictions.duplicated(['symbol','interval_code','bar_timestamp']).any():
        raise ValueError('Empty or duplicate prediction batch')
    if predictions.model_version.nunique() != 1 or predictions.model_version.iloc[0] != bundle['model_version']:
        raise ValueError('Model version mismatch')
    if set(pd.to_datetime(predictions.bar_timestamp, utc=True).dt.date) != {date}:
        raise ValueError('Prediction bar dates differ from explicitly confirmed session')
    prices = raw_prices.copy()
    prices['bar_timestamp'] = pd.to_datetime(prices.bar_timestamp, utc=True)
    if (prices.bar_timestamp.dt.date > date).any():
        raise ValueError('Newer observed bars exist: prediction is retrospective')
    all_latest = prices.sort_values('bar_timestamp').groupby(['symbol','interval_code']).tail(1)
    at_session = all_latest.loc[all_latest.bar_timestamp.dt.date.eq(date)]
    eligible_count = int(len(at_session))
    if not eligible_count or len(predictions) / eligible_count < min_coverage:
        raise ValueError(f'Insufficient prediction coverage: {len(predictions)}/{eligible_count}')
    train_end = pd.Timestamp(bundle['training_end'])
    train_end = train_end.tz_localize('UTC') if train_end.tzinfo is None else train_end.tz_convert('UTC')
    if train_end > pd.to_datetime(predictions.bar_timestamp, utc=True).min():
        raise ValueError('Model training end is after prediction bar timestamp')
    return {'session_date':date.isoformat(),'model_version':bundle['model_version'],
            'predictions':int(len(predictions)),'source_latest_symbols':eligible_count,
            'coverage':float(len(predictions)/eligible_count),'confirmed_at_utc':now.isoformat()}


def ledger_audit(ledger):
    """Validate immutable prediction fields and summarize unresolved outcomes."""
    if ledger.empty:
        return {'rows':0,'resolved':0,'pending':0,'duplicate_keys':0,'invalid_probabilities':0}, pd.DataFrame()
    keys=['model_version','interval_code','symbol','bar_timestamp']
    duplicates=int(ledger.duplicated(keys).sum())
    p=ledger[[f'prob_{i}' for i in CLASS_IDS]].to_numpy(dtype=float)
    invalid=int((~np.isfinite(p).all(axis=1) | (p < 0).any(axis=1) | (np.abs(p.sum(axis=1)-1)>1e-5)).sum())
    if duplicates or invalid:
        raise ValueError(f'Ledger integrity failure: duplicate_keys={duplicates}, invalid_probabilities={invalid}')
    resolved=ledger.actual_direction.notna()
    by_version=ledger.assign(resolved=resolved).groupby('model_version',as_index=False).agg(
        predictions=('symbol','size'),resolved=('resolved','sum'),
        first_bar=('bar_timestamp','min'),last_bar=('bar_timestamp','max'))
    by_version['pending']=by_version.predictions-by_version.resolved
    return {'rows':int(len(ledger)),'resolved':int(resolved.sum()),'pending':int((~resolved).sum()),
            'duplicate_keys':duplicates,'invalid_probabilities':invalid},by_version


def evaluation_report(ledger, *, min_rows=30):
    """Score only resolved predictions and include per-class recall and counts."""
    if ledger.empty:
        return prospective_scores(ledger,min_rows=min_rows), pd.DataFrame(
            columns=['model_version','class_id','support','recall','predicted_count'])
    summary=prospective_scores(ledger,min_rows=min_rows)
    details=[]
    for version,group in ledger.loc[ledger.actual_direction.notna()].groupby('model_version'):
        y=group.actual_direction.astype(int).to_numpy()
        pred=group.predicted_direction.astype(int).to_numpy()
        cm=confusion_matrix(y,pred,labels=list(CLASS_IDS))
        for idx,cls in enumerate(CLASS_IDS):
            n=int(cm[idx].sum())
            details.append({'model_version':version,'class_id':cls,'support':n,
                            'recall':float(cm[idx,idx]/n) if n else np.nan,
                            'predicted_count':int(cm[:,idx].sum())})
    return summary,pd.DataFrame(details,columns=['model_version','class_id','support','recall','predicted_count'])


def save_operational_report(root, *, source_report, gate_report, ledger_report, scores, per_class):
    """Write run-specific reports without overwriting previous run snapshots."""
    root=Path(root); root.mkdir(parents=True,exist_ok=True)
    run_id=pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%S%fZ')+'_'+uuid.uuid4().hex[:6]
    folder=root/'runs'/run_id
    folder.mkdir(parents=True,exist_ok=False)
    (folder/'audit.json').write_text(json.dumps({'source':source_report,'recording_gate':gate_report,'ledger':ledger_report},indent=2,default=str))
    scores.to_csv(folder/'scores.csv',index=False)
    per_class.to_csv(folder/'per_class.csv',index=False)
    return folder
