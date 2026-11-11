"""Read-only diagnostics for saved pooled stock ML training/inference tables."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from .master_table import FEATURE_COLUMNS, IDENTIFIERS


def audit_saved_master(directory='artifacts/master_tables/1d'):
    """Return (summary, missing_latest, missing_features, violations) without refitting."""
    root = Path(directory)
    master = pd.read_parquet(root / 'master_labeled.parquet')
    latest = pd.read_parquet(root / 'latest_inference.parquet')
    bundle = joblib.load(root / 'transformers.joblib')
    manifest = json.loads((root / 'manifest.json').read_text())
    for df in (master, latest):
        df['bar_timestamp'] = pd.to_datetime(df['bar_timestamp'], utc=True)
    master['next_bar_timestamp'] = pd.to_datetime(master['next_bar_timestamp'], utc=True)
    cutoff = pd.Timestamp(bundle['cutoff'])
    if cutoff.tzinfo is None:
        cutoff = cutoff.tz_localize('UTC')
    missing = latest[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).isna()
    missing_latest = latest.loc[missing.any(axis=1), ['symbol','interval_code','bar_timestamp']].copy()
    missing_latest['missing_features'] = missing.loc[missing.any(axis=1)].apply(lambda row: ', '.join(row.index[row]), axis=1).values
    missing_features = pd.DataFrame({'feature': FEATURE_COLUMNS, 'latest_missing':missing.sum().values,
                                     'master_missing':master[FEATURE_COLUMNS].replace([np.inf,-np.inf],np.nan).isna().sum().values})
    violations = []
    if master.duplicated(IDENTIFIERS).any() or latest.duplicated(IDENTIFIERS).any():
        violations.append('Duplicate (symbol, interval_code, timestamp) rows')
    if master['next_bar_timestamp'].isna().any():
        violations.append('Labeled rows with missing next-bar timestamp')
    if (master['next_bar_timestamp'] <= master['bar_timestamp']).any():
        violations.append('Non-future next-bar timestamp in labeled data')
    if master[['direction','next_return','next_adj_close']].isna().any().any():
        violations.append('Missing labels or regression outcomes in labeled data')
    if latest['direction'].notna().any() or latest['next_adj_close'].notna().any() or latest['next_return'].notna().any():
        violations.append('Latest inference row has known future target')
    train = master['next_bar_timestamp'] <= cutoff
    test = master['bar_timestamp'] > cutoff
    if (train & test).any():
        violations.append('Training/test masks overlap')
    if not train.any() or not test.any():
        violations.append('Empty train or test partition')
    if bundle.get('features') != FEATURE_COLUMNS or manifest.get('feature_columns') != FEATURE_COLUMNS:
        violations.append('Feature-order mismatch between saved bundle and source code')
    if not hasattr(bundle['quantiles'], 'edges_') or bundle['quantiles'].edges_ is None:
        violations.append('Direction quantile boundaries missing')
    if not hasattr(bundle['clusters'], 'model_') or bundle['clusters'].model_ is None:
        violations.append('Fitted candle clusters missing')
    if not hasattr(bundle['preprocessing'], 'transformers_'):
        violations.append('Fitted preprocessor missing')
    all_symbols = set(master['symbol']).union(set(latest['symbol']))
    summary = {
        'interval':manifest.get('interval'), 'cutoff':str(cutoff),
        'master_rows':len(master), 'latest_rows':len(latest),
        'master_symbols':master['symbol'].nunique(), 'latest_symbols':latest['symbol'].nunique(),
        'total_symbols':len(all_symbols), 'train_rows_before_missing_filter':int(train.sum()),
        'test_rows_before_missing_filter':int(test.sum()),
        'latest_complete_rows':int((~missing.any(axis=1)).sum()),
        'latest_incomplete_rows':len(missing_latest),
        'latest_symbol_missing_entirely':sorted(set(master['symbol'])-set(latest['symbol'])),
        'manifest_symbol_count':manifest.get('symbols'),
        'quantile_edges':bundle['quantiles'].edges_.tolist(),
        'chosen_k':int(bundle['clusters'].chosen_k_),
        'encoded_features':len(bundle['preprocessing'].get_feature_names_out()),
        'violations':violations,
    }
    return summary, missing_latest, missing_features, violations
