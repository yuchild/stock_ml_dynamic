"""Pooled stock feature table, with reusable train-fitted transforms and inference rows.

No predictive model is trained here. Fits KMeans, its scaler, and quantile edges
only on a chronological training portion; never on validation/future rows.
"""
from dataclasses import dataclass
from pathlib import Path
import json
import re

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sqlalchemy import text

from .candle_table import (
    build_candle_table, DirectionQuantiles, CandleClusters,
    NUMERIC_FEATURES, CATEGORICAL_FEATURES, LABEL_MAPPING,
)
from .db import qualified_table

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
IDENTIFIERS = ['symbol', 'interval_code', 'bar_timestamp']
TARGET_COLUMNS = ['direction', 'next_adj_close', 'next_return']


def fetch_universe_prices(conn, interval='1d', bars_per_symbol=2000, symbols=None):
    """One read-only SQL query, keeping latest N rows for each ticker.

    If symbols=None, uses every ticker in the specified interval. Data is
    sorted chronologically within symbol after retrieval.
    """
    if not isinstance(bars_per_symbol, int) or bars_per_symbol < 50:
        raise ValueError('bars_per_symbol must be an integer >= 50')
    schema, table = qualified_table()
    args = {'interval': interval, 'limit': bars_per_symbol}
    symbol_clause = ''
    if symbols is not None:
        symbols = list(dict.fromkeys(symbols))
        if not symbols:
            raise ValueError('symbols must not be empty')
        symbol_clause = 'AND symbol = ANY(:symbols)'
        args['symbols'] = symbols
    sql = text(f'''
        WITH ranked AS (
            SELECT symbol, interval_code, bar_timestamp, open_price,
                   high_price, low_price, close_price, adj_close_price, volume,
                   ROW_NUMBER() OVER (
                       PARTITION BY symbol ORDER BY bar_timestamp DESC
                   ) AS rn
            FROM "{schema}"."{table}"
            WHERE interval_code = :interval {symbol_clause}
        )
        SELECT symbol, interval_code, bar_timestamp, open_price,
               high_price, low_price, close_price, adj_close_price, volume
        FROM ranked WHERE rn <= :limit
        ORDER BY symbol, bar_timestamp
    ''')
    result = pd.read_sql(sql, conn, params=args)
    if not result.empty:
        result['bar_timestamp'] = pd.to_datetime(result['bar_timestamp'], utc=True)
    return result


def build_universe_features(prices, z_window=30, timezone='America/New_York'):
    """Calculate causal features and next-bar outcomes independently per stock."""
    if prices.empty:
        raise ValueError('No price rows')
    required = set(IDENTIFIERS + ['open_price','high_price','low_price','close_price','adj_close_price'])
    if required - set(prices.columns):
        raise ValueError(f'Missing columns: {sorted(required - set(prices.columns))}')
    if prices.duplicated(IDENTIFIERS).any():
        raise ValueError('Duplicate symbol/interval/timestamp combinations')
    frames = []
    for (symbol, interval), group in prices.groupby(['symbol','interval_code'], sort=True):
        frame = build_candle_table(group, z_window=z_window, timestamp_timezone=timezone)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True).sort_values(IDENTIFIERS).reset_index(drop=True)


@dataclass
class MasterTableResult:
    master: pd.DataFrame
    inference: pd.DataFrame
    quantiles: DirectionQuantiles
    clusters: CandleClusters
    preprocessing: ColumnTransformer
    train_mask: pd.Series
    cutoff: pd.Timestamp


def assemble_master_table(features, train_fraction=0.8, selected_k=None, k_values=range(2,9)):
    """Fit pooled quantile edges and KMeans on earlier data; prepare labeled and live rows.

    A global timestamp cutoff is used for all stocks to avoid one ticker's
    future data being used when another ticker is in validation. A next-bar
    outcome is training-eligible only when the *next bar timestamp* is <= cutoff.
    """
    if not 0.5 <= train_fraction < 1:
        raise ValueError('train_fraction must be in [0.5, 1)')
    df = features.sort_values(IDENTIFIERS).reset_index(drop=True).copy()
    df['bar_timestamp'] = pd.to_datetime(df['bar_timestamp'], utc=True)
    if df.duplicated(IDENTIFIERS).any():
        raise ValueError('Duplicate stock/time rows')
    # Compute next timestamp per ticker to avoid labels crossing stock boundaries.
    df['next_bar_timestamp'] = df.groupby(['symbol','interval_code'], sort=False)['bar_timestamp'].shift(-1)
    unique_times = df['bar_timestamp'].drop_duplicates().sort_values().reset_index(drop=True)
    if len(unique_times) < 10:
        raise ValueError('Not enough distinct timestamps for chronological split')
    cutoff = unique_times.iloc[max(0, int(len(unique_times)*train_fraction)-1)]
    train_mask = (df['next_bar_timestamp'].notna() & (df['next_bar_timestamp'] <= cutoff))
    train_returns = df.loc[train_mask, 'next_return'].replace([np.inf,-np.inf],np.nan)
    quantiles = DirectionQuantiles().fit(train_returns)
    train_features = df.loc[df['bar_timestamp'] <= cutoff]
    clusters = CandleClusters().fit(train_features, k_values=k_values, selected_k=selected_k)
    df['direction'] = quantiles.transform(df['next_return'])
    df['candle_type'] = clusters.transform(df)
    # Latest bar per stock has no future target, but all its features remain
    # available for the next-interval prediction.
    inference = df[df['next_bar_timestamp'].isna()].copy()
    master = df[df['next_bar_timestamp'].notna() & df['next_adj_close'].notna() & df['direction'].notna()].copy()
    # Fit encoders and numeric normalization on train portion only. No future
    # features or target information enters the fitted preprocessor.
    fit_rows = master['next_bar_timestamp'] <= cutoff
    numeric = list(NUMERIC_FEATURES)
    categorical = list(CATEGORICAL_FEATURES)
    preprocessor = ColumnTransformer([
        ('numeric', StandardScaler(), numeric),
        ('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical),
    ], remainder='drop', verbose_feature_names_out=True)
    valid_train = master.loc[fit_rows, FEATURE_COLUMNS].replace([np.inf,-np.inf],np.nan).dropna()
    if valid_train.empty:
        raise ValueError('No complete training rows to fit preprocessing')
    preprocessor.fit(valid_train)
    return MasterTableResult(master, inference, quantiles, clusters, preprocessor, fit_rows, cutoff)


def model_ready(result, subset='master'):
    """Return transformed X, aligned rows and labels; drop incomplete features.

    subset='inference' gives latest per-stock rows for live prediction. Reuse
    result.preprocessing and result.clusters; do not refit on live data.
    """
    if subset not in {'master', 'inference'}:
        raise ValueError('subset must be master or inference')
    frame = getattr(result, subset)
    valid = frame[FEATURE_COLUMNS].replace([np.inf,-np.inf],np.nan).notna().all(axis=1)
    rows = frame.loc[valid].copy()
    transformed = result.preprocessing.transform(rows[FEATURE_COLUMNS])
    columns = result.preprocessing.get_feature_names_out()
    X = pd.DataFrame(transformed, columns=columns, index=rows.index)
    return X, rows


def save_master_outputs(result, directory='artifacts/master_tables', interval='1d'):
    """Persist table, latest-bar features, and fitted transforms for inference."""
    if not re.fullmatch(r'[A-Za-z0-9_]+', interval):
        raise ValueError('Invalid interval')
    dest = Path(directory) / interval
    dest.mkdir(parents=True, exist_ok=True)
    result.master.to_parquet(dest/'master_labeled.parquet', index=False)
    result.inference.to_parquet(dest/'latest_inference.parquet', index=False)
    joblib.dump({'quantiles':result.quantiles, 'clusters':result.clusters,
                 'preprocessing':result.preprocessing, 'features':FEATURE_COLUMNS,
                 'numeric':NUMERIC_FEATURES, 'categorical':CATEGORICAL_FEATURES,
                 'cutoff':str(result.cutoff), 'labels':LABEL_MAPPING}, dest/'transformers.joblib')
    summary = {'interval':interval, 'labeled_rows':len(result.master),
               'inference_rows':len(result.inference),
               'symbols':int(result.master['symbol'].nunique()),
               'train_cutoff':str(result.cutoff),
               'cluster_k':int(result.clusters.chosen_k_),
               'quantile_edges':result.quantiles.edges_.tolist(),
               'feature_columns':FEATURE_COLUMNS,
               'numeric_features':NUMERIC_FEATURES,
               'categorical_features':CATEGORICAL_FEATURES,
               'label_mapping':{str(k):v for k,v in LABEL_MAPPING.items()},
               'note':'Training labels and fitted preprocessing use only observations before cutoff; latest bar per symbol reserved for inference.'}
    (dest/'manifest.json').write_text(json.dumps(summary, indent=2))
    return dest


def add_forward_price_targets(frame: pd.DataFrame, horizons=range(1, 16)) -> pd.DataFrame:
    """Build t+1..t+N regression targets per stock; do not cross ticker boundaries.

    These targets are for supervised training only. A stock's last N rows have
    missing future values and must not be used for N-step supervised training.
    """
    result = frame.sort_values(IDENTIFIERS).copy()
    if result.duplicated(IDENTIFIERS).any():
        raise ValueError('Duplicate stock/time rows')
    horizons = tuple(sorted(set(int(h) for h in horizons)))
    if not horizons or any(h < 1 for h in horizons):
        raise ValueError('horizons must be positive')
    grouped = result.groupby(['symbol', 'interval_code'], sort=False)['adj_close']
    for horizon in horizons:
        result[f'adj_close_t+{horizon}'] = grouped.shift(-horizon)
    return result
