"""Leakage-controlled pooled expanding-window evaluation.

Re-fits quantiles, candle clustering and preprocessing inside each fold.
Uses the saved causal raw feature columns, NOT saved global direction/candle_type.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import accuracy_score, balanced_accuracy_score, mean_absolute_error, mean_squared_error, r2_score
from .candle_table import DirectionQuantiles, CandleClusters, NUMERIC_FEATURES, CATEGORICAL_FEATURES

FEATURES = list(NUMERIC_FEATURES) + list(CATEGORICAL_FEATURES)

@dataclass(frozen=True)
class Fold:
    number: int
    train_end: pd.Timestamp
    test_end: pd.Timestamp


def expanding_folds(frame, n_folds=3, initial_fraction=0.55):
    """Global timestamp boundaries; every symbol shares the same periods."""
    if n_folds < 1 or not 0.3 <= initial_fraction < 0.9:
        raise ValueError('Invalid fold configuration')
    times = pd.DatetimeIndex(pd.to_datetime(frame['bar_timestamp'], utc=True).unique()).sort_values()
    if len(times) < 30:
        raise ValueError('Not enough timestamps')
    start = int(len(times) * initial_fraction)
    bounds = np.linspace(start, len(times)-1, n_folds+1, dtype=int)
    if len(set(bounds)) != len(bounds):
        raise ValueError('Not enough timestamps for distinct folds')
    return [Fold(i+1, times[bounds[i]], times[bounds[i+1]]) for i in range(n_folds)]


def fold_data(frame, fold, *, selected_k=4, cluster_fit_limit=30000, target_q=3):
    """Return fold-specific X/y and fitted transformations.

    Training labels are observed no later than train_end; test features are
    strictly later, and test outcomes no later than test_end. Last-bar rows
    are therefore never scored, although they remain eligible for inference.
    """
    df = frame.copy()
    for c in ('bar_timestamp','next_bar_timestamp'):
        df[c] = pd.to_datetime(df[c], utc=True)
    df = df.sort_values(['symbol','interval_code','bar_timestamp']).reset_index(drop=True)
    train = df['next_bar_timestamp'].notna() & (df['next_bar_timestamp'] <= fold.train_end)
    test = (df['bar_timestamp'] > fold.train_end) & (df['next_bar_timestamp'] <= fold.test_end)
    if not train.any() or not test.any():
        raise ValueError('Empty train/test fold')
    quantiles = DirectionQuantiles(q=target_q).fit(df.loc[train,'next_return'])
    # Fit clustering only with training-period candles; cap rows to keep notebook practical.
    cluster_rows = df.loc[df['bar_timestamp'] <= fold.train_end]
    if len(cluster_rows) > cluster_fit_limit:
        cluster_rows = cluster_rows.sample(cluster_fit_limit, random_state=42)
    clusters = CandleClusters().fit(cluster_rows, selected_k=selected_k, k_values=range(2,9))
    df['candle_type'] = clusters.transform(df)
    df['direction'] = quantiles.transform(df['next_return'])
    complete = df[FEATURES].replace([np.inf,-np.inf],np.nan).notna().all(axis=1)
    train &= complete & df['direction'].notna()
    test &= complete & df['direction'].notna()
    if train.sum() < 30 or test.sum() < 10:
        raise ValueError('Too few complete observations')
    transformer = ColumnTransformer([
        ('numeric',StandardScaler(),list(NUMERIC_FEATURES)),
        ('categorical',OneHotEncoder(handle_unknown='ignore',sparse_output=False),list(CATEGORICAL_FEATURES))
    ], remainder='drop')
    transformer.fit(df.loc[train, FEATURES])
    X_train = transformer.transform(df.loc[train,FEATURES])
    X_test = transformer.transform(df.loc[test,FEATURES])
    return df.loc[train], df.loc[test], X_train, X_test, {'quantiles':quantiles,'clusters':clusters,'preprocessing':transformer,'features':FEATURES}


def evaluate_fold(train, test, X_train, X_test, *, fold_number):
    """Compare baseline and gradient boosting; never train on test data."""
    reports=[]
    yc_train=train['direction'].astype(int)
    yc_test=test['direction'].astype(int)
    for name, model in {
        'majority_baseline':DummyClassifier(strategy='prior'),
        'hist_gradient_boosting':HistGradientBoostingClassifier(max_iter=150,max_leaf_nodes=15,learning_rate=.05,l2_regularization=1,random_state=42)
    }.items():
        model.fit(X_train,yc_train)
        pred=model.predict(X_test)
        reports.append({'fold':fold_number,'task':'classification','model':name,'train_rows':len(train),'test_rows':len(test),
                        'accuracy':accuracy_score(yc_test,pred),'balanced_accuracy':balanced_accuracy_score(yc_test,pred)})
    valid_train=np.isfinite(train['next_return']) & (train['adj_close']>0)
    valid_test=np.isfinite(test['next_return']) & (test['adj_close']>0)
    if valid_train.sum() < 30 or valid_test.sum() < 10:
        raise ValueError('Insufficient valid regression rows')
    yr_train=train.loc[valid_train,'next_return'].astype(float)
    yr_test=test.loc[valid_test,'next_return'].astype(float)
    for name,model in {
        'mean_return_baseline':DummyRegressor(strategy='mean'),
        'hist_gradient_boosting':HistGradientBoostingRegressor(max_iter=150,max_leaf_nodes=15,learning_rate=.05,l2_regularization=1,random_state=42)
    }.items():
        model.fit(X_train[valid_train],yr_train)
        pred=model.predict(X_test[valid_test])
        reports.append({'fold':fold_number,'task':'regression','model':name,'train_rows':int(valid_train.sum()),'test_rows':int(valid_test.sum()),
                        'return_mae':mean_absolute_error(yr_test,pred),
                        'return_rmse':np.sqrt(mean_squared_error(yr_test,pred)),
                        'return_r2':r2_score(yr_test,pred),
                        'price_mae':mean_absolute_error(test.loc[valid_test,'next_adj_close'],test.loc[valid_test,'adj_close']*(1+pred))})
    return pd.DataFrame(reports)


def diagnose_latest_z(prices, symbol='VYLR', window=30):
    """Explain missing latest z-score from raw adjusted closes; no imputation."""
    part=prices.loc[prices['symbol'].eq(symbol)].sort_values('bar_timestamp').copy()
    if part.empty:
        raise ValueError(f'{symbol} not found')
    adj=pd.to_numeric(part['adj_close_price'],errors='coerce')
    trailing=adj.iloc[-(window+1):-1]
    return {'symbol':symbol,'total_bars':len(part),'trailing_count':len(trailing),
            'trailing_valid':int(np.isfinite(trailing).sum()),
            'latest_price':float(adj.iloc[-1]) if pd.notna(adj.iloc[-1]) else None,
            'trailing_mean':float(trailing.mean()) if len(trailing) else None,
            'trailing_std':float(trailing.std(ddof=1)) if len(trailing)>1 else None,
            'latest_z_missing':bool(len(trailing)<window or not np.isfinite(trailing).all() or not np.isfinite(adj.iloc[-1]))}
