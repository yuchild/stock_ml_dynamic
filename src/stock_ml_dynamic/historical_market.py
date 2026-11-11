"""Causal historical and equal-weight-universe-relative next-bar classifier benchmark.

All features at bar t use data at or before t. The market proxy is the
contemporaneous equal-weight return of other eligible stocks, not SPY.
"""
from __future__ import annotations
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss, roc_auc_score, roc_curve
from .classifier_benchmark import eligible_universe, LABELS
from .walk_forward import expanding_folds, fold_data

HIST_WINDOWS = (5, 10, 20)
LAGS = (1, 2, 3, 5, 10, 20)


def add_historical_market_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Compute stock-specific and contemporaneous cross-sectional features.

    Requires adjusted close, bar timestamp, symbol; volume is optional. Does
    not use next_return or direction. Market proxy excludes the stock itself.
    No cross-sectional features are calculated from future bars.
    """
    df = frame.copy().sort_values(['symbol', 'interval_code', 'bar_timestamp']).reset_index(drop=True)
    if df.duplicated(['symbol', 'interval_code', 'bar_timestamp']).any():
        raise ValueError('Duplicate symbol/interval/timestamp')
    df['bar_timestamp'] = pd.to_datetime(df['bar_timestamp'], utc=True)
    price = pd.to_numeric(df['adj_close'], errors='coerce')
    if (price.dropna() <= 0).any():
        raise ValueError('Adjusted closes must be positive')
    df['_price'] = price
    g = df.groupby(['symbol', 'interval_code'], sort=False)
    df['_return'] = g['_price'].pct_change(fill_method=None)
    names = []
    for lag in LAGS:
        name = f'return_{lag}bar'
        df[name] = df['_price'] / g['_price'].shift(lag) - 1
        names.append(name)
    for window in HIST_WINDOWS:
        vol = f'realized_vol_{window}'
        df[vol] = df.groupby(['symbol', 'interval_code'], sort=False)['_return'].transform(
            lambda s: s.rolling(window, min_periods=window).std(ddof=1))
        names.append(vol)
        ma = f'ma_distance_{window}'
        df[ma] = df['_price'] / g['_price'].transform(lambda s: s.rolling(window, min_periods=window).mean()) - 1
        names.append(ma)
        high = f'rolling_position_{window}'
        low_series = g['_price'].transform(lambda s: s.rolling(window, min_periods=window).min())
        high_series = g['_price'].transform(lambda s: s.rolling(window, min_periods=window).max())
        spread = high_series - low_series
        df[high] = ((df['_price'] - low_series) / spread).where(spread.gt(0), 0.5)
        df.loc[high_series.isna(), high] = np.nan
        names.append(high)
    # Normalize quadratic derivatives to make their scale comparable across tickers.
    for raw, new in [('slope', 'slope_relative'), ('concavity', 'concavity_relative')]:
        df[new] = pd.to_numeric(df[raw], errors='coerce') / df['_price']
        names.append(new)
    if 'volume' in df:
        df['_volume'] = pd.to_numeric(df['volume'], errors='coerce').where(lambda x: x >= 0)
        df['relative_volume_20'] = df['_volume'] / df.groupby(['symbol', 'interval_code'])['_volume'].transform(
            lambda s: s.shift(1).rolling(20, min_periods=20).mean())
        names.append('relative_volume_20')
    # Market proxy: mean contemporaneous return of *other* stocks at the same bar.
    # At least 10 peers required; leave-one-out avoids mechanical self-correlation.
    by_time = df.groupby(['interval_code', 'bar_timestamp'])['_return']
    sums = by_time.transform('sum')
    counts = by_time.transform('count')
    self_valid = df['_return'].notna().astype(int)
    peers = counts - self_valid
    df['peer_market_return'] = ((sums - df['_return'].fillna(0)) / peers).where(peers >= 10)
    df['relative_strength_1bar'] = df['_return'] - df['peer_market_return']
    names += ['peer_market_return', 'relative_strength_1bar']
    # Rolling peer return context must be built on each symbol's own chronology.
    df['peer_market_momentum_5'] = df.groupby(['symbol', 'interval_code'])['peer_market_return'].transform(
        lambda s: s.rolling(5, min_periods=5).sum())
    names.append('peer_market_momentum_5')
    df.drop(columns=['_price', '_return', '_volume'] if '_volume' in df else ['_price', '_return'], inplace=True)
    return df, names


def _probabilities(model, x):
    p = model.predict_proba(x)
    aligned = np.zeros((len(x), 3), dtype=float)
    for j, label in enumerate(model.classes_):
        aligned[:, LABELS.index(int(label))] = p[:, j]
    return aligned


def run_feature_comparison(frame, *, n_folds=5, max_train_rows=120000,
                           selected_k=4, cluster_fit_limit=15000,
                           output_dir=None, random_state=42):
    """Five expanding folds, original vs expanded features, simple classifiers.

    The existing fold_data refits targets, KMeans, scaling and OHE on each
    training fold. Extra feature scaling is also fit on train only. We require
    complete expanded features in *both* variants to make paired comparisons.
    """
    eligible, universe = eligible_universe(frame)
    expanded, extra = add_historical_market_features(eligible)
    folds = expanding_folds(expanded, n_folds=n_folds)
    output = Path(output_dir) if output_dir else None
    if output:
        output.mkdir(parents=True, exist_ok=True)
        universe.to_csv(output/'universe_eligibility.csv', index=False)
    metrics, curves = [], []
    models = {
        'prior_baseline': lambda: DummyClassifier(strategy='prior'),
        'logistic_l2': lambda: LogisticRegression(C=0.1, max_iter=300, solver='lbfgs'),
        'logistic_l1': lambda: LogisticRegression(C=0.1, l1_ratio=1, solver='saga', max_iter=200, tol=0.003, random_state=random_state),
        'gaussian_nb': lambda: GaussianNB(),
    }
    for fold in folds:
        start = time.perf_counter()
        train, test, xb_train, xb_test, _ = fold_data(expanded, fold, selected_k=selected_k,
                                                     cluster_fit_limit=cluster_fit_limit)
        # Keep only complete, finite enhanced features for a matched sample.
        train_extra = train[extra].replace([np.inf, -np.inf], np.nan)
        test_extra = test[extra].replace([np.inf, -np.inf], np.nan)
        good_train = train_extra.notna().all(axis=1).to_numpy()
        good_test = test_extra.notna().all(axis=1).to_numpy()
        if good_train.sum() < 100 or good_test.sum() < 20:
            raise ValueError(f'Fold {fold.number}: too few complete enhanced feature rows')
        y_train = train.direction.to_numpy(dtype=int)[good_train]
        y_test = test.direction.to_numpy(dtype=int)[good_test]
        base_train, base_test = xb_train[good_train], xb_test[good_test]
        scaler = StandardScaler().fit(train_extra.loc[good_train])
        full_train = np.column_stack([base_train, scaler.transform(train_extra.loc[good_train])])
        full_test = np.column_stack([base_test, scaler.transform(test_extra.loc[good_test])])
        prep_seconds = time.perf_counter()-start
        # Same training sample for both feature sets and all models.
        if max_train_rows and len(y_train) > max_train_rows:
            rng = np.random.default_rng(random_state+fold.number)
            fit_idx = np.sort(rng.choice(len(y_train), max_train_rows, replace=False))
        else:
            fit_idx = np.arange(len(y_train))
        print(f'Fold {fold.number}/{len(folds)}: train={len(y_train):,}, fit={len(fit_idx):,}, '
              f'test={len(y_test):,}, preparation={prep_seconds:.1f}s', flush=True)
        for scheme, x_train, x_test in [('original',base_train,base_test),('historical_market',full_train,full_test)]:
            for name, constructor in models.items():
                model = constructor()
                start = time.perf_counter()
                model.fit(x_train[fit_idx], y_train[fit_idx])
                fit_seconds = time.perf_counter()-start
                start = time.perf_counter()
                predicted = model.predict(x_test)
                proba = _probabilities(model, x_test)
                predict_seconds = time.perf_counter()-start
                try:
                    auc = roc_auc_score(y_test, proba, labels=LABELS, multi_class='ovr', average='macro')
                except ValueError:
                    auc = np.nan
                metrics.append(dict(fold=fold.number, feature_set=scheme, model=name,
                                    train_rows=len(y_train), fit_rows=len(fit_idx), test_rows=len(y_test),
                                    n_features=x_train.shape[1], accuracy=accuracy_score(y_test,predicted),
                                    balanced_accuracy=balanced_accuracy_score(y_test,predicted),
                                    roc_auc_ovr_macro=auc, log_loss=log_loss(y_test,proba,labels=LABELS),
                                    prep_seconds=prep_seconds, fit_seconds=fit_seconds,
                                    predict_seconds=predict_seconds, train_end=str(fold.train_end),
                                    test_end=str(fold.test_end)))
                for class_id in LABELS:
                    truth = (y_test==class_id).astype(int)
                    if len(np.unique(truth)) == 2:
                        fpr,tpr,_ = roc_curve(truth,proba[:,LABELS.index(class_id)])
                        curves.append(pd.DataFrame({'fold':fold.number,'feature_set':scheme,'model':name,
                                                    'class_id':class_id,'fpr':fpr,'tpr':tpr}))
                print(f'  {scheme:18} {name:15} acc={metrics[-1]["accuracy"]:.4f} '
                      f'auc={auc:.4f} fit={fit_seconds:.1f}s', flush=True)
                if output:
                    pd.DataFrame(metrics).to_csv(output/'fold_metrics.csv', index=False)
    scores = pd.DataFrame(metrics)
    roc = pd.concat(curves,ignore_index=True) if curves else pd.DataFrame()
    if output:
        scores.to_parquet(output/'fold_metrics.parquet',index=False)
        roc.to_parquet(output/'roc_curves.parquet',index=False)
        (output/'run_config.json').write_text(json.dumps({'n_folds':n_folds,'max_train_rows':max_train_rows,
            'feature_names':extra,'models':list(models),'feature_sets':['original','historical_market'],
            'market_proxy':'leave-one-out equal-weight return of other eligible stocks'},indent=2))
    return scores, roc, universe, extra


def plot_feature_comparison(scores, curves, output_dir=None):
    import matplotlib.pyplot as plt
    summary = scores.groupby(['feature_set','model'],as_index=False).agg(
        accuracy=('accuracy','mean'),balanced_accuracy=('balanced_accuracy','mean'),
        roc_auc=('roc_auc_ovr_macro','mean'),log_loss=('log_loss','mean'),
        fit_seconds=('fit_seconds','sum'),accuracy_std=('accuracy','std'))
    summary = summary.sort_values(['roc_auc','accuracy'],ascending=False)
    fig,ax=plt.subplots(figsize=(10,5))
    for feature_set, part in summary.groupby('feature_set'):
        ax.scatter(part.roc_auc,part.accuracy*100,label=feature_set,s=60)
        for _,r in part.iterrows():
            ax.annotate(r.model,(r.roc_auc,r.accuracy*100),fontsize=7,xytext=(4,3),textcoords='offset points')
    ax.set(xlabel='Mean macro one-vs-rest ROC-AUC',ylabel='Mean accuracy (%)',title='Paired five-fold classifier comparison')
    ax.legend(); fig.tight_layout()
    if output_dir: fig.savefig(Path(output_dir)/'accuracy_auc_comparison.png',dpi=160,bbox_inches='tight')
    plt.show()
    fig,ax=plt.subplots(figsize=(9,6))
    grid=np.linspace(0,1,201)
    for _,r in summary.head(6).iterrows():
        sub=curves.loc[(curves.feature_set==r.feature_set)&(curves.model==r.model)]
        folds=[]
        for _,fold_data_ in sub.groupby('fold'):
            class_curves=[np.interp(grid,c.fpr.to_numpy(),c.tpr.to_numpy()) for _,c in fold_data_.groupby('class_id')]
            if class_curves: folds.append(np.mean(class_curves,axis=0))
        if folds: ax.plot(grid,np.mean(folds,axis=0),label=f'{r.feature_set}/{r.model} AUC={r.roc_auc:.3f}')
    ax.plot([0,1],[0,1],'k--',alpha=.5,label='Random')
    ax.set(xlabel='False positive rate',ylabel='True positive rate',title='Fold-averaged macro one-vs-rest ROC curves')
    ax.legend(fontsize=8);fig.tight_layout()
    if output_dir:fig.savefig(Path(output_dir)/'roc_curves.png',dpi=160,bbox_inches='tight')
    plt.show()
    return summary
