"""Leakage-conscious pooled classification and return-regression experiments.

Inputs are the persisted master table and its *training-fitted* transformer bundle.
The historical split is defined by the master table's global cutoff.
"""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, log_loss,
                             mean_absolute_error, mean_squared_error, r2_score, roc_auc_score)
from .master_table import FEATURE_COLUMNS


def load_master(directory='artifacts/master_tables/1d'):
    root = Path(directory)
    master = pd.read_parquet(root/'master_labeled.parquet')
    latest = pd.read_parquet(root/'latest_inference.parquet')
    bundle = joblib.load(root/'transformers.joblib')
    manifest = json.loads((root/'manifest.json').read_text())
    if bundle['features'] != FEATURE_COLUMNS or manifest['feature_columns'] != FEATURE_COLUMNS:
        raise ValueError('Saved feature order differs from current code; rebuild master table')
    for frame in (master, latest):
        frame['bar_timestamp'] = pd.to_datetime(frame['bar_timestamp'], utc=True)
    master['next_bar_timestamp'] = pd.to_datetime(master['next_bar_timestamp'], utc=True)
    return master, latest, bundle, manifest


def prepare_pooled_data(master, latest, bundle):
    """Transform with saved scaler/encoder; retain original row indices for alignment."""
    def transform(frame):
        valid = frame[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
        rows = frame.loc[valid].copy()
        x = bundle['preprocessing'].transform(rows[FEATURE_COLUMNS])
        return pd.DataFrame(x, index=rows.index, columns=bundle['preprocessing'].get_feature_names_out()), rows
    return (*transform(master), *transform(latest))


def chronological_masks(rows, cutoff, horizon=1):
    """Split by target observation time; purge h-1 bars before validation.

    For next-bar targets the master table's next_bar_timestamp suffices. For
    multi-step outcomes, callers must use their horizon-specific target timestamp.
    """
    if horizon != 1:
        raise ValueError('Use horizon-specific future target timestamps for multi-step splits')
    cutoff = pd.Timestamp(cutoff)
    if cutoff.tzinfo is None:
        cutoff = cutoff.tz_localize('UTC')
    train = rows['next_bar_timestamp'] <= cutoff
    test = rows['bar_timestamp'] > cutoff
    if train.sum() == 0 or test.sum() == 0:
        raise ValueError('Need both chronological train and test rows')
    return train, test


def classification_experiment(x, rows, cutoff):
    train, test = chronological_masks(rows, cutoff)
    y = rows['direction'].astype(int)
    models = {
        'majority_baseline': DummyClassifier(strategy='prior'),
        'hist_gradient_boosting': HistGradientBoostingClassifier(max_iter=150, max_leaf_nodes=15,
                                                              learning_rate=0.05, l2_regularization=1, random_state=42),
    }
    reports = []
    for name, model in models.items():
        model.fit(x.loc[train], y.loc[train])
        pred = model.predict(x.loc[test]); prob = model.predict_proba(x.loc[test])
        classes = list(model.classes_)
        metrics = {'model': name, 'train_rows': int(train.sum()), 'test_rows': int(test.sum()),
                   'accuracy': float(accuracy_score(y.loc[test], pred)),
                   'balanced_accuracy': float(balanced_accuracy_score(y.loc[test], pred))}
        if len(classes) >= 2 and set(y.loc[test].unique()).issubset(classes):
            try:
                metrics['log_loss'] = float(log_loss(y.loc[test], prob, labels=classes))
                if len(classes) == 2:
                    metrics['roc_auc'] = float(roc_auc_score(y.loc[test], prob[:, 1])) if y.loc[test].nunique()==2 else np.nan
                elif set(y.loc[test].unique()) == set(classes):
                    metrics['roc_auc'] = float(roc_auc_score(y.loc[test], prob, labels=classes, multi_class='ovr', average='macro'))
                else:
                    metrics['roc_auc'] = np.nan
            except ValueError:
                metrics['roc_auc'] = np.nan
        else:
            metrics['log_loss'] = np.nan; metrics['roc_auc'] = np.nan
        reports.append(metrics)
    return models, pd.DataFrame(reports), (train, test)


def regression_experiment(x, rows, cutoff):
    """Predict next-bar percent return, then reconstruct next adjusted close."""
    train, test = chronological_masks(rows, cutoff)
    valid = rows['next_return'].notna() & np.isfinite(rows['next_return']) & (rows['adj_close'] > 0)
    train = train & valid; test = test & valid
    if train.sum() == 0 or test.sum() == 0:
        raise ValueError('No valid regression rows')
    y = rows['next_return'].astype(float)
    models = {'mean_return_baseline': DummyRegressor(strategy='mean'),
              'hist_gradient_boosting': HistGradientBoostingRegressor(max_iter=150, max_leaf_nodes=15,
                                                                    learning_rate=0.05, l2_regularization=1, random_state=42)}
    reports = []
    for name, model in models.items():
        model.fit(x.loc[train], y.loc[train]); pred = model.predict(x.loc[test])
        actual_close = rows.loc[test, 'next_adj_close'].to_numpy(dtype=float)
        predicted_close = rows.loc[test, 'adj_close'].to_numpy(dtype=float) * (1 + pred)
        reports.append({'model':name, 'train_rows':int(train.sum()), 'test_rows':int(test.sum()),
                        'return_mae':float(mean_absolute_error(y.loc[test], pred)),
                        'return_rmse':float(np.sqrt(mean_squared_error(y.loc[test], pred))),
                        'return_r2':float(r2_score(y.loc[test], pred)),
                        'price_mae':float(mean_absolute_error(actual_close, predicted_close))})
    return models, pd.DataFrame(reports), (train, test)


def predict_latest(models, x_latest, latest_rows, task):
    """Prediction is for the next observed bar; class probabilities follow classes_."""
    result = latest_rows[['symbol', 'interval_code', 'bar_timestamp', 'adj_close']].copy()
    for name, model in models.items():
        if task == 'classification':
            result[f'{name}_direction'] = model.predict(x_latest)
            probs = model.predict_proba(x_latest)
            for i, class_id in enumerate(model.classes_):
                result[f'{name}_prob_{int(class_id)}'] = probs[:, i]
        elif task == 'regression':
            returns = model.predict(x_latest)
            result[f'{name}_next_return'] = returns
            result[f'{name}_next_adj_close'] = result['adj_close'] * (1 + returns)
        else:
            raise ValueError('Unknown task')
    return result


def save_pooled_run(models, metrics, latest_predictions, *, task, master_directory, destination='artifacts/pooled_runs', run_name=None):
    from datetime import datetime, timezone
    run_name = run_name or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    if not run_name.replace('_','').replace('-','').isalnum():
        raise ValueError('Invalid run name')
    root = Path(destination)/run_name
    root.mkdir(parents=True, exist_ok=False)
    for name, model in models.items():
        joblib.dump(model, root/f'{task}_{name}.joblib')
    metrics.to_json(root/f'{task}_metrics.json', orient='records', indent=2)
    latest_predictions.to_parquet(root/f'{task}_latest_predictions.parquet', index=False)
    metadata = {'task':task, 'master_directory':str(Path(master_directory).resolve()),
                'feature_columns':FEATURE_COLUMNS, 'transformer_file':'transformers.joblib',
                'note':'Models consume saved training-fitted preprocessing. Single chronological holdout is exploratory; do not select and deploy solely on this score.'}
    (root/f'{task}_metadata.json').write_text(json.dumps(metadata, indent=2))
    return root
