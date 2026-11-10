"""Reproducible chronological training and read-only experiment comparisons."""
from datetime import datetime, timezone
from pathlib import Path
import json
import uuid
import numpy as np
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, brier_score_loss,
                             mean_absolute_error, mean_squared_error, roc_auc_score)
from .experiment import ExperimentSpec
from .experiments import build_experiment_frame


def train_experiment(prices, spec: ExperimentSpec, symbol: str, interval: str):
    frame = build_experiment_frame(prices, spec)
    split = int(len(frame) * (1 - spec.test_fraction))
    # Training target at i uses close at i+horizon; purge all labels touching test.
    train_end = split - spec.horizon
    if train_end <= 0:
        raise ValueError('Not enough bars for purged split')
    valid = frame[list(spec.features)].notna().all(axis=1) & frame.target.notna()
    train = frame.iloc[:train_end].loc[valid.iloc[:train_end]]
    test = frame.iloc[split:].loc[valid.iloc[split:]]
    if len(train) + len(test) < spec.min_labeled_rows or len(train) < 50 or len(test) < 20:
        raise ValueError(f'Insufficient labeled rows: train={len(train)}, test={len(test)}')
    x_train, x_test = train[list(spec.features)], test[list(spec.features)]
    y_train, y_test = train.target, test.target
    if spec.target_type == 'return':
        models = {'mean_baseline': DummyRegressor(strategy='mean'),
                  'hist_gradient_boosting': HistGradientBoostingRegressor(max_iter=150, max_leaf_nodes=15, learning_rate=.05, l2_regularization=1, random_state=spec.random_state)}
    else:
        if y_train.nunique() < 2:
            raise ValueError('Training set contains only one target class; adjust threshold or window')
        models = {'prior_baseline': DummyClassifier(strategy='prior'),
                  'hist_gradient_boosting': HistGradientBoostingClassifier(max_iter=150, max_leaf_nodes=15, learning_rate=.05, l2_regularization=1, random_state=spec.random_state)}
    metrics = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        if spec.target_type == 'return':
            metrics[name] = {'mae': float(mean_absolute_error(y_test, pred)),
                             'rmse': float(np.sqrt(mean_squared_error(y_test, pred)))}
        else:
            result = {'accuracy': float(accuracy_score(y_test, pred)),
                      'balanced_accuracy': float(balanced_accuracy_score(y_test, pred))}
            proba = model.predict_proba(x_test)
            positive = proba[:, list(model.classes_).index(1)] if 1 in model.classes_ else np.zeros(len(y_test))
            result['brier_score'] = float(brier_score_loss(y_test.astype(int), positive))
            result['roc_auc'] = float(roc_auc_score(y_test, positive)) if y_test.nunique() == 2 else None
            metrics[name] = result
    return models, {
        'run_id': datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '_' + uuid.uuid4().hex[:8],
        'created_utc': datetime.now(timezone.utc).isoformat(), 'symbol': symbol, 'interval': interval,
        'spec': spec.as_dict(), 'spec_hash': spec.fingerprint(), 'metrics': metrics,
        'train_rows': len(train), 'test_rows': len(test), 'raw_bars': len(frame),
        'purged_bars': spec.horizon, 'train_end': str(train.time.iloc[-1]),
        'test_start': str(test.time.iloc[0]), 'test_end': str(test.time.iloc[-1]),
        'warning': 'Single holdout metrics are exploratory. Selecting on this test set biases comparisons; confirm with untouched out-of-sample data.'}


def save_experiment(models, meta, root='artifacts/experiments'):
    import joblib
    folder = Path(root) / meta['run_id']
    folder.mkdir(parents=True, exist_ok=False)
    for name, model in models.items():
        joblib.dump(model, folder / f'{name}.joblib')
    (folder / 'metadata.json').write_text(json.dumps(meta, indent=2) + '\n', encoding='utf-8')
    return folder


def list_experiments(root='artifacts/experiments'):
    records = []
    for file in sorted(Path(root).glob('*/metadata.json')):
        try:
            record = json.loads(file.read_text(encoding='utf-8'))
            if isinstance(record, dict) and 'run_id' in record and 'metrics' in record:
                records.append(record)
        except (OSError, ValueError):
            continue
    return records


def comparison_rows(records, symbol=None, interval=None, spec_hash=None):
    rows = []
    for r in records:
        if symbol and r.get('symbol') != symbol: continue
        if interval and r.get('interval') != interval: continue
        if spec_hash and r.get('spec_hash') != spec_hash: continue
        for name, metrics in r['metrics'].items():
            rows.append({'run_id': r['run_id'], 'symbol': r['symbol'], 'interval': r['interval'],
                         'target': r['spec']['target_type'], 'horizon': r['spec']['horizon'],
                         'spec_hash': r['spec_hash'], 'model': name, **metrics})
    return rows
