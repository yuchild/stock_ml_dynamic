"""One-time chronological final holdout for the frozen next-bar classifier.

Configuration fixed before holdout: pooled 3-class tertile target, historical and
market features excluding realized_vol_10, LogisticRegression L2 C=0.1.
"""
from __future__ import annotations
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, log_loss, roc_curve, confusion_matrix
from .classifier_benchmark import eligible_universe
from .historical_market import add_historical_market_features
from .feature_ablation import development_holdout_split
from .walk_forward import Fold, fold_data
from .regularization_diagnostics import sample_rows

CLASSES = (0, 1, 2)
FROZEN_C = 0.1
EXCLUDED_EXTRA = ('realized_vol_10',)


def holdout_boundaries(frame, fraction=0.20):
    """Split by global feature timestamps, never random rows."""
    _, boundary = development_holdout_split(frame, fraction)
    last = pd.to_datetime(frame['next_bar_timestamp'], utc=True).max()
    if pd.isna(last) or last <= boundary:
        raise ValueError('No labeled outcomes beyond development boundary')
    return boundary, last


def fit_and_score_holdout(frame, *, holdout_fraction=0.2, max_train_rows=120000,
                          selected_k=4, cluster_fit_limit=15000, random_state=42,
                          output_dir=None, overwrite=False):
    """Score holdout once. Refuses to overwrite a prior completed holdout report.

    This does not tune hyperparameters on the holdout. Fold-specific target
    quantiles, candle clustering, scaling and encoding use development data.
    """
    output = Path(output_dir) if output_dir is not None else None
    if output is not None:
        report = output / 'final_holdout_metrics.csv'
        if report.exists() and not overwrite:
            raise FileExistsError(f'Final holdout already evaluated at {report}; refusing to re-score. '
                                  'Set overwrite=True only for a documented technical correction, never tuning.')
    eligible, eligibility = eligible_universe(frame)
    expanded, extra = add_historical_market_features(eligible)
    keep = [c for c in extra if c not in EXCLUDED_EXTRA]
    boundary, last = holdout_boundaries(expanded, holdout_fraction)
    fold = Fold(1, boundary, last)
    t0 = time.perf_counter()
    tr, te, xb, xv, bundle = fold_data(expanded, fold, selected_k=selected_k,
                                       cluster_fit_limit=cluster_fit_limit)
    a = tr[keep].replace([np.inf, -np.inf], np.nan)
    b = te[keep].replace([np.inf, -np.inf], np.nan)
    va = a.notna().all(axis=1).to_numpy()
    vb = b.notna().all(axis=1).to_numpy()
    if va.sum() < 100 or vb.sum() < 20:
        raise ValueError('Insufficient complete rows in final holdout')
    train_rows = tr.loc[va].copy()
    test_rows = te.loc[vb].copy()
    scaler = StandardScaler().fit(a.loc[va])
    x_train = np.column_stack([xb[va], scaler.transform(a.loc[va])])
    x_test = np.column_stack([xv[vb], scaler.transform(b.loc[vb])])
    y_train = train_rows['direction'].to_numpy(dtype=int)
    y_test = test_rows['direction'].to_numpy(dtype=int)
    if not set(CLASSES).issubset(set(np.unique(y_train))) or not set(CLASSES).issubset(set(np.unique(y_test))):
        raise ValueError('All three target classes are required for evaluation')
    x_fit, y_fit = sample_rows(x_train, y_train, max_train_rows, random_state)
    prep_seconds = time.perf_counter() - t0
    metrics = []
    curves = []
    predictions = test_rows[['symbol','bar_timestamp','next_bar_timestamp']].reset_index(drop=True).copy()
    predictions['actual_direction'] = y_test
    for name, model in [
        ('prior_baseline', DummyClassifier(strategy='prior')),
        ('logistic_l2_frozen', LogisticRegression(C=FROZEN_C, max_iter=600, solver='lbfgs')),
    ]:
        start = time.perf_counter()
        model.fit(x_fit, y_fit)
        fit_seconds = time.perf_counter() - start
        start = time.perf_counter()
        proba_raw = model.predict_proba(x_test)
        predict_seconds = time.perf_counter() - start
        prob = np.zeros((len(y_test), 3), dtype=float)
        for j, cls in enumerate(model.classes_):
            prob[:, CLASSES.index(int(cls))] = proba_raw[:, j]
        pred = np.array(CLASSES)[prob.argmax(axis=1)]
        metrics.append({
            'model': name, 'accuracy': float(accuracy_score(y_test, pred)),
            'balanced_accuracy': float(balanced_accuracy_score(y_test, pred)),
            'roc_auc_ovr_macro': float(roc_auc_score(y_test, prob, labels=list(CLASSES),
                                                    multi_class='ovr', average='macro')),
            'log_loss': float(log_loss(y_test, prob, labels=list(CLASSES))),
            'fit_seconds': fit_seconds, 'predict_seconds': predict_seconds,
            'preprocessing_seconds': prep_seconds, 'train_rows_available': len(y_train),
            'train_rows_fitted': len(y_fit), 'holdout_rows': len(y_test),
            'development_end': str(boundary), 'holdout_last_outcome': str(last),
            'C': FROZEN_C if name == 'logistic_l2_frozen' else np.nan,
        })
        predictions[f'{name}_prediction'] = pred
        for cls in CLASSES:
            predictions[f'{name}_prob_{cls}'] = prob[:, cls]
            truth = (y_test == cls).astype(int)
            fpr, tpr, _ = roc_curve(truth, prob[:, cls])
            curves.append(pd.DataFrame({'model': name, 'class_id': cls, 'fpr': fpr, 'tpr': tpr}))
    scores = pd.DataFrame(metrics)
    roc = pd.concat(curves, ignore_index=True)
    confusion = pd.DataFrame(confusion_matrix(y_test, predictions['logistic_l2_frozen_prediction'],
                                              labels=list(CLASSES)),
                             index=[f'actual_{i}' for i in CLASSES],
                             columns=[f'predicted_{i}' for i in CLASSES])
    if output is not None:
        output.mkdir(parents=True, exist_ok=True)
        eligibility.to_csv(output / 'universe_eligibility.csv', index=False)
        predictions.to_parquet(output / 'final_holdout_predictions.parquet', index=False)
        roc.to_parquet(output / 'final_holdout_roc.parquet', index=False)
        confusion.to_csv(output / 'final_holdout_confusion.csv')
        (output / 'final_holdout_manifest.json').write_text(json.dumps({
            'status': 'EVALUATED', 'development_end': str(boundary),
            'holdout_last_outcome': str(last), 'holdout_fraction': holdout_fraction,
            'target': 'training-only next-return tertiles',
            'model': 'LogisticRegression L2', 'C': FROZEN_C,
            'excluded_extra_features': list(EXCLUDED_EXTRA),
            'train_rows_fitted': len(y_fit), 'holdout_rows': len(y_test),
            'note': 'Final holdout is for reporting, not further feature or hyperparameter selection.'
        }, indent=2))
        scores.to_csv(output / 'final_holdout_metrics.csv', index=False)
    return scores, predictions, roc, confusion, eligibility


def plot_holdout(scores, roc, confusion, output_dir=None):
    import matplotlib.pyplot as plt
    output = Path(output_dir) if output_dir is not None else None
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric in zip(axes, ('accuracy', 'roc_auc_ovr_macro')):
        ax.bar(scores['model'], scores[metric])
        ax.set(title=metric.replace('_', ' ').title(), ylim=(0, 1))
        ax.tick_params(axis='x', rotation=15)
    fig.tight_layout()
    if output: fig.savefig(output / 'holdout_scores.png', dpi=150)
    plt.show()
    fig, ax = plt.subplots(figsize=(7, 6))
    for (model, cls), group in roc.groupby(['model', 'class_id']):
        if model == 'logistic_l2_frozen':
            ax.plot(group['fpr'], group['tpr'], label=f'Class {cls}')
    ax.plot([0, 1], [0, 1], '--', color='gray', label='Random')
    ax.set(xlabel='False positive rate', ylabel='True positive rate', title='Final holdout · One-vs-rest ROC')
    ax.legend()
    fig.tight_layout()
    if output: fig.savefig(output / 'holdout_roc.png', dpi=150)
    plt.show()
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(confusion.to_numpy(), cmap='Blues')
    ax.set(xticks=range(3), yticks=range(3), xlabel='Predicted class', ylabel='Actual class', title='Final holdout confusion matrix')
    for (i, j), val in np.ndenumerate(confusion.to_numpy()): ax.text(j, i, str(val), ha='center', va='center')
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    if output: fig.savefig(output / 'holdout_confusion.png', dpi=150)
    plt.show()
