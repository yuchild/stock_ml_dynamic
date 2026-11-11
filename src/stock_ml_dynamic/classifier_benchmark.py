"""Five-fold, training-only feature selection and multiclass classifier benchmark."""
from __future__ import annotations
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss, roc_auc_score, roc_curve
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from .walk_forward import expanding_folds, fold_data

LABELS = [0, 1, 2]
LABEL_NAMES = {0: 'middle', 1: 'up', 2: 'down'}


def eligible_universe(frame: pd.DataFrame, min_bars: int = 31):
    """Exclude entire ticker/interval histories without 30 prior bars and valid latest z.

    This eligibility check is structural and causal: it uses history availability,
    not outcomes, and applies before training or cross-validation.
    """
    df = frame.copy()
    df['bar_timestamp'] = pd.to_datetime(df['bar_timestamp'], utc=True)
    groups = df.sort_values('bar_timestamp').groupby(['symbol', 'interval_code'], sort=False)
    summary = groups.agg(bars=('bar_timestamp', 'size'), latest_adj_z=('adj_z', 'last'))
    # agg(last) skips NA; explicitly check the final row instead.
    last = groups.tail(1).set_index(['symbol', 'interval_code'])['adj_z']
    summary['latest_adj_z'] = last.reindex(summary.index)
    summary['eligible'] = (summary.bars >= min_bars) & np.isfinite(pd.to_numeric(summary.latest_adj_z, errors='coerce'))
    allowed = summary.index[summary.eligible]
    mask = pd.MultiIndex.from_frame(df[['symbol', 'interval_code']]).isin(allowed)
    return df.loc[mask].copy(), summary.reset_index()


def _selector(kind, X_train, y_train, X_test):
    """Fit every selection/reduction step on training fold only."""
    if kind == 'all':
        return X_train, X_test, X_train.shape[1], None
    if kind == 'pca_95':
        reduction = PCA(n_components=0.95, svd_solver='full').fit(X_train)
    elif kind == 'lasso_l1':
        probe = LogisticRegression(penalty='l1', solver='saga', C=0.1, max_iter=180, tol=2e-3, random_state=42)
        reduction = SelectFromModel(probe, threshold='median').fit(X_train, y_train)
    elif kind == 'ridge_l2':
        # Ridge is shrinkage, not sparse selection. Rank absolute standardized coefficients.
        probe = RidgeClassifier(alpha=10.0).fit(X_train, y_train)
        importance = np.max(np.abs(probe.coef_), axis=0)
        k = max(1, int(np.ceil(X_train.shape[1] * 0.5)))
        selected = np.argsort(importance)[-k:]
        return X_train[:, selected], X_test[:, selected], k, selected
    elif kind == 'anova_50':
        reduction = SelectKBest(f_classif, k=min(30, X_train.shape[1])).fit(X_train, y_train)
    else:
        raise ValueError(f'Unknown feature scheme: {kind}')
    xt = reduction.transform(X_train)
    return xt, reduction.transform(X_test), xt.shape[1], reduction


def _models(random_state=42):
    return {
        'prior_baseline': DummyClassifier(strategy='prior'),
        'logistic_l2': LogisticRegression(C=1.0, max_iter=250, solver='lbfgs'),
        'logistic_l1': LogisticRegression(C=0.1, penalty='l1', solver='saga', max_iter=180, tol=2e-3, random_state=random_state),
        'random_forest': RandomForestClassifier(n_estimators=100, max_depth=12, min_samples_leaf=20, n_jobs=-1, random_state=random_state),
        'gaussian_nb': GaussianNB(),
        'hist_gradient_boosting': HistGradientBoostingClassifier(max_iter=120, max_leaf_nodes=15, learning_rate=0.05, random_state=random_state),
        'mlp_neural_network': MLPClassifier(hidden_layer_sizes=(64, 32), early_stopping=True, validation_fraction=0.12, n_iter_no_change=8, max_iter=65, batch_size=512, random_state=random_state),
    }


def _probabilities(model, X):
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(X)
    else:
        raise ValueError('Classifier does not support probabilities')
    aligned = np.zeros((len(X), len(LABELS)), dtype=float)
    for j, c in enumerate(model.classes_):
        aligned[:, LABELS.index(int(c))] = probs[:, j]
    return aligned


def run_benchmark(frame, *, n_folds=5, initial_fraction=0.55, schemes=('all','pca_95','lasso_l1','ridge_l2'),
                  models=None, selected_k=4, cluster_fit_limit=15000, max_train_rows=120000, random_state=42,
                  output_dir=None):
    """Chronological CV; sampling only training rows; no future outcome in preprocessing.

    Note: The MLP's early-stopping validation is random *within training data*;
    the outer evaluation remains forward-only. Model comparison is exploratory.
    """
    eligible, universe = eligible_universe(frame)
    folds = expanding_folds(eligible, n_folds=n_folds, initial_fraction=initial_fraction)
    chosen_models = models or tuple(_models())
    metrics, curves, selections = [], [], []
    out = Path(output_dir) if output_dir else None
    if out:
        out.mkdir(parents=True, exist_ok=True)
        universe.to_csv(out / 'universe_eligibility.csv', index=False)
    for fold in folds:
        t0 = time.perf_counter()
        train, test, X_train, X_test, bundle = fold_data(eligible, fold, selected_k=selected_k, cluster_fit_limit=cluster_fit_limit)
        prep_sec = time.perf_counter()-t0
        y_train, y_test = train.direction.to_numpy(dtype=int), test.direction.to_numpy(dtype=int)
        # Bound expensive training; sample only within training fold, never validation.
        if max_train_rows and len(y_train) > max_train_rows:
            rng = np.random.default_rng(random_state+fold.number)
            ids = np.sort(rng.choice(len(y_train), max_train_rows, replace=False))
            X_fit, y_fit = X_train[ids], y_train[ids]
        else:
            X_fit, y_fit = X_train, y_train
        print(f'Fold {fold.number}/{len(folds)}: train={len(y_train):,} (fit={len(y_fit):,}) test={len(y_test):,} prep={prep_sec:.1f}s', flush=True)
        for scheme in schemes:
            st = time.perf_counter()
            Xf, Xt, nfeatures, selector = _selector(scheme, X_fit, y_fit, X_test)
            select_sec = time.perf_counter()-st
            selections.append({'fold':fold.number, 'scheme':scheme, 'features':nfeatures, 'selection_seconds':select_sec})
            for name in chosen_models:
                if name not in _models():
                    raise ValueError(f'Unknown model {name}')
                model = _models(random_state)[name]
                start = time.perf_counter()
                try:
                    model.fit(Xf, y_fit)
                    fit_sec = time.perf_counter()-start
                    start = time.perf_counter()
                    predictions = model.predict(Xt)
                    probs = _probabilities(model, Xt)
                    predict_sec = time.perf_counter()-start
                    try:
                        auc = roc_auc_score(y_test, probs, labels=LABELS, multi_class='ovr', average='macro')
                    except ValueError:
                        auc = np.nan
                    row = {'fold':fold.number, 'scheme':scheme, 'model':name,
                           'train_rows':len(y_train), 'fit_rows':len(y_fit), 'test_rows':len(y_test),
                           'features':nfeatures, 'accuracy':accuracy_score(y_test, predictions),
                           'balanced_accuracy':balanced_accuracy_score(y_test, predictions),
                           'roc_auc_ovr_macro':auc, 'log_loss':log_loss(y_test, probs, labels=LABELS),
                           'preprocessing_seconds':prep_sec, 'selection_seconds':select_sec,
                           'fit_seconds':fit_sec, 'predict_seconds':predict_sec,
                           'total_seconds':prep_sec+select_sec+fit_sec+predict_sec,
                           'train_end':str(fold.train_end), 'test_end':str(fold.test_end), 'status':'ok'}
                    for j, label in enumerate(LABELS):
                        truth = (y_test == label).astype(int)
                        if len(np.unique(truth)) == 2:
                            fpr, tpr, _ = roc_curve(truth, probs[:, j])
                            curves.append(pd.DataFrame({'fold':fold.number,'scheme':scheme,'model':name,
                                                        'class_id':label,'fpr':fpr,'tpr':tpr}))
                except Exception as exc:
                    row = {'fold':fold.number,'scheme':scheme,'model':name,'status':f'ERROR: {type(exc).__name__}: {exc}',
                           'fit_seconds':time.perf_counter()-start}
                metrics.append(row)
                print(f'  {scheme:10} {name:23} acc={row.get("accuracy",float("nan")):.4f} '
                      f'auc={row.get("roc_auc_ovr_macro",float("nan")):.4f} fit={row["fit_seconds"]:.1f}s', flush=True)
                if out:
                    pd.DataFrame(metrics).to_csv(out/'fold_metrics.csv',index=False)
        if out:
            pd.DataFrame(selections).to_csv(out/'feature_selection.csv',index=False)
    scores = pd.DataFrame(metrics)
    roc = pd.concat(curves, ignore_index=True) if curves else pd.DataFrame(columns=['fold','scheme','model','class_id','fpr','tpr'])
    if out:
        roc.to_parquet(out/'roc_curves.parquet',index=False)
        scores.to_parquet(out/'fold_metrics.parquet',index=False)
        (out/'run_config.json').write_text(json.dumps({'n_folds':n_folds,'schemes':list(schemes),'models':list(chosen_models),
            'max_train_rows':max_train_rows,'initial_fraction':initial_fraction,'selected_k':selected_k},indent=2))
    return scores, roc, universe


def plot_results(scores, curves, output_dir=None, top_n=6):
    import matplotlib.pyplot as plt
    ok = scores.loc[scores.status.eq('ok')].copy()
    summary = ok.groupby(['scheme','model'],as_index=False).agg(accuracy=('accuracy','mean'),
        balanced_accuracy=('balanced_accuracy','mean'),roc_auc=('roc_auc_ovr_macro','mean'),
        fit_seconds=('fit_seconds','sum'))
    summary = summary.sort_values(['roc_auc','accuracy'],ascending=False)
    if summary.empty:
        raise ValueError('No successful benchmark results')
    fig, ax = plt.subplots(figsize=(11, max(4, len(summary)*0.28)))
    labels = summary['scheme']+' / '+summary['model']
    ax.barh(labels[::-1],summary.accuracy.iloc[::-1])
    ax.set(xlabel='Mean accuracy (five chronological folds)',title='Classifier and feature scheme accuracy')
    ax.axvline(1/3,linestyle='--',color='gray',label='33.3% reference (not fold-specific baseline)')
    ax.legend(loc='lower right')
    fig.tight_layout()
    if output_dir: fig.savefig(Path(output_dir)/'accuracy_comparison.png',dpi=160,bbox_inches='tight')
    plt.show()
    fig, ax = plt.subplots(figsize=(10,6))
    for _, r in summary.head(top_n).iterrows():
        part = curves.loc[(curves.scheme==r.scheme)&(curves.model==r.model)]
        if part.empty: continue
        # Average each class ROC onto a shared FPR grid, then average folds equally.
        grid = np.linspace(0,1,201)
        fold_lines = []
        for fold, g in part.groupby('fold'):
            cls_lines = [np.interp(grid, c.fpr.to_numpy(),c.tpr.to_numpy()) for _,c in g.groupby('class_id')]
            if cls_lines: fold_lines.append(np.mean(cls_lines,axis=0))
        if fold_lines:
            ax.plot(grid,np.mean(fold_lines,axis=0),label=f'{r.scheme}/{r.model} (AUC={r.roc_auc:.3f})')
    ax.plot([0,1],[0,1],'k--',alpha=0.4,label='Random')
    ax.set(xlabel='False positive rate',ylabel='True positive rate',title='Macro one-vs-rest ROC curves (fold-averaged)')
    ax.legend(fontsize=8)
    fig.tight_layout()
    if output_dir: fig.savefig(Path(output_dir)/'roc_curves.png',dpi=160,bbox_inches='tight')
    plt.show()
    return summary
