"""Volatility-window ablation on five chronological development folds.

No final holdout evaluation. Target definition stays fixed (training-only tertiles).
All schemes use the same complete rows, preprocessing and sampled training indices.
"""
from __future__ import annotations
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, roc_curve, log_loss
from .classifier_benchmark import eligible_universe, LABELS
from .historical_market import add_historical_market_features
from .feature_ablation import development_holdout_split
from .walk_forward import expanding_folds, fold_data

VOL = ('realized_vol_5', 'realized_vol_10', 'realized_vol_20')


def volatility_schemes(extra):
    """Keep nonvolatility features constant while testing volatility windows."""
    extra = list(extra)
    nonvol = [c for c in extra if c not in VOL]
    return {
        'no_volatility': nonvol,
        'vol_5_only': nonvol + ['realized_vol_5'],
        'vol_10_only': nonvol + ['realized_vol_10'],
        'vol_20_only': nonvol + ['realized_vol_20'],
        'vol_5_10': nonvol + ['realized_vol_5', 'realized_vol_10'],
        'vol_5_20': nonvol + ['realized_vol_5', 'realized_vol_20'],
        'vol_10_20': nonvol + ['realized_vol_10', 'realized_vol_20'],
        'all_volatility': extra,
    }


def _model(name):
    if name == 'prior_baseline':
        return DummyClassifier(strategy='prior')
    if name == 'logistic_l2':
        return LogisticRegression(C=.1, max_iter=500, solver='lbfgs')
    if name == 'logistic_l1':
        return LogisticRegression(C=.1, l1_ratio=1, max_iter=500, solver='saga', tol=.002, random_state=42)
    raise ValueError(name)


def _proba(model, X):
    raw = model.predict_proba(X)
    out = np.zeros((len(X), len(LABELS)))
    for i, cls in enumerate(model.classes_):
        out[:, LABELS.index(int(cls))] = raw[:, i]
    return out


def _score(y, predicted, probability):
    try:
        auc = roc_auc_score(y, probability, labels=LABELS, multi_class='ovr', average='macro')
    except ValueError:
        auc = np.nan
    return dict(accuracy=accuracy_score(y, predicted),
                balanced_accuracy=balanced_accuracy_score(y, predicted),
                roc_auc_ovr_macro=auc,
                log_loss=log_loss(y, probability, labels=LABELS))


def run_volatility_diagnostics(frame, *, n_folds=5, holdout_fraction=.20,
                               max_train_rows=120000, selected_k=4,
                               cluster_fit_limit=15000, output_dir=None,
                               models=('prior_baseline','logistic_l2','logistic_l1')):
    eligible, eligibility = eligible_universe(frame)
    expanded, extra = add_historical_market_features(eligible)
    development, boundary = development_holdout_split(expanded, holdout_fraction)
    schemes = volatility_schemes(extra)
    out = Path(output_dir) if output_dir else None
    if out:
        out.mkdir(parents=True, exist_ok=True)
        eligibility.to_csv(out/'universe_eligibility.csv',index=False)
        (out/'holdout_lock.json').write_text(json.dumps({
            'status':'NOT_EVALUATED','development_end':str(boundary),
            'holdout_fraction':holdout_fraction,'n_folds':n_folds,
            'target':'next_return training-fold tertiles; unchanged',
            'selection_rule':'Choose using development CV only; keep final holdout untouched'
        },indent=2))
    rows=[]; curves=[]; regimes=[]; prevalence=[]
    for fold in expanding_folds(development,n_folds=n_folds):
        begin=time.perf_counter()
        train,test,base_train,base_test,_=fold_data(development,fold,selected_k=selected_k,
                                                    cluster_fit_limit=cluster_fit_limit)
        et=train[extra].replace([np.inf,-np.inf],np.nan)
        ev=test[extra].replace([np.inf,-np.inf],np.nan)
        keep_train=et.notna().all(axis=1).to_numpy()
        keep_test=ev.notna().all(axis=1).to_numpy()
        if keep_train.sum()<100 or keep_test.sum()<20:
            raise ValueError(f'Fold {fold.number}: insufficient complete rows')
        yt=train.direction.to_numpy(dtype=int)[keep_train]
        yv=test.direction.to_numpy(dtype=int)[keep_test]
        bt=base_train[keep_train]; bv=base_test[keep_test]
        et=et.loc[keep_train]; ev=ev.loc[keep_test]
        # Quantile boundaries for volatility regimes are TRAINING ONLY.
        edges=np.quantile(et['realized_vol_20'],[1/3,2/3])
        if edges[0] >= edges[1]:
            raise ValueError('Cannot distinguish volatility regimes: equal quantile cutoffs')
        regime=np.searchsorted(edges,ev['realized_vol_20'].to_numpy(),side='right')
        for cls in LABELS:
            prevalence.append(dict(fold=fold.number,class_id=cls,
                                   train_share=float(np.mean(yt==cls)),
                                   test_share=float(np.mean(yv==cls))))
        scaler=StandardScaler().fit(et)
        zt=pd.DataFrame(scaler.transform(et),columns=extra)
        zv=pd.DataFrame(scaler.transform(ev),columns=extra)
        if max_train_rows and len(yt)>max_train_rows:
            idx=np.sort(np.random.default_rng(42+fold.number).choice(len(yt),max_train_rows,replace=False))
        else:
            idx=np.arange(len(yt))
        prep=time.perf_counter()-begin
        print(f'Fold {fold.number}/{n_folds}: train={len(yt):,}, sampled={len(idx):,}, '
              f'test={len(yv):,}, preprocessing={prep:.1f}s',flush=True)
        for scheme, cols in schemes.items():
            xt=np.column_stack([bt,zt[cols].to_numpy()])
            xv=np.column_stack([bv,zv[cols].to_numpy()])
            for name in models:
                model=_model(name)
                start=time.perf_counter()
                model.fit(xt[idx],yt[idx])
                fit_seconds=time.perf_counter()-start
                start=time.perf_counter()
                pred=model.predict(xv)
                prob=_proba(model,xv)
                predict_seconds=time.perf_counter()-start
                rows.append(dict(fold=fold.number,scheme=scheme,model=name,
                                 train_rows=len(yt),fit_rows=len(idx),test_rows=len(yv),
                                 n_features=xt.shape[1],preprocess_seconds=prep,
                                 fit_seconds=fit_seconds,predict_seconds=predict_seconds,
                                 train_end=str(fold.train_end),test_end=str(fold.test_end),
                                 **_score(yv,pred,prob)))
                for cls in LABELS:
                    truth=(yv==cls).astype(int)
                    if np.unique(truth).size==2:
                        fpr,tpr,_=roc_curve(truth,prob[:,LABELS.index(cls)])
                        curves.append(pd.DataFrame(dict(fold=fold.number,scheme=scheme,
                            model=name,class_id=cls,fpr=fpr,tpr=tpr)))
                for r in range(3):
                    mask=regime==r
                    if mask.sum()<10:continue
                    vals=_score(yv[mask],pred[mask],prob[mask])
                    regimes.append(dict(fold=fold.number,scheme=scheme,model=name,
                                        volatility_regime=('low','medium','high')[r],
                                        observations=int(mask.sum()),**vals))
                print(f'  {scheme:17} {name:15} accuracy={rows[-1]["accuracy"]:.4f} '
                      f'auc={rows[-1]["roc_auc_ovr_macro"]:.4f} fit={fit_seconds:.1f}s',flush=True)
                if out:pd.DataFrame(rows).to_csv(out/'fold_metrics.csv',index=False)
    scores=pd.DataFrame(rows)
    roc=pd.concat(curves,ignore_index=True) if curves else pd.DataFrame()
    regime_scores=pd.DataFrame(regimes)
    shares=pd.DataFrame(prevalence)
    if out:
        scores.to_parquet(out/'fold_metrics.parquet',index=False)
        roc.to_parquet(out/'roc_curves.parquet',index=False)
        regime_scores.to_csv(out/'volatility_regime_scores.csv',index=False)
        shares.to_csv(out/'class_prevalence.csv',index=False)
        summary_volatility(scores).to_csv(out/'summary.csv',index=False)
    return scores,roc,regime_scores,shares,eligibility,boundary


def summary_volatility(scores):
    return scores.groupby(['scheme','model'],as_index=False).agg(
        accuracy=('accuracy','mean'),accuracy_std=('accuracy','std'),
        balanced_accuracy=('balanced_accuracy','mean'),
        auc=('roc_auc_ovr_macro','mean'),auc_std=('roc_auc_ovr_macro','std'),
        log_loss=('log_loss','mean'),fit_seconds=('fit_seconds','sum')
    ).sort_values(['auc','accuracy'],ascending=False)


def plot_volatility(scores,curves,regimes,output_dir=None):
    import matplotlib.pyplot as plt
    out=Path(output_dir) if output_dir else None
    summary=summary_volatility(scores)
    fig,ax=plt.subplots(figsize=(11,5))
    data=summary[summary.model.eq('logistic_l2')].sort_values('accuracy')
    ax.barh(data.scheme,data.accuracy*100)
    ax.set(xlabel='Five-fold mean accuracy (%)',title='Volatility-window ablation: Logistic L2')
    fig.tight_layout()
    if out:fig.savefig(out/'volatility_accuracy.png',dpi=160)
    plt.show()
    fig,ax=plt.subplots(figsize=(9,6))
    grid=np.linspace(0,1,201)
    for _,r in summary[summary.model.ne('prior_baseline')].head(5).iterrows():
        subset=curves[(curves.scheme==r.scheme)&(curves.model==r.model)]
        fold_lines=[]
        for _,fold_curves in subset.groupby('fold'):
            cls_lines=[np.interp(grid,c.fpr,c.tpr) for _,c in fold_curves.groupby('class_id')]
            if cls_lines:fold_lines.append(np.mean(cls_lines,axis=0))
        if fold_lines:ax.plot(grid,np.mean(fold_lines,axis=0),label=f'{r.scheme} / {r.model} ({r.auc:.3f})')
    ax.plot([0,1],[0,1],'k--',alpha=.5)
    ax.set(xlabel='False positive rate',ylabel='True positive rate',title='Development CV macro OVR ROC curves')
    ax.legend(fontsize=8);fig.tight_layout()
    if out:fig.savefig(out/'volatility_roc.png',dpi=160)
    plt.show()
    pivot=regimes[regimes.model.eq('logistic_l2')].groupby(['scheme','volatility_regime']).accuracy.mean().unstack()
    fig,ax=plt.subplots(figsize=(11,5))
    pivot.plot(kind='bar',ax=ax)
    ax.set(ylabel='Accuracy',title='Logistic L2: accuracy by training-defined volatility regime')
    fig.tight_layout()
    if out:fig.savefig(out/'regime_accuracy.png',dpi=160)
    plt.show()
    return summary
