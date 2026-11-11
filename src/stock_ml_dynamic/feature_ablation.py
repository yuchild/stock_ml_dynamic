"""Five-fold development-only feature-group ablation for next-bar classification.

The newest 20% of timestamps are reserved as an untouched final holdout.
Run the holdout only once after selecting a configuration from development CV.
"""
from __future__ import annotations
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, log_loss, roc_curve
from sklearn.preprocessing import StandardScaler
from .classifier_benchmark import eligible_universe, LABELS
from .historical_market import add_historical_market_features
from .walk_forward import expanding_folds, fold_data, Fold

GROUPS = {
    'momentum': ['return_1bar','return_2bar','return_3bar','return_5bar','return_10bar','return_20bar'],
    'volatility': ['realized_vol_5','realized_vol_10','realized_vol_20'],
    'trend_position': ['ma_distance_5','ma_distance_10','ma_distance_20',
                       'rolling_position_5','rolling_position_10','rolling_position_20',
                       'slope_relative','concavity_relative'],
    'volume': ['relative_volume_20'],
    'market_relative': ['peer_market_return','relative_strength_1bar','peer_market_momentum_5'],
}

def feature_schemes(extra):
    available = {k: [c for c in v if c in extra] for k,v in GROUPS.items()}
    return {'original': [], 'all_expanded': list(extra),
            **{f'without_{k}': [c for c in extra if c not in cols]
               for k,cols in available.items() if cols}}

def development_holdout_split(frame, holdout_fraction=0.20):
    if not 0.05 <= holdout_fraction <= 0.40:
        raise ValueError('holdout_fraction must be between 0.05 and 0.40')
    times = pd.DatetimeIndex(pd.to_datetime(frame.bar_timestamp, utc=True).unique()).sort_values()
    if len(times) < 100: raise ValueError('Not enough timestamps for holdout')
    boundary = times[int(len(times)*(1-holdout_fraction))-1]
    # Keep holdout observations in memory for eventual locked final evaluation;
    # CV fold_data uses outcomes bounded by the development boundary.
    dev = frame.loc[pd.to_datetime(frame.bar_timestamp,utc=True) <= boundary].copy()
    return dev, boundary

def _model(name):
    if name=='prior_baseline': return DummyClassifier(strategy='prior')
    if name=='logistic_l2': return LogisticRegression(C=0.1,max_iter=300,solver='lbfgs')
    if name=='logistic_l1': return LogisticRegression(C=0.1,l1_ratio=1,max_iter=200,
                                                      solver='saga',tol=.003,random_state=42)
    raise ValueError(name)

def _prob(model,x):
    result=np.zeros((len(x),3))
    for j,c in enumerate(model.classes_): result[:,LABELS.index(int(c))]=model.predict_proba(x)[:,j]
    return result

def _metrics(y,p,prob):
    try: auc=roc_auc_score(y,prob,labels=LABELS,multi_class='ovr',average='macro')
    except ValueError: auc=np.nan
    return dict(accuracy=accuracy_score(y,p),balanced_accuracy=balanced_accuracy_score(y,p),
                roc_auc_ovr_macro=auc,log_loss=log_loss(y,prob,labels=LABELS))

def run_ablation(frame, *, n_folds=5, holdout_fraction=.20, max_train_rows=120000,
                 selected_k=4, cluster_fit_limit=15000, output_dir=None):
    eligible, eligibility = eligible_universe(frame)
    expanded, extra = add_historical_market_features(eligible)
    development, boundary = development_holdout_split(expanded,holdout_fraction)
    schemes = feature_schemes(extra)
    # Strict paired sample across schemes; never impute invalid features using test data.
    results=[]; curves=[]
    out=Path(output_dir) if output_dir else None
    if out:
        out.mkdir(parents=True,exist_ok=True)
        eligibility.to_csv(out/'universe_eligibility.csv',index=False)
    for fold in expanding_folds(development,n_folds=n_folds):
        t0=time.perf_counter()
        train,test,xb_train,xb_test,_=fold_data(development,fold,selected_k=selected_k,
                                               cluster_fit_limit=cluster_fit_limit)
        et=train[extra].replace([np.inf,-np.inf],np.nan)
        ev=test[extra].replace([np.inf,-np.inf],np.nan)
        good_train=et.notna().all(axis=1).to_numpy()
        good_test=ev.notna().all(axis=1).to_numpy()
        if good_train.sum()<100 or good_test.sum()<20: raise ValueError('Insufficient paired samples')
        ytrain=train.direction.to_numpy(dtype=int)[good_train]
        ytest=test.direction.to_numpy(dtype=int)[good_test]
        xbtrain=xb_train[good_train]; xbtest=xb_test[good_test]
        et=et.loc[good_train]; ev=ev.loc[good_test]
        # Fit all added-feature scalers on training rows only.
        scaler=StandardScaler().fit(et)
        ztrain=pd.DataFrame(scaler.transform(et),columns=extra)
        ztest=pd.DataFrame(scaler.transform(ev),columns=extra)
        if max_train_rows and len(ytrain)>max_train_rows:
            rng=np.random.default_rng(42+fold.number)
            idx=np.sort(rng.choice(len(ytrain),max_train_rows,replace=False))
        else: idx=np.arange(len(ytrain))
        prep=time.perf_counter()-t0
        print(f'Fold {fold.number}/{n_folds}: train={len(ytrain):,}, fit={len(idx):,}, test={len(ytest):,}; prep={prep:.1f}s',flush=True)
        for scheme,cols in schemes.items():
            xt=np.column_stack([xbtrain,ztrain[cols].to_numpy()]) if cols else xbtrain
            xv=np.column_stack([xbtest,ztest[cols].to_numpy()]) if cols else xbtest
            for name in ('prior_baseline','logistic_l2','logistic_l1'):
                model=_model(name)
                start=time.perf_counter();model.fit(xt[idx],ytrain[idx]);fit=time.perf_counter()-start
                start=time.perf_counter();p=model.predict(xv);proba=_prob(model,xv);pred=time.perf_counter()-start
                results.append(dict(fold=fold.number,scheme=scheme,model=name,
                                    train_rows=len(ytrain),fit_rows=len(idx),test_rows=len(ytest),
                                    n_features=xt.shape[1],prep_seconds=prep,fit_seconds=fit,
                                    predict_seconds=pred,train_end=str(fold.train_end),test_end=str(fold.test_end),
                                    **_metrics(ytest,p,proba)))
                for cls in LABELS:
                    truth=(ytest==cls).astype(int)
                    if len(np.unique(truth))==2:
                        fpr,tpr,_=roc_curve(truth,proba[:,LABELS.index(cls)])
                        curves.append(pd.DataFrame({'fold':fold.number,'scheme':scheme,'model':name,
                                                    'class_id':cls,'fpr':fpr,'tpr':tpr}))
                print(f'  {scheme:24} {name:15} accuracy={results[-1]["accuracy"]:.4f} auc={results[-1]["roc_auc_ovr_macro"]:.4f} fit={fit:.1f}s',flush=True)
                if out:pd.DataFrame(results).to_csv(out/'development_fold_metrics.csv',index=False)
    metrics=pd.DataFrame(results)
    roc=pd.concat(curves,ignore_index=True) if curves else pd.DataFrame()
    if out:
        metrics.to_parquet(out/'development_fold_metrics.parquet',index=False)
        roc.to_parquet(out/'development_roc_curves.parquet',index=False)
        (out/'holdout_lock.json').write_text(json.dumps({'boundary':str(boundary),'holdout_fraction':holdout_fraction,
            'status':'NOT_EVALUATED','n_folds':n_folds,'schemes':list(schemes),
            'warning':'Do not select configurations using final holdout scores.'},indent=2))
    return metrics,roc,eligibility,boundary

def summarize_ablation(metrics):
    summary=metrics.groupby(['scheme','model'],as_index=False).agg(
        accuracy=('accuracy','mean'),accuracy_std=('accuracy','std'),
        balanced_accuracy=('balanced_accuracy','mean'),auc=('roc_auc_ovr_macro','mean'),
        auc_std=('roc_auc_ovr_macro','std'),log_loss=('log_loss','mean'),
        fit_seconds=('fit_seconds','sum'))
    return summary.sort_values(['auc','accuracy'],ascending=False)

def plot_ablation(metrics,curves,output_dir=None):
    import matplotlib.pyplot as plt
    summary=summarize_ablation(metrics)
    view=summary[summary.model.eq('logistic_l2')].sort_values('accuracy')
    fig,ax=plt.subplots(figsize=(10,5))
    ax.barh(view.scheme,view.accuracy*100)
    ax.set(xlabel='Five-fold mean accuracy (%)',title='Feature-group ablation — Logistic L2')
    fig.tight_layout()
    if output_dir:fig.savefig(Path(output_dir)/'ablation_accuracy.png',dpi=160)
    plt.show()
    best=summary[summary.model.ne('prior_baseline')].head(4)
    fig,ax=plt.subplots(figsize=(9,6)); grid=np.linspace(0,1,201)
    for _,row in best.iterrows():
        part=curves[(curves.scheme==row.scheme)&(curves.model==row.model)]
        foldlines=[]
        for _,f in part.groupby('fold'):
            cl=[np.interp(grid,g.fpr,g.tpr) for _,g in f.groupby('class_id')]
            if cl:foldlines.append(np.mean(cl,axis=0))
        if foldlines:ax.plot(grid,np.mean(foldlines,axis=0),label=f'{row.scheme} / {row.model}: {row.auc:.3f}')
    ax.plot([0,1],[0,1],'k--',alpha=.5,label='Random')
    ax.set(xlabel='False positive rate',ylabel='True positive rate',title='Development-fold mean macro OVR ROC curves')
    ax.legend(fontsize=8);fig.tight_layout()
    if output_dir:fig.savefig(Path(output_dir)/'ablation_roc.png',dpi=160)
    plt.show()
    return summary
