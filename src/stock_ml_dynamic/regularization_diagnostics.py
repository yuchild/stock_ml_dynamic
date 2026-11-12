"""Nested chronological L2 regularization tuning for three-class next-bar labels.

The latest 20% of timestamps is locked out. Each outer fold uses an inner
chronological split for C selection, with all learned transforms refitted.
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
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, log_loss, roc_curve
from .classifier_benchmark import eligible_universe
from .historical_market import add_historical_market_features
from .feature_ablation import development_holdout_split
from .walk_forward import Fold, expanding_folds, fold_data

CLASSES = [0, 1, 2]

def inner_fold_for_outer(frame, outer, fraction=0.2):
    if not 0.05 <= fraction <= 0.4:
        raise ValueError('Inner validation fraction must be 0.05–0.40')
    ts = pd.DatetimeIndex(pd.to_datetime(frame.loc[frame.bar_timestamp <= outer.train_end, 'bar_timestamp'],utc=True).unique()).sort_values()
    if len(ts)<50: raise ValueError('Not enough timestamps for inner validation')
    split = ts[max(0,int(len(ts)*(1-fraction))-1)]
    if split >= outer.train_end: raise ValueError('Inner split is not chronological')
    return Fold(outer.number, split, outer.train_end)

def matrices(frame, fold, extra, *, selected_k=4, cluster_fit_limit=15000):
    tr,te,xb,xv,_=fold_data(frame,fold,selected_k=selected_k,cluster_fit_limit=cluster_fit_limit)
    a=tr[extra].replace([np.inf,-np.inf],np.nan)
    b=te[extra].replace([np.inf,-np.inf],np.nan)
    valid_a=a.notna().all(axis=1).to_numpy();valid_b=b.notna().all(axis=1).to_numpy()
    if valid_a.sum()<100 or valid_b.sum()<20:raise ValueError('Insufficient valid rows')
    a=a.loc[valid_a];b=b.loc[valid_b]
    scaler=StandardScaler().fit(a)
    columns=[c for c in extra if c!='realized_vol_10']
    ia=[extra.index(c) for c in columns]
    x=np.column_stack((xb[valid_a],scaler.transform(a)[:,ia]))
    z=np.column_stack((xv[valid_b],scaler.transform(b)[:,ia]))
    return x,tr.direction.to_numpy(dtype=int)[valid_a],z,te.direction.to_numpy(dtype=int)[valid_b],tr.loc[valid_a],te.loc[valid_b]

def sample_rows(x,y,cap,seed):
    if cap and len(y)>cap:
        ix=np.sort(np.random.default_rng(seed).choice(len(y),cap,replace=False))
        return x[ix],y[ix]
    return x,y

def metrics(y,model,x):
    p=model.predict_proba(x)
    aligned=np.zeros((len(x),3))
    for j,c in enumerate(model.classes_):aligned[:,CLASSES.index(int(c))]=p[:,j]
    pred=model.predict(x)
    return {'accuracy':float(accuracy_score(y,pred)), 'balanced_accuracy':float(balanced_accuracy_score(y,pred)),
            'roc_auc':float(roc_auc_score(y,aligned,labels=CLASSES,multi_class='ovr',average='macro')),
            'log_loss':float(log_loss(y,aligned,labels=CLASSES))},aligned

def run_regularization_diagnostics(frame, *, n_folds=5, holdout_fraction=.2, c_grid=(.03,.1,.3,1.),
                                   inner_fraction=.2, max_train_rows=120000, selected_k=4,
                                   cluster_fit_limit=15000, output_dir=None):
    if not c_grid or any(float(c)<=0 for c in c_grid):raise ValueError('C grid must be positive')
    eligible,eligibility=eligible_universe(frame)
    expanded,extra=add_historical_market_features(eligible)
    dev,boundary=development_holdout_split(expanded,holdout_fraction)
    output=Path(output_dir) if output_dir else None
    if output:
        output.mkdir(parents=True,exist_ok=True)
        eligibility.to_csv(output/'universe_eligibility.csv',index=False)
        (output/'holdout_lock.json').write_text(json.dumps({'status':'NOT_EVALUATED','development_end':str(boundary),'holdout_fraction':holdout_fraction},indent=2))
    rows=[];tuning=[];curves=[]
    for outer in expanding_folds(dev,n_folds=n_folds):
        t0=time.perf_counter()
        inner=inner_fold_for_outer(dev,outer,inner_fraction)
        xi,yi,xv,yv,_,_=matrices(dev,inner,extra,selected_k=selected_k,cluster_fit_limit=cluster_fit_limit)
        xi,yi=sample_rows(xi,yi,max_train_rows,110+outer.number)
        for c in c_grid:
            model=LogisticRegression(C=float(c),max_iter=600,solver='lbfgs')
            start=time.perf_counter();model.fit(xi,yi);elapsed=time.perf_counter()-start
            score,_=metrics(yv,model,xv)
            tuning.append({'fold':outer.number,'C':float(c),'inner_train_end':str(inner.train_end),'inner_test_end':str(inner.test_end),
                           'fit_seconds':elapsed,**score})
        fold_tuning=pd.DataFrame([r for r in tuning if r['fold']==outer.number])
        # Inner log-loss is the only tuning criterion; outer fold never selects C.
        best=float(fold_tuning.sort_values(['log_loss','C']).iloc[0]['C'])
        xo,yo,xt,yt,_,_=matrices(dev,outer,extra,selected_k=selected_k,cluster_fit_limit=cluster_fit_limit)
        xo,yo=sample_rows(xo,yo,max_train_rows,210+outer.number)
        for name,model in [('prior_baseline',DummyClassifier(strategy='prior')),
                           ('logistic_l2_tuned',LogisticRegression(C=best,max_iter=600,solver='lbfgs')),
                           ('logistic_l2_fixed',LogisticRegression(C=.1,max_iter=600,solver='lbfgs'))]:
            start=time.perf_counter();model.fit(xo,yo);elapsed=time.perf_counter()-start
            score,prob=metrics(yt,model,xt)
            rows.append({'fold':outer.number,'model':name,'selected_C':best if name=='logistic_l2_tuned' else (.1 if name=='logistic_l2_fixed' else np.nan),
                         'fit_rows':len(yo),'test_rows':len(yt),'fit_seconds':elapsed,'total_fold_seconds':time.perf_counter()-t0,
                         'train_end':str(outer.train_end),'test_end':str(outer.test_end),**score})
            for cls in CLASSES:
                truth=(yt==cls).astype(int)
                if truth.min()!=truth.max():
                    fpr,tpr,_=roc_curve(truth,prob[:,CLASSES.index(cls)])
                    curves.append(pd.DataFrame({'fold':outer.number,'model':name,'class_id':cls,'fpr':fpr,'tpr':tpr}))
            print(f'Fold {outer.number}/{n_folds}: {name}, C={best:.3g}, acc={score["accuracy"]:.4f}, AUC={score["roc_auc"]:.4f}, fit={elapsed:.1f}s',flush=True)
        if output:
            pd.DataFrame(rows).to_csv(output/'fold_metrics.csv',index=False)
            pd.DataFrame(tuning).to_csv(output/'inner_tuning.csv',index=False)
    scores=pd.DataFrame(rows);tune=pd.DataFrame(tuning);roc=pd.concat(curves,ignore_index=True)
    if output:
        roc.to_parquet(output/'roc_curves.parquet',index=False)
        summary_regularization(scores).to_csv(output/'summary.csv',index=False)
    return scores,tune,roc,eligibility,boundary

def summary_regularization(scores):
    return scores.groupby('model',as_index=False).agg(accuracy=('accuracy','mean'),accuracy_sd=('accuracy','std'),
       balanced_accuracy=('balanced_accuracy','mean'),roc_auc=('roc_auc','mean'),auc_sd=('roc_auc','std'),
       log_loss=('log_loss','mean'),total_fit_seconds=('fit_seconds','sum')).sort_values('roc_auc',ascending=False)

def plot_regularization(scores,tune,roc,output_dir=None):
    import matplotlib.pyplot as plt
    out=Path(output_dir) if output_dir else None
    summary=summary_regularization(scores)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,metric in zip(axes,('accuracy','roc_auc')):
        for model,g in scores.groupby('model'):
            ax.plot(g.fold,g[metric],marker='o',label=model)
        ax.set(xlabel='Outer fold',ylabel=metric,title=f'Outer-fold {metric}')
        ax.legend(fontsize=8)
    fig.tight_layout()
    if out:fig.savefig(out/'outer_scores.png',dpi=150)
    plt.show()
    grid=np.linspace(0,1,201)
    fig,ax=plt.subplots(figsize=(7,6))
    for model,sub in roc.groupby('model'):
        fold_lines=[]
        for _,f in sub.groupby('fold'):
            curves=[np.interp(grid,c.fpr,c.tpr) for _,c in f.groupby('class_id')]
            fold_lines.append(np.mean(curves,axis=0))
        ax.plot(grid,np.mean(fold_lines,axis=0),label=model)
    ax.plot([0,1],[0,1],'k--',alpha=.5)
    ax.set(xlabel='False positive rate',ylabel='True positive rate',title='Mean one-vs-rest ROC · development folds')
    ax.legend();fig.tight_layout()
    if out:fig.savefig(out/'roc_curves.png',dpi=150)
    plt.show()
    return summary
