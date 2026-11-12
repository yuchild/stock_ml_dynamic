"""Notebook 19: strictly chronological development-only calibration and stacking.

All inner preprocessing (including target quantiles and candle clustering) is
refit within each inner train window. The untouched 20% holdout is excluded.
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from .architecture_ensemble import blend_probabilities, normalize_probabilities
from .classifier_benchmark import eligible_universe
from .historical_market import add_historical_market_features
from .feature_ablation import development_holdout_split
from .regularization_diagnostics import matrices
from .walk_forward import Fold, expanding_folds
from .nonlinear_benchmark import aligned_proba, score_probabilities

CLASSES=(0,1,2)

def temperature_scale(probs, temperature):
    if not np.isfinite(temperature) or temperature<=0:raise ValueError('Temperature must be positive')
    p=normalize_probabilities(probs)
    logits=np.log(np.clip(p,1e-12,1))/float(temperature)
    logits-=logits.max(axis=1,keepdims=True)
    exp=np.exp(logits)
    return normalize_probabilities(exp/exp.sum(axis=1,keepdims=True))

def fit_temperature(probs,y):
    p=normalize_probabilities(probs)
    y=np.asarray(y,dtype=int)
    if len(y)!=len(p) or not set(np.unique(y)).issubset(CLASSES):raise ValueError('Invalid calibration labels')
    result=minimize_scalar(lambda log_t:log_loss(y,temperature_scale(p,np.exp(log_t)),labels=CLASSES),bounds=(np.log(.3),np.log(5)),method='bounded')
    if not result.success:raise RuntimeError('Temperature optimization failed')
    return float(np.exp(result.x))

def chronological_blocks(frame,outer,breaks=(.55,.70,.82,.94)):
    """Three sequential inner prediction windows: two meta, one calibration."""
    if len(breaks)!=4 or not all(0<a<b<1 for a,b in zip(breaks,breaks[1:])):raise ValueError('Invalid inner boundaries')
    ts=pd.DatetimeIndex(pd.to_datetime(frame.loc[pd.to_datetime(frame.next_bar_timestamp,utc=True)<=outer.train_end,'bar_timestamp'],utc=True).unique()).sort_values()
    if len(ts)<70:raise ValueError('Insufficient timestamps for inner calibration')
    edges=[ts[min(len(ts)-1,int((len(ts)-1)*f))] for f in breaks]
    if len(set(edges))!=4:raise ValueError('Inner windows collapsed')
    return [Fold(i+1,edges[i],edges[i+1]) for i in range(3)]

def sample(x,y,cap,seed):
    if cap and len(y)>cap:
        ix=np.sort(np.random.default_rng(seed).choice(len(y),cap,replace=False))
        return x[ix],y[ix]
    return x,y

def fit_base(x,y,*,seed,tree_cap=20000):
    lr=LogisticRegression(C=.1,max_iter=600)
    rf=RandomForestClassifier(n_estimators=160,max_depth=12,min_samples_leaf=30,max_features='sqrt',n_jobs=-1,random_state=seed)
    lr.fit(x,y)
    xt,yt=sample(x,y,tree_cap,seed+1000)
    rf.fit(xt,yt)
    return lr,rf

def base_probs(models,x):
    return [normalize_probabilities(aligned_proba(m,x)) for m in models]

def ece(probs,y,n_bins=10):
    p=normalize_probabilities(probs); y=np.asarray(y)
    conf=p.max(axis=1); pred=p.argmax(axis=1); correct=(pred==y).astype(float)
    edges=np.linspace(0,1,n_bins+1); value=0.
    for j in range(n_bins):
        mask=(conf>=edges[j]) & ((conf<=edges[j+1]) if j==n_bins-1 else (conf<edges[j+1]))
        if mask.any():value+=mask.mean()*abs(correct[mask].mean()-conf[mask].mean())
    return float(value)

def run_temporal_calibration(master,*,n_folds=2,max_train_rows=60000,tree_train_rows=35000,seed=42,output_dir=None):
    """Compare raw RF, raw 50/50 blend, calibrated blend, and nested temporal stack.

    Inner meta training uses first two chronological blocks. The third block
    independently fits temperatures; outer test is never used for fitting.
    """
    eligible,eligibility=eligible_universe(master)
    expanded,extra=add_historical_market_features(eligible)
    dev,boundary=development_holdout_split(expanded,.2)
    out=Path(output_dir) if output_dir else None
    if out:out.mkdir(parents=True,exist_ok=True)
    results=[];manifest=[]
    for outer in expanding_folds(dev,n_folds=n_folds):
        start=time.perf_counter();blocks=chronological_blocks(dev,outer)
        inner_predictions=[];inner_y=[]
        for inner in blocks:
            xi,yi,xv,yv,tr,va=matrices(dev,inner,extra)
            xi,yi=sample(xi,yi,max_train_rows,seed+outer.number+inner.number)
            models=fit_base(xi,yi,seed=seed+100*outer.number+inner.number,tree_cap=tree_train_rows)
            p=base_probs(models,xv)
            inner_predictions.append(np.hstack(p));inner_y.append(yv)
            if not (pd.to_datetime(tr.next_bar_timestamp,utc=True).max()<=inner.train_end and pd.to_datetime(va.bar_timestamp,utc=True).min()>inner.train_end):
                raise RuntimeError('Inner temporal boundary violation')
        # Fit meta learner exclusively on first two inner blocks.
        meta=LogisticRegression(C=.1,max_iter=600)
        meta.fit(np.vstack(inner_predictions[:2]),np.concatenate(inner_y[:2]))
        cal_features=inner_predictions[2];cy=inner_y[2]
        cal_blend=blend_probabilities([cal_features[:,:3],cal_features[:,3:]],[1,1])
        cal_stack=normalize_probabilities(aligned_proba(meta,cal_features))
        t_blend=fit_temperature(cal_blend,cy)
        t_stack=fit_temperature(cal_stack,cy)
        # Refit all base preprocessing/models using only outer train; no outer test exposure.
        xo,yo,xt,yt,tr,te=matrices(dev,outer,extra)
        xo,yo=sample(xo,yo,max_train_rows,seed+outer.number)
        models=fit_base(xo,yo,seed=seed+outer.number,tree_cap=tree_train_rows)
        p_lr,p_rf=base_probs(models,xt)
        raw_blend=blend_probabilities([p_lr,p_rf],[1,1])
        raw_stack=normalize_probabilities(aligned_proba(meta,np.hstack([p_lr,p_rf])))
        preds={'logistic_l2':p_lr,'random_forest':p_rf,'soft_vote_lr_rf':raw_blend,
               'soft_vote_calibrated':temperature_scale(raw_blend,t_blend),
               'temporal_stack':raw_stack,'temporal_stack_calibrated':temperature_scale(raw_stack,t_stack)}
        for name,p in preds.items():
            metrics=score_probabilities(yt,p)
            results.append({'fold':outer.number,'model':name,'train_rows':len(yo),'test_rows':len(yt),
                            'train_end':str(outer.train_end),'test_end':str(outer.test_end),
                            'ece':ece(p,yt),'elapsed_fold_seconds':time.perf_counter()-start,**metrics})
            print(f'Fold {outer.number}/{n_folds} {name}: AUC={metrics["roc_auc"]:.4f} log_loss={metrics["log_loss"]:.4f}',flush=True)
        manifest.append({'outer_fold':outer.number,'outer_train_end':str(outer.train_end),'outer_test_end':str(outer.test_end),
                         'inner_meta_windows':[(str(f.train_end),str(f.test_end)) for f in blocks[:2]],
                         'calibration_window':(str(blocks[2].train_end),str(blocks[2].test_end)),
                         'temperature_blend':t_blend,'temperature_stack':t_stack,
                         'meta_rows':int(sum(map(len,inner_y[:2]))),'calibration_rows':int(len(cy))})
        if out:pd.DataFrame(results).to_csv(out/'fold_metrics.csv',index=False)
    scores=pd.DataFrame(results)
    summary=scores.groupby('model',as_index=False).agg(roc_auc_mean=('roc_auc','mean'),roc_auc_std=('roc_auc','std'),accuracy_mean=('accuracy','mean'),balanced_accuracy_mean=('balanced_accuracy','mean'),log_loss_mean=('log_loss','mean'),ece_mean=('ece','mean')).sort_values('roc_auc_mean',ascending=False)
    if out:
        summary.to_csv(out/'summary.csv',index=False)
        eligibility.to_csv(out/'universe_eligibility.csv',index=False)
        (out/'manifest.json').write_text(json.dumps({'status':'DEVELOPMENT_ONLY','development_end':str(boundary),'final_holdout':'NOT_ACCESSED','prospective_ledger':'NOT_ACCESSED','inner_fold_preprocessing':'REFIT_PER_INNER_FOLD','note':'Meta trained on two earlier blocks; temperatures trained on later calibration block; outer test untouched. Fold scores are correlated.', 'folds':manifest},indent=2))
    return scores,summary,pd.DataFrame(manifest),boundary
