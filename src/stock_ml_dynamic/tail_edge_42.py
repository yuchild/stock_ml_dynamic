"""Notebook 42: development-only upside/downside event modeling.

All fitting, clipping, event definitions, and scaling use training data only.
This module never loads the closed historical holdout or prediction ledger.
"""
from pathlib import Path
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss, roc_auc_score, average_precision_score
from .walk_forward import expanding_folds, fold_data
from .economic_target_research import prepare_extras
from .edge_robustness_41 import rank_with_exclusions, _metrics
from .payoff_search import date_block_ci

THRESHOLDS=(.03,.05,.10)
POSITIONS=(5,10,20)

def _event(y,t):return (np.asarray(y)>t).astype(int)

def _safe_auc(y,p):
    return float(roc_auc_score(y,p)) if len(np.unique(y))==2 else np.nan

def _logistic_prob(X,y,Z,seed):
    if len(np.unique(y))<2:return np.full(len(Z),float(np.mean(y)))
    clf=LogisticRegression(C=.1,max_iter=250,solver='lbfgs',random_state=seed)
    clf.fit(X,y)
    return clf.predict_proba(Z)[:,list(clf.classes_).index(1)]

def _hgb_prob(X,y,Z,seed):
    if len(np.unique(y))<2:return np.full(len(Z),float(np.mean(y)))
    clf=HistGradientBoostingClassifier(max_iter=90,max_leaf_nodes=15,l2_regularization=10.,learning_rate=.05,min_samples_leaf=100,random_state=seed)
    clf.fit(X,y)
    return clf.predict_proba(Z)[:,list(clf.classes_).index(1)]

def _fit_fold(tr,va,xb,xv,obj,extra_cols,cap,seed,fold):
    ax,az=prepare_extras(tr,va,extra_cols)
    X=np.column_stack([np.asarray(xb,dtype='float32'),ax]);Z=np.column_stack([np.asarray(xv,dtype='float32'),az])
    rng=np.random.default_rng(seed+fold)
    ids=np.sort(rng.choice(len(X),min(cap,len(X)),replace=False))
    y=tr.next_return.to_numpy(float)[ids]
    lo,hi=np.quantile(y,[.005,.995]);yc=np.clip(y,lo,hi)
    scale=StandardScaler();xs=scale.fit_transform(X[ids]);zs=scale.transform(Z)
    base=Ridge(alpha=100.).fit(xs,yc).predict(zs)
    # All tail probabilities use the same capped training sample and fixed absolute event thresholds.
    scores={'ridge_all':base}
    diagnostics=[]
    for t in THRESHOLDS:
        binary=_event(y,t)
        for name,fn in [('logistic',_logistic_prob),('hgb',_hgb_prob)]:
            tic=time.perf_counter()
            pred=fn(xs,binary,zs,seed+fold)
            scores[f'up_{int(t*100)}pct_{name}']=pred
            diagnostics.append(dict(fold=fold,threshold=t,model=name,training_prevalence=float(binary.mean()),validation_prevalence=float(_event(va.next_return,t).mean()),auc=_safe_auc(_event(va.next_return,t),pred),average_precision=float(average_precision_score(_event(va.next_return,t),pred)) if _event(va.next_return,t).sum() else np.nan,brier=float(brier_score_loss(_event(va.next_return,t),pred)),fit_seconds=time.perf_counter()-tic))
    # Quantile regressors: lower, median, upper conditional-return estimates.
    for q in (.1,.5,.9):
        tic=time.perf_counter()
        m=HistGradientBoostingRegressor(loss='quantile',quantile=q,max_iter=100,max_leaf_nodes=15,learning_rate=.05,l2_regularization=10.,min_samples_leaf=100,random_state=seed+fold)
        m.fit(X[ids],yc)
        scores[f'q{int(q*100)}']=m.predict(Z)
        diagnostics.append(dict(fold=fold,threshold=np.nan,model=f'quantile_{q}',training_prevalence=np.nan,validation_prevalence=np.nan,auc=np.nan,average_precision=np.nan,brier=np.nan,fit_seconds=time.perf_counter()-tic))
    # Ex-ante combinations: no validation outcomes used for ranking.
    p=scores['up_5pct_logistic'];h=scores['up_5pct_hgb']
    scores['two_stage_ridge_x_p5']=base*p
    scores['two_stage_ridge_x_hgb5']=base*h
    scores['upside_minus_downside']=scores['q90']-np.abs(np.minimum(scores['q10'],0))
    scores['upside_downside_ratio']=np.maximum(scores['q90'],0)/(np.maximum(-scores['q10'],0)+.01)
    scores['quantile_midpoint']=(scores['q90']+scores['q10'])/2
    scores['quantile_spread']=scores['q90']-scores['q10']
    scores['ridge_per_spread']=base/(np.maximum(scores['quantile_spread'],.005))
    scores['ridge_plus_tail_prob']=base+.05*(p-float(_event(y,.05).mean()))
    v=va[['bar_timestamp','symbol','next_return']].copy().reset_index(drop=True)
    v['fold']=fold
    for key,val in scores.items():v[key]=np.asarray(val,float)
    audit=dict(fold=fold,train_rows=len(tr),validation_rows=len(va),train_sample=len(ids),training_label_end=str(pd.to_datetime(tr.next_bar_timestamp,utc=True).max()),validation_start=str(pd.to_datetime(va.bar_timestamp,utc=True).min()),n_features=X.shape[1],target_clip_low=float(lo),target_clip_high=float(hi))
    return v,diagnostics,audit

def run_tail_research(frame,extra_cols,cap=60000,seed=42,k_values=POSITIONS):
    scored=[];diagnostics=[];audit=[]
    for f in expanding_folds(frame,n_folds=3):
        tr,va,xb,xv,obj=fold_data(frame,f,target_q=7)
        assert pd.to_datetime(tr.next_bar_timestamp,utc=True).max()<=f.train_end
        assert pd.to_datetime(va.bar_timestamp,utc=True).min()>f.train_end
        v,d,a=_fit_fold(tr,va,xb,xv,obj,extra_cols,cap,seed,f.number)
        scored.append(v);diagnostics.extend(d);audit.append(a)
    scores=pd.concat(scored,ignore_index=True)
    metrics=[];daily=[];trades=[]
    score_names=[c for c in scores if c not in ('bar_timestamp','symbol','next_return','fold')]
    for fold,v in scores.groupby('fold'):
        v=v.reset_index(drop=True)
        market=v.groupby('bar_timestamp').next_return.mean()
        for name in score_names:
            for k in k_values:
                chosen=rank_with_exclusions(v,v[name].to_numpy(),k)
                if chosen.empty:continue
                m,d=_metrics(chosen,name,int(fold),k,market);metrics.append(m)
                chosen['fold']=fold;chosen['model']=name;chosen['k']=k;trades.append(chosen)
                daily.extend(dict(fold=fold,model=name,k=k,bar_timestamp=t,daily_return=float(r)) for t,r in d.items())
    return dict(scored=scores,metrics=pd.DataFrame(metrics),daily=pd.DataFrame(daily),trades=pd.concat(trades,ignore_index=True),events=pd.DataFrame(diagnostics),audit=pd.DataFrame(audit))

def compare_to_ridge(daily,k=5,reps=1200,seed=42):
    rows=[]
    x=daily.loc[daily['k'].eq(k)]
    for fold,g in x.groupby('fold'):
        p=g.pivot(index='bar_timestamp',columns='model',values='daily_return')
        for name in p.columns:
            if name=='ridge_all':continue
            d=(p[name]-p['ridge_all']).dropna()
            if len(d)<20:continue
            lo,hi=date_block_ci(d,block=10,reps=reps,seed=seed+int(fold))
            rows.append(dict(fold=fold,model=name,k=k,days=len(d),mean_lift=float(d.mean()),ci_low=lo,ci_high=hi))
    return pd.DataFrame(rows)

def tail_concentration(trades,top_fractions=(.005,.01,.025)):
    rows=[]
    for (fold,model,k),g in trades.groupby(['fold','model','k']):
        total=g.next_return.sum()
        for frac in top_fractions:
            n=max(1,int(np.ceil(len(g)*frac)))
            top=g.next_return.nlargest(n).sum()
            rest=g.drop(g.next_return.nlargest(n).index).next_return.mean()
            rows.append(dict(fold=fold,model=model,k=k,top_fraction=frac,top_sum=float(top),net_sum=float(total),top_share_of_net=float(top/total) if abs(total)>1e-12 else np.nan,rest_mean=float(rest)))
    return pd.DataFrame(rows)

def event_calibration(scored,threshold=.05,bins=10):
    rows=[]
    target=(scored.next_return.to_numpy(float)>threshold).astype(int)
    for fold,g in scored.groupby('fold'):
        y=target[g.index.to_numpy()]
        for model in ('up_5pct_logistic','up_5pct_hgb'):
            p=g[model].to_numpy(float)
            # Fixed probability bins, not hindsight-selected percentile thresholds.
            edges=np.linspace(0,1,bins+1)
            b=np.clip(np.digitize(p,edges,right=False)-1,0,bins-1)
            for i in range(bins):
                mask=b==i
                if mask.sum():rows.append(dict(fold=fold,model=model,bin=i,n=int(mask.sum()),predicted=float(p[mask].mean()),observed=float(y[mask].mean())))
    return pd.DataFrame(rows)

def save_outputs(path,**tables):
    p=Path(path);p.mkdir(parents=True,exist_ok=True)
    for name,df in tables.items():df.to_csv(p/(name+'.csv'),index=False)
