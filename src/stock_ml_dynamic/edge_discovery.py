"""Notebook 39: broad, leakage-controlled, zero-cost economic edge discovery.

All fitted preprocessing uses training observations only. Historical final holdout
and prospective prediction ledger are excluded by the notebook caller.
"""
from pathlib import Path
import time
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge, ElasticNet, HuberRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_regression
from .walk_forward import expanding_folds, fold_data
from .economic_target_research import prepare_extras
from .payoff_search import payoff_stats, daily_top_k, date_block_ci

K_VALUES=(5,10,20)

def random_portfolio_null(frame,k=5,reps=1000,seed=42):
    """Independent random selection within each date, preserving market/date effects."""
    rng=np.random.default_rng(seed)
    groups=[g.next_return.to_numpy(float) for _,g in frame.groupby('bar_timestamp',sort=True)]
    groups=[x[np.isfinite(x)] for x in groups if len(x)]
    if not groups:raise ValueError('No eligible dates')
    draws=np.zeros(reps)
    for a in groups:
        m=min(k,len(a))
        # Sample without replacement within each day for each randomized portfolio.
        for i in range(reps):draws[i]+=a[rng.choice(len(a),size=m,replace=False)].mean()
    return draws/len(groups)

def tail_sensitivity(returns,trim=(0,.005,.01,.025)):
    """Remove upper tail of realized winners as a *diagnostic*, never for selection."""
    a=np.asarray(returns,float);a=a[np.isfinite(a)]
    result=[]
    for fraction in trim:
        n=int(np.floor(len(a)*fraction))
        b=np.sort(a)[:len(a)-n] if n else a
        s=payoff_stats(b)
        result.append(dict(upper_tail_removed=fraction,**s))
    return pd.DataFrame(result)

def _daily_ic(frame,score):
    z=frame[['bar_timestamp','next_return']].copy();z['score']=score
    vals=[]
    for _,g in z.groupby('bar_timestamp'):
        if len(g)>7 and g.score.nunique()>1 and g.next_return.nunique()>1:
            v=spearmanr(g.score,g.next_return).statistic
            if np.isfinite(v):vals.append(v)
    return float(np.mean(vals)) if vals else np.nan

def _candidate_specs(seed,trees):
    return [
      ('ridge_a1',Ridge(alpha=1.),'scaled','all'),
      ('ridge_a30',Ridge(alpha=30.),'scaled','all'),
      ('ridge_a100',Ridge(alpha=100.),'scaled','all'),
      ('ridge_a1000',Ridge(alpha=1000.),'scaled','all'),
      ('ridge_top30',Ridge(alpha=100.),'scaled','top30'),
      ('elasticnet_l1_02',ElasticNet(alpha=.0001,l1_ratio=.2,max_iter=2500,selection='cyclic'),'scaled','all'),
      ('elasticnet_l1_08',ElasticNet(alpha=.0001,l1_ratio=.8,max_iter=2500,selection='cyclic'),'scaled','all'),
      ('huber',HuberRegressor(epsilon=1.5,alpha=1.,max_iter=180),'scaled','top30'),
      ('hgb_leaf15',HistGradientBoostingRegressor(max_iter=180,max_leaf_nodes=15,learning_rate=.05,l2_regularization=1,random_state=seed),'raw','all'),
      ('hgb_leaf31',HistGradientBoostingRegressor(max_iter=180,max_leaf_nodes=31,learning_rate=.05,l2_regularization=5,random_state=seed),'raw','all'),
      ('extra_trees',ExtraTreesRegressor(n_estimators=trees,max_depth=14,min_samples_leaf=20,n_jobs=-1,random_state=seed),'raw','all'),
    ]

def run_edge_discovery(frame,feature_cols,*,cap=60000,trees=120,reps=1000,seed=42,output_dir=None):
    """Three chronological folds; 11 model variants, matched top-k, random null and tails."""
    metrics=[]; trades=[]; timing=[]; nulls=[]; audit=[]; paired=[]; tails=[];daily=[]
    for fold in expanding_folds(frame,n_folds=3):
        tr,va,xb,xv,_=fold_data(frame,fold,target_q=7)
        last=pd.to_datetime(tr.next_bar_timestamp,utc=True).max()
        first=pd.to_datetime(va.bar_timestamp,utc=True).min()
        assert last<=fold.train_end<first, 'Training labels overlap validation'
        ax,az=prepare_extras(tr,va,feature_cols)
        X=np.column_stack([xb.astype('float32'),ax]);Z=np.column_stack([xv.astype('float32'),az])
        rng=np.random.default_rng(seed+fold.number)
        idx=np.sort(rng.choice(len(X),size=min(cap,len(X)),replace=False))
        x=X[idx];ret=tr.next_return.to_numpy(float)[idx]
        low,high=np.quantile(ret,[.005,.995]);y=np.clip(ret,low,high)
        scaler=StandardScaler();xs=scaler.fit_transform(x);zs=scaler.transform(Z)
        # Training-only univariate screening; no validation returns used.
        fscores=f_regression(xs,y)[0];fscores=np.nan_to_num(fscores,nan=-np.inf,posinf=-np.inf)
        selected=np.argsort(fscores)[-min(30,x.shape[1]):]
        audit.append(dict(fold=fold.number,train_rows=len(tr),validation_rows=len(va),train_sample=len(idx),feature_count=x.shape[1],selected_features=len(selected),last_train_label=last,first_validation_bar=first))
        # Market baseline: same-date equal-weight universe; never a trained predictive model.
        base=va.groupby('bar_timestamp').next_return.mean()
        for k in K_VALUES:
            null=random_portfolio_null(va,k=k,reps=reps,seed=seed+fold.number*100+k)
            nulls.append(dict(fold=fold.number,k=k,random_mean=float(null.mean()),random_ci_low=float(np.quantile(null,.025)),random_ci_high=float(np.quantile(null,.975)),random_sd=float(null.std()),universe_daily_mean=float(base.mean())))
        for name,model,mode,subset in _candidate_specs(seed+fold.number,trees):
            xx,zz=(xs,zs) if mode=='scaled' else (x,Z)
            if subset=='top30':xx=xx[:,selected];zz=zz[:,selected]
            t0=time.perf_counter()
            model.fit(xx,y);score=np.asarray(model.predict(zz),float)
            timing.append(dict(fold=fold.number,model=name,seconds=time.perf_counter()-t0,ic=_daily_ic(va,score)))
            ranked=daily_top_k(va,score,max(K_VALUES),name,fold.number)
            for k in K_VALUES:
                chosen=ranked.loc[ranked['rank']<=k,['fold','model','k','bar_timestamp','symbol','next_return','score','rank']].copy()
                chosen['k']=k;trades.append(chosen)
                byday=chosen.groupby('bar_timestamp').next_return.mean()
                d=byday.rename('daily_return').reset_index();d['model']=name;d['fold']=fold.number;d['k']=k;daily.append(d)
                p=payoff_stats(chosen.next_return)
                nullrow=next(z for z in nulls if z['fold']==fold.number and z['k']==k)
                metrics.append(dict(fold=fold.number,model=name,k=k,**p,mean_daily_return=float(byday.mean()),market_excess_daily=float((byday-base.reindex(byday.index)).mean()),random_lift=float(byday.mean()-nullrow['random_mean']),random_z=float((byday.mean()-nullrow['random_mean'])/max(nullrow['random_sd'],1e-12))))
                tail=tail_sensitivity(chosen.next_return);tail['fold']=fold.number;tail['model']=name;tail['k']=k;tails.append(tail)
    result=dict(metrics=pd.DataFrame(metrics),trades=pd.concat(trades,ignore_index=True),timing=pd.DataFrame(timing),null=pd.DataFrame(nulls),audit=pd.DataFrame(audit),tails=pd.concat(tails,ignore_index=True),daily=pd.concat(daily,ignore_index=True))
    # Paired daily confidence intervals vs original ridge setting, with identical dates and k.
    for (fold,k),g in result['daily'].groupby(['fold','k']):
        p=g.pivot(index='bar_timestamp',columns='model',values='daily_return')
        for name in p.columns:
            if name=='ridge_a100':continue
            delta=(p[name]-p['ridge_a100']).dropna()
            lo,hi=date_block_ci(delta,block=10,reps=1000,seed=seed+fold+k)
            paired.append(dict(fold=fold,k=k,challenger=name,mean_daily_lift=float(delta.mean()),ci_low=lo,ci_high=hi,days=len(delta)))
    result['paired']=pd.DataFrame(paired)
    if output_dir:
        dest=Path(output_dir);dest.mkdir(parents=True,exist_ok=True)
        for key,df in result.items():df.to_csv(dest/f'{key}.csv',index=False)
    return result
