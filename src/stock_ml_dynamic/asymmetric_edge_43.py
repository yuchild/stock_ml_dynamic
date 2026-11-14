"""Notebook 43: chronological, development-only asymmetric tail research.
No validation labels enter fitting, calibration, or ranking. No final holdout access.
"""
from pathlib import Path
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from .walk_forward import expanding_folds, fold_data
from .economic_target_research import prepare_extras
from .edge_robustness_41 import rank_with_exclusions, _metrics
from .payoff_search import date_block_ci

THRESHOLDS=(.03,.05,.10)
K_VALUES=(5,10,20)

def safe_auc(y,p):
    return float(roc_auc_score(y,p)) if np.unique(y).size==2 else np.nan

def fit_binary(x,y,z,kind,seed):
    if np.unique(y).size<2:return np.full(len(z),float(np.mean(y)))
    if kind=='logistic':
        model=LogisticRegression(C=.1,max_iter=350,solver='lbfgs',random_state=seed)
    else:
        model=HistGradientBoostingClassifier(max_iter=90,max_leaf_nodes=15,learning_rate=.05,
            min_samples_leaf=100,l2_regularization=10.,random_state=seed)
    model.fit(x,y)
    return model.predict_proba(z)[:,list(model.classes_).index(1)]

def temporal_calibrate(p_cal,y_cal,p_val,eps=1e-6):
    """Platt logistic calibration using strictly earlier, disjoint calibration observations.
    Caller must pass probabilities from a model fitted before the calibration dates.
    """
    p_cal=np.clip(np.asarray(p_cal,float),eps,1-eps)
    p_val=np.clip(np.asarray(p_val,float),eps,1-eps)
    if np.unique(y_cal).size<2:return np.full(len(p_val),float(np.mean(y_cal)))
    transform=lambda p: np.log(p/(1-p)).reshape(-1,1)
    calibrator=LogisticRegression(C=10.,max_iter=300)
    calibrator.fit(transform(p_cal),y_cal)
    return calibrator.predict_proba(transform(p_val))[:,1]

def fit_fold(tr,va,xb,xv,extra_cols,cap,seed,fold):
    ax,az=prepare_extras(tr,va,extra_cols)
    X=np.column_stack((np.asarray(xb,dtype='float32'),ax))
    Z=np.column_stack((np.asarray(xv,dtype='float32'),az))
    # Cap is sampled ONLY from the training region, never from validation.
    rng=np.random.default_rng(seed+fold)
    ids=np.sort(rng.choice(len(X),size=min(cap,len(X)),replace=False))
    y=tr.next_return.to_numpy(float)[ids]
    lo,hi=np.quantile(y,[.005,.995]);yc=np.clip(y,lo,hi)
    scaler=StandardScaler().fit(X[ids]);xs=scaler.transform(X[ids]);zs=scaler.transform(Z)
    scores={'ridge_all':Ridge(alpha=100.).fit(xs,yc).predict(zs)}
    diagnostics=[];timings=[]
    for t in THRESHOLDS:
        for side in ('up','down'):
            binary=(y>t).astype(int) if side=='up' else (y< -t).astype(int)
            val_event=(va.next_return.to_numpy(float)>t).astype(int) if side=='up' else (va.next_return.to_numpy(float)< -t).astype(int)
            for kind in ('logistic','hgb'):
                start=time.perf_counter();pred=fit_binary(xs,binary,zs,kind,seed+fold)
                name=f'{side}_{int(t*100)}_{kind}'
                scores[name]=pred
                diagnostics.append(dict(fold=fold,model=name,side=side,threshold=t,train_prevalence=float(binary.mean()),validation_prevalence=float(val_event.mean()),auc=safe_auc(val_event,pred),average_precision=float(average_precision_score(val_event,pred)) if val_event.sum() else np.nan,brier=float(brier_score_loss(val_event,pred)),seconds=time.perf_counter()-start))
    for q in (.05,.10,.25,.50,.75,.90,.95):
        start=time.perf_counter()
        model=HistGradientBoostingRegressor(loss='quantile',quantile=q,max_iter=100,max_leaf_nodes=15,learning_rate=.05,min_samples_leaf=100,l2_regularization=10.,random_state=seed+fold)
        model.fit(X[ids],yc)
        scores[f'q{int(q*100):02d}']=model.predict(Z)
        timings.append(dict(fold=fold,model=f'q{int(q*100):02d}',seconds=time.perf_counter()-start))
    # Each ranking uses ONLY contemporaneously available model scores.
    p3=scores['up_3_hgb'];p5=scores['up_5_hgb'];p10=scores['up_10_hgb']
    d3=scores['down_3_hgb'];d5=scores['down_5_hgb'];d10=scores['down_10_hgb']
    scores['tail_direction_3']=p3-d3
    scores['tail_direction_5']=p5-d5
    scores['tail_direction_10']=p10-d10
    scores['up_down_ratio_5']=(p5+.001)/(d5+.001)
    scores['up_down_logodds_5']=np.log((p5+.001)/(d5+.001))
    scores['weighted_tail_utility']=.03*(p3-d3)+.05*(p5-d5)+.10*(p10-d10)
    scores['tail_utility_5_penalty2']=.05*p5-.10*d5
    scores['tail_utility_5_penalty05']=.05*p5-.025*d5
    scores['ridge_plus_tail']=scores['ridge_all']+.05*(p5-d5)
    scores['ridge_minus_down5']=scores['ridge_all']-.05*d5
    scores['ridge_per_down5']=scores['ridge_all']/(.02+d5)
    scores['q90_minus_q10_risk']=scores['q90']-np.maximum(-scores['q10'],0)
    scores['q95_minus_q05_risk']=scores['q95']-np.maximum(-scores['q05'],0)
    scores['q75_minus_q25_risk']=scores['q75']-np.maximum(-scores['q25'],0)
    scores['q90_downside_ratio']=np.maximum(scores['q90'],0)/(.01+np.maximum(-scores['q10'],0))
    scores['q95_downside_ratio']=np.maximum(scores['q95'],0)/(.01+np.maximum(-scores['q05'],0))
    scores['q90_minus_2down']=scores['q90']-2*np.maximum(-scores['q10'],0)
    scores['q90_minus_halfdown']=scores['q90']-.5*np.maximum(-scores['q10'],0)
    scores['ridge_over_iqr']=scores['ridge_all']/(.01+np.maximum(scores['q75']-scores['q25'],0))
    scores['upside_probability_x_ridge']=p5*np.maximum(scores['ridge_all'],0)
    # Diagnostics of directional asymmetry vs volatility (both-tail likelihood).
    scores['extreme_volatility_5']=p5+d5
    scores['extreme_direction_fraction_5']=(p5-d5)/(.01+p5+d5)
    v=va[['bar_timestamp','symbol','next_return']].reset_index(drop=True).copy();v['fold']=fold
    for name,s in scores.items():v[name]=np.asarray(s,float)
    return v,diagnostics,timings,dict(fold=fold,train_rows=len(tr),validation_rows=len(va),train_sample=len(ids),feature_count=X.shape[1],training_label_end=str(pd.to_datetime(tr.next_bar_timestamp,utc=True).max()),validation_start=str(pd.to_datetime(va.bar_timestamp,utc=True).min()),clip_low=float(lo),clip_high=float(hi))

def run_research(frame,features,cap=60000,seed=42,k_values=K_VALUES):
    scored=[];diagnostics=[];timings=[];audits=[]
    for f in expanding_folds(frame,n_folds=3):
        tr,va,xb,xv,_=fold_data(frame,f,target_q=7)
        assert pd.to_datetime(tr.next_bar_timestamp,utc=True).max()<=f.train_end
        assert pd.to_datetime(va.bar_timestamp,utc=True).min()>f.train_end
        v,d,t,a=fit_fold(tr,va,xb,xv,features,cap,seed,f.number)
        scored.append(v);diagnostics+=d;timings+=t;audits.append(a)
    scored=pd.concat(scored,ignore_index=True)
    models=[c for c in scored.columns if c not in ('bar_timestamp','symbol','next_return','fold')]
    metrics=[];daily=[];trades=[]
    for fold,g in scored.groupby('fold'):
        g=g.reset_index(drop=True);market=g.groupby('bar_timestamp').next_return.mean()
        for model in models:
            for k in k_values:
                selected=rank_with_exclusions(g,g[model].to_numpy(),k)
                if selected.empty:continue
                m,d=_metrics(selected,model,int(fold),int(k),market);metrics.append(m)
                selected['fold']=fold;selected['model']=model;selected['k']=k;trades.append(selected)
                daily.extend(dict(fold=fold,model=model,k=k,bar_timestamp=dt,daily_return=float(r)) for dt,r in d.items())
    return dict(scored=scored,events=pd.DataFrame(diagnostics),timings=pd.DataFrame(timings),audit=pd.DataFrame(audits),metrics=pd.DataFrame(metrics),daily=pd.DataFrame(daily),trades=pd.concat(trades,ignore_index=True))

def paired_comparison(daily,k=5,reps=1200,seed=42):
    rows=[]
    for fold,g in daily.loc[daily['k'].eq(k)].groupby('fold'):
        pivot=g.pivot(index='bar_timestamp',columns='model',values='daily_return')
        for model in pivot.columns:
            if model=='ridge_all':continue
            diff=(pivot[model]-pivot['ridge_all']).dropna()
            if len(diff)<20:continue
            low,high=date_block_ci(diff,block=10,reps=reps,seed=seed+int(fold))
            rows.append(dict(fold=fold,model=model,k=k,days=len(diff),mean_lift=float(diff.mean()),ci_low=low,ci_high=high))
    return pd.DataFrame(rows)

def event_reliability(scored,bins=10):
    rows=[]
    for fold,g in scored.groupby('fold'):
        for t in THRESHOLDS:
            for side in ('up','down'):
                y=(g.next_return.to_numpy()>t) if side=='up' else (g.next_return.to_numpy() < -t)
                for kind in ('logistic','hgb'):
                    name=f'{side}_{int(t*100)}_{kind}';p=np.clip(g[name].to_numpy(float),0,1)
                    ix=np.minimum((p*bins).astype(int),bins-1)
                    for b in range(bins):
                        m=ix==b
                        if m.any():rows.append(dict(fold=fold,model=name,bin=b,n=int(m.sum()),predicted=float(p[m].mean()),observed=float(y[m].mean())))
    return pd.DataFrame(rows)

def concentration(trades,k=5,fraction=.01):
    rows=[]
    for (fold,model),g in trades.loc[trades['k'].eq(k)].groupby(['fold','model']):
        n=max(1,int(np.ceil(len(g)*fraction)));total=g.next_return.sum()
        winners=g.next_return.nlargest(n)
        rows.append(dict(fold=fold,model=model,n=len(g),top_n=n,top_share=float(winners.sum()/total) if abs(total)>1e-12 else np.nan,without_top_mean=float(g.drop(winners.index).next_return.mean())))
    return pd.DataFrame(rows)

def save_tables(directory,**tables):
    path=Path(directory);path.mkdir(parents=True,exist_ok=True)
    for name,df in tables.items():df.to_csv(path/f'{name}.csv',index=False)
