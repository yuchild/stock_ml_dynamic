"""Notebook 29: leakage-aware Random Forest research on development data only."""
from __future__ import annotations
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, brier_score_loss
from sklearn.impute import SimpleImputer
from .classifier_benchmark import eligible_universe
from .feature_ablation import development_holdout_split
from .historical_market import add_historical_market_features
from .accuracy_features import add_accuracy_features
from .volume_regime_research import add_volume_regime_features
from .walk_forward import expanding_folds, fold_data

THRESHOLDS=(.40,.425,.45,.475,.50,.525,.55,.575,.60)
COSTS=(0,5,10,20,35)

def candidate_grid():
    """36 RF settings: 3 leaves × 3 depths × 2 max_features × 2 weights."""
    return [dict(name=f'L{leaf}_D{depth or "none"}_F{feat}_W{weight or "none"}',
                 min_samples_leaf=leaf,max_depth=depth,max_features=feat,
                 class_weight=weight)
            for leaf in (8,20,40) for depth in (12,20,None)
            for feat in ('sqrt',.6) for weight in (None,'balanced_subsample')]

def training_indices(frame,cap,seed,mode):
    if len(frame)<=cap:return np.arange(len(frame))
    if mode=='recent':return np.arange(len(frame)-cap,len(frame))
    if mode=='uniform':return np.sort(np.random.default_rng(seed).choice(len(frame),cap,replace=False))
    raise ValueError('sampling mode must be recent or uniform')

def probability_matrix(model,x):
    p=np.zeros((len(x),3),dtype=float)
    for j,c in enumerate(model.classes_):p[:,int(c)]=model.predict_proba(x)[:,j]
    return p

def metrics(y,p):
    y=np.asarray(y,dtype=int)
    try:auc=float(roc_auc_score(y,p,labels=[0,1,2],multi_class='ovr',average='macro'))
    except ValueError:auc=np.nan
    return dict(accuracy=float(accuracy_score(y,p.argmax(axis=1))),auc=auc,
                logloss=float(log_loss(y,p,labels=[0,1,2])),
                brier_up=float(brier_score_loss((y==1).astype(int),p[:,1])))

def entry_metrics(y,p,r,threshold,cost_bps=10):
    mask=(p.argmax(axis=1)==1)&(p[:,1]>=threshold)
    n=int(mask.sum());net=np.asarray(r)[mask]-cost_bps/10000
    return dict(entry_count=n,coverage=n/len(y),precision=float(np.mean(np.asarray(y)[mask]==1)) if n else np.nan,
                mean_net=float(net.mean()) if n else np.nan,median_net=float(np.median(net)) if n else np.nan,
                hit_rate=float(np.mean(net>0)) if n else np.nan)

def choose(rows,objective,min_count=150,min_coverage=.025):
    df=pd.DataFrame(rows)
    group=df.groupby(['config','features','sampling','threshold'],as_index=False).agg(
        accuracy=('accuracy','mean'),auc=('auc','mean'),logloss=('logloss','mean'),
        precision=('precision','mean'),mean_net=('mean_net','mean'),
        min_coverage=('coverage','min'),min_entries=('entry_count','min'),
        folds=('inner_fold','nunique'),worst_net=('mean_net','min'))
    if objective=='accuracy':
        return group.sort_values(['accuracy','auc'],ascending=[False,False]).iloc[0].to_dict(),group
    valid=group[(group.folds>=2)&(group.min_entries>=min_count)&(group.min_coverage>=min_coverage)].copy()
    if valid.empty:return None,group
    # Robustness-first selection: do not maximize rare-signal precision alone.
    valid['score']=valid['mean_net'] + .25*valid['worst_net']
    return valid.sort_values(['score','precision','min_coverage'],ascending=False).iloc[0].to_dict(),group

def fit_one(tr,te,xb,xv,cols,cfg,cap,seed,n_estimators,sampling):
    imp=SimpleImputer(strategy='median',keep_empty_features=True)
    a=imp.fit_transform(tr[cols].replace([np.inf,-np.inf],np.nan))
    b=imp.transform(te[cols].replace([np.inf,-np.inf],np.nan))
    x=np.column_stack([xb,a]);z=np.column_stack([xv,b])
    ix=training_indices(tr,cap,seed,sampling)
    model=RandomForestClassifier(n_estimators=n_estimators,n_jobs=-1,random_state=seed,
                                 min_samples_leaf=cfg['min_samples_leaf'],max_depth=cfg['max_depth'],
                                 max_features=cfg['max_features'],class_weight=cfg['class_weight'])
    model.fit(x[ix],tr.direction.to_numpy(dtype=int)[ix])
    return probability_matrix(model,z),model,xb.shape[1]

def run_rf_deep_research(frame,*,n_outer=3,n_inner=2,cap=60000,n_estimators=200,
                         seed=42,cost_bps=10,min_count=150,min_coverage=.025,
                         sampling_modes=('uniform','recent'),configs=None,output_dir=None):
    if n_outer!=3 or n_inner!=2:raise ValueError('Three outer and two inner folds required')
    if cap<100:raise ValueError('cap must be >= 100')
    configs=list(candidate_grid() if configs is None else configs)
    eligible,_=eligible_universe(frame)
    development,cutoff=development_holdout_split(eligible,.2)
    expanded,hist=add_historical_market_features(development)
    expanded,tech=add_accuracy_features(expanded)
    expanded,vol=add_volume_regime_features(expanded)
    feature_sets={'price':list(dict.fromkeys(hist+tech)),
                  'price_volume':list(dict.fromkeys(hist+tech+vol))}
    inner_rows=[];outer_rows=[];selection_rows=[];trade_rows=[];calibration_rows=[];importance_rows=[];runtime_rows=[]
    for outer in expanding_folds(expanded,n_folds=n_outer):
        # The last observable outcome in inner training must precede the outer training end.
        inner_frame=expanded[pd.to_datetime(expanded.next_bar_timestamp,utc=True)<=outer.train_end].copy()
        local=[]
        for inner in expanding_folds(inner_frame,n_folds=n_inner):
            tr,va,xb,xv,_=fold_data(inner_frame,inner,target_q=7)
            y=va.direction.to_numpy(dtype=int);r=va.next_return.to_numpy(dtype=float)
            for fname,cols in feature_sets.items():
                for sampling in sampling_modes:
                    for cfg in configs:
                        start=time.perf_counter()
                        p,_,_=fit_one(tr,va,xb,xv,cols,cfg,cap,seed+outer.number*100+inner.number,
                                      n_estimators,sampling)
                        elapsed=time.perf_counter()-start
                        common=dict(outer_fold=outer.number,inner_fold=inner.number,config=cfg['name'],
                                    features=fname,sampling=sampling,train_rows=min(cap,len(tr)),
                                    validation_rows=len(va),fit_seconds=elapsed,**metrics(y,p))
                        for threshold in THRESHOLDS:
                            local.append(dict(**common,threshold=threshold,**entry_metrics(y,p,r,threshold,cost_bps)))
        inner_rows.extend(local)
        tr,te,xb,xv,_=fold_data(expanded,outer,target_q=7)
        y=te.direction.to_numpy(dtype=int);r=te.next_return.to_numpy(dtype=float)
        for objective in ('accuracy','entry'):
            choice,ranked=choose(local,objective,min_count,min_coverage)
            if choice is None:
                selection_rows.append(dict(fold=outer.number,objective=objective,status='no_eligible_inner_candidate'))
                continue
            cfg=next(c for c in configs if c['name']==choice['config'])
            cols=feature_sets[choice['features']]
            start=time.perf_counter()
            p,model,base_width=fit_one(tr,te,xb,xv,cols,cfg,cap,seed+outer.number,
                                        n_estimators,choice['sampling'])
            elapsed=time.perf_counter()-start
            threshold=float(choice['threshold']);signal=(p.argmax(axis=1)==1)&(p[:,1]>=threshold)
            outer_rows.append(dict(fold=outer.number,objective=objective,config=cfg['name'],
                                   features=choice['features'],sampling=choice['sampling'],threshold=threshold,
                                   n=len(y),majority_baseline=float(np.bincount(y,minlength=3).max()/len(y)),
                                   up_baseline=float(np.mean(y==1)),**metrics(y,p),
                                   **entry_metrics(y,p,r,threshold,cost_bps)))
            selection_rows.append(dict(fold=outer.number,objective=objective,status='selected',**choice))
            runtime_rows.append(dict(fold=outer.number,objective=objective,fit_seconds=elapsed,train_rows=min(cap,len(tr))))
            for col,imp in zip(cols,model.feature_importances_[base_width:]):
                importance_rows.append(dict(fold=outer.number,objective=objective,feature=col,importance=float(imp)))
            for low,high in zip(np.linspace(0,1,11)[:-1],np.linspace(0,1,11)[1:]):
                mask=(p[:,1]>=low)&((p[:,1]<high) if high<1 else (p[:,1]<=high))
                if mask.sum():calibration_rows.append(dict(fold=outer.number,objective=objective,bin_low=low,bin_high=high,
                    n=int(mask.sum()),mean_probability=float(p[mask,1].mean()),actual_up=float(np.mean(y[mask]==1)),
                    mean_return=float(np.mean(r[mask]))))
            # Preserve all signals at the chosen threshold for cost and tail-risk analyses.
            for j in np.flatnonzero(signal):
                trade_rows.append(dict(fold=outer.number,objective=objective,symbol=str(te.iloc[j].symbol),
                    bar_timestamp=str(te.iloc[j].bar_timestamp),prob_up=float(p[j,1]),
                    actual_up=int(y[j]==1),gross_return=float(r[j]),net_return=float(r[j]-cost_bps/10000)))
    result={key:pd.DataFrame(rows) for key,rows in dict(inner=inner_rows,outer=outer_rows,
        selections=selection_rows,trades=trade_rows,calibration=calibration_rows,
        importances=importance_rows,timings=runtime_rows).items()}
    if output_dir is not None:
        path=Path(output_dir);path.mkdir(parents=True,exist_ok=True)
        for key,table in result.items():table.to_csv(path/f'{key}.csv',index=False)
    return result,cutoff

def cost_sensitivity(trades,costs=COSTS):
    if trades.empty:return pd.DataFrame()
    return pd.DataFrame([dict(objective=obj,fold=fold,cost_bps=cost,
          n=len(group),mean_net=float((group.gross_return-cost/10000).mean()),
          win_rate=float(((group.gross_return-cost/10000)>0).mean()))
        for (obj,fold),group in trades.groupby(['objective','fold']) for cost in costs])

def daily_aggregated_returns(trades):
    """Equal-weight concurrent selected signals by date, avoiding per-ticker pseudo-independence."""
    if trades.empty:return pd.DataFrame()
    frame=trades.copy();frame['date']=pd.to_datetime(frame.bar_timestamp,utc=True).dt.date
    return frame.groupby(['objective','fold','date'],as_index=False).agg(
        trades=('net_return','size'),mean_net=('net_return','mean'),gross=('gross_return','mean'))
