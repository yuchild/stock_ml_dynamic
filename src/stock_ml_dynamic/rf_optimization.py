"""Notebook 28: nested chronological Random Forest optimization, development only."""
from __future__ import annotations
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.impute import SimpleImputer
from .classifier_benchmark import eligible_universe
from .feature_ablation import development_holdout_split
from .historical_market import add_historical_market_features
from .accuracy_features import add_accuracy_features
from .volume_regime_research import add_volume_regime_features
from .walk_forward import expanding_folds, fold_data

CONFIGS = (
    dict(name='rf_leaf8_unweighted', min_samples_leaf=8, max_depth=16, class_weight=None),
    dict(name='rf_leaf15_unweighted', min_samples_leaf=15, max_depth=14, class_weight=None),
    dict(name='rf_leaf30_unweighted', min_samples_leaf=30, max_depth=12, class_weight=None),
    dict(name='rf_leaf15_balanced', min_samples_leaf=15, max_depth=14, class_weight='balanced_subsample'),
    dict(name='rf_leaf15_deep', min_samples_leaf=15, max_depth=None, class_weight=None),
)
THRESHOLDS = (.40,.45,.50,.55,.60)

def sample_indices(n, cap, seed):
    if cap < 100: raise ValueError('Training cap must be >=100')
    if n <= cap: return np.arange(n)
    return np.sort(np.random.default_rng(seed).choice(n,cap,replace=False))

def score_entries(y, p, returns, threshold, cost_bps=10):
    """Long-only signal when class 1 is argmax and clears the probability threshold."""
    selected=(p.argmax(axis=1)==1)&(p[:,1]>=threshold)
    count=int(selected.sum()); coverage=count/len(y)
    net=np.asarray(returns)[selected]-cost_bps/10000
    return dict(entry_count=count,coverage=coverage,precision=float(np.mean(np.asarray(y)[selected]==1)) if count else np.nan,
                mean_net=float(net.mean()) if count else np.nan,median_net=float(np.median(net)) if count else np.nan,
                win_rate=float(np.mean(net>0)) if count else np.nan)

def prepare_features(train,test,cols):
    # Only the training fold determines imputation; all columns are causal.
    a=train[cols].replace([np.inf,-np.inf],np.nan)
    b=test[cols].replace([np.inf,-np.inf],np.nan)
    imp=SimpleImputer(strategy='median',keep_empty_features=True)
    return imp.fit_transform(a),imp.transform(b)

def fit_predict(train,test,base_train,base_test,cols,cfg,cap,seed,n_estimators):
    a,b=prepare_features(train,test,cols)
    x=np.column_stack([base_train,a]); z=np.column_stack([base_test,b])
    ix=sample_indices(len(train),cap,seed)
    model=RandomForestClassifier(n_estimators=n_estimators,n_jobs=-1,random_state=seed,
                                  max_features='sqrt',**{k:v for k,v in cfg.items() if k!='name'})
    model.fit(x[ix],train.direction.to_numpy(dtype=int)[ix])
    p=np.zeros((len(test),3))
    probabilities=model.predict_proba(z)
    for j,c in enumerate(model.classes_):p[:,int(c)]=probabilities[:,j]
    return p,model,base_train.shape[1]

def safe_auc(y,p):
    try:return float(roc_auc_score(y,p,labels=[0,1,2],multi_class='ovr',average='macro'))
    except ValueError:return np.nan

def choose_candidate(inner_results,min_entries=100,min_coverage=.02,objective='accuracy'):
    """Selection depends only on inner folds, never on outer outcomes."""
    df=pd.DataFrame(inner_results)
    if df.empty:raise ValueError('No inner-fold results')
    agg=df.groupby(['config','features','threshold'],as_index=False).agg(
        accuracy=('accuracy','mean'),auc=('auc','mean'),precision=('precision','mean'),
        coverage=('coverage','mean'),mean_net=('mean_net','mean'),
        min_entries=('entry_count','min'),folds=('fold','nunique'))
    valid=agg[(agg.min_entries>=min_entries)&(agg.coverage>=min_coverage)&(agg.folds>=2)]
    if objective=='entry' and len(valid):
        # Precision first; return is diagnostic, not a guaranteed edge.
        chosen=valid.sort_values(['precision','mean_net','coverage'],ascending=False).iloc[0]
    else:
        chosen=agg.sort_values(['accuracy','auc'],ascending=False).iloc[0]
    return chosen.to_dict(),agg

def run_rf_optimization(frame,*,n_outer=3,n_inner=2,train_cap=35000,n_estimators=120,
                        cost_bps=10,min_entries=100,min_coverage=.02,seed=42,output_dir=None):
    if n_outer!=3 or n_inner!=2:raise ValueError('Use 3 outer and 2 inner chronological folds')
    eligible,_=eligible_universe(frame)
    development,cutoff=development_holdout_split(eligible,.2)
    expanded,hist=add_historical_market_features(development)
    expanded,technical=add_accuracy_features(expanded)
    expanded,volume=add_volume_regime_features(expanded)
    feature_sets={'price':list(dict.fromkeys(hist+technical)),
                  'price_volume':list(dict.fromkeys(hist+technical+volume))}
    inner_rows=[]; outer_rows=[]; selections=[]; predictions=[]; importances=[]; timings=[]
    for outer in expanding_folds(expanded,n_folds=n_outer):
        # Restrict inner-fold data to outcomes observable by outer training end.
        inner_frame=expanded[pd.to_datetime(expanded.next_bar_timestamp,utc=True)<=outer.train_end].copy()
        if inner_frame.empty:raise ValueError('Empty outer training history')
        local=[]
        for inner in expanding_folds(inner_frame,n_folds=n_inner):
            tr,va,xb,xv,_=fold_data(inner_frame,inner,target_q=7)
            y=va.direction.to_numpy(dtype=int); returns=va.next_return.to_numpy(dtype=float)
            for features,cols in feature_sets.items():
                for cfg in CONFIGS:
                    start=time.perf_counter()
                    p,_,_=fit_predict(tr,va,xb,xv,cols,cfg,train_cap,seed+outer.number*100+inner.number,n_estimators)
                    elapsed=time.perf_counter()-start
                    common=dict(outer_fold=outer.number,fold=inner.number,config=cfg['name'],features=features,
                                accuracy=float(accuracy_score(y,p.argmax(axis=1))),auc=safe_auc(y,p),
                                train_rows=min(len(tr),train_cap),validation_rows=len(va),fit_seconds=elapsed)
                    for threshold in THRESHOLDS:
                        local.append(dict(**common,threshold=threshold,**score_entries(y,p,returns,threshold,cost_bps)))
        inner_rows.extend(local)
        # Two distinct objectives; outer data cannot influence configuration or threshold selection.
        for objective in ('accuracy','entry'):
            choice,ranked=choose_candidate(local,min_entries,min_coverage,objective)
            cfg=next(c for c in CONFIGS if c['name']==choice['config'])
            tr,te,xb,xv,_=fold_data(expanded,outer,target_q=7)
            start=time.perf_counter()
            p,model,base_width=fit_predict(tr,te,xb,xv,feature_sets[choice['features']],cfg,
                                           train_cap,seed+outer.number,n_estimators)
            elapsed=time.perf_counter()-start
            y=te.direction.to_numpy(dtype=int); r=te.next_return.to_numpy(dtype=float)
            entry=score_entries(y,p,r,float(choice['threshold']),cost_bps)
            outer_rows.append(dict(fold=outer.number,objective=objective,config=cfg['name'],features=choice['features'],
                                   threshold=float(choice['threshold']),n=len(te),accuracy=float(accuracy_score(y,p.argmax(axis=1))),
                                   auc=safe_auc(y,p),majority_baseline=float(np.bincount(y,minlength=3).max()/len(y)),
                                   up_base=float(np.mean(y==1)),**entry))
            selections.append(dict(fold=outer.number,objective=objective,**choice))
            timings.append(dict(fold=outer.number,objective=objective,fit_seconds=elapsed,train_rows=min(len(tr),train_cap)))
            for i,importance in enumerate(model.feature_importances_[base_width:]):
                importances.append(dict(fold=outer.number,objective=objective,feature=feature_sets[choice['features']][i],importance=float(importance)))
            selected=(p.argmax(axis=1)==1)&(p[:,1]>=float(choice['threshold']))
            # Aggregate trade-level diagnostics with identifiers for offline inspection.
            for j in np.flatnonzero(selected):
                predictions.append(dict(fold=outer.number,objective=objective,symbol=str(te.iloc[j].symbol),
                                        bar_timestamp=str(te.iloc[j].bar_timestamp),prob_up=float(p[j,1]),
                                        actual_up=bool(y[j]==1),next_return=float(r[j]),net_return=float(r[j]-cost_bps/10000)))
    result=dict(inner=pd.DataFrame(inner_rows),outer=pd.DataFrame(outer_rows),
                selections=pd.DataFrame(selections),trades=pd.DataFrame(predictions),
                importances=pd.DataFrame(importances),timings=pd.DataFrame(timings))
    if output_dir is not None:
        out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
        for key,df in result.items():df.to_csv(out/f'{key}.csv',index=False)
    return result,cutoff
