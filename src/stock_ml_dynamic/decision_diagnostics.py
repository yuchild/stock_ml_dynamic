"""Notebook 27: development-only diagnostic research, never accesses the locked holdout.

Targets are fit on each training fold. Features use information at or before bar t.
Outer fold comparisons are descriptive; no post-hoc threshold is validated here.
"""
from __future__ import annotations
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, brier_score_loss
from .classifier_benchmark import eligible_universe
from .feature_ablation import development_holdout_split
from .historical_market import add_historical_market_features
from .accuracy_features import add_accuracy_features
from .volume_regime_research import add_volume_regime_features
from .walk_forward import expanding_folds, fold_data

SECTOR_CANDIDATES=('sector','sector_name','gics_sector','sector_code','sector_id')

def _fit_model(kind, seed):
    if kind.startswith('forest'):
        return RandomForestClassifier(n_estimators=140,max_depth=14,min_samples_leaf=15,
                                      n_jobs=-1,random_state=seed)
    return LogisticRegression(C=.1,max_iter=400)

def _aligned(model,x):
    raw=model.predict_proba(x); out=np.zeros((len(x),3))
    for i,c in enumerate(model.classes_):out[:,int(c)]=raw[:,i]
    return out

def _safe_auc(y,p):
    if len(np.unique(y))<3:return np.nan
    try:return float(roc_auc_score(y,p,labels=[0,1,2],multi_class='ovr',average='macro'))
    except ValueError:return np.nan

def _report(y,p,r,*,fold,model,segment='all',segment_value='all',cost_bps=10):
    pred=p.argmax(axis=1); chosen=(pred==1)&(p[:,1]>=.5)
    return dict(fold=fold,model=model,segment=segment,segment_value=str(segment_value),n=len(y),
                up_base=float(np.mean(y==1)),accuracy=float(accuracy_score(y,pred)),
                majority_baseline=float(np.bincount(y,minlength=3).max()/len(y)),
                balanced_accuracy=float(balanced_accuracy_score(y,pred)),auc=_safe_auc(y,p),
                entry_count=int(chosen.sum()),entry_coverage=float(chosen.mean()),
                entry_precision=float(np.mean(y[chosen]==1)) if chosen.any() else np.nan,
                entry_mean_net=float(np.mean(r[chosen])-cost_bps/10000) if chosen.any() else np.nan)

def _segment_rows(test, y, p, r, fold, model, cost_bps):
    out=[]
    for col in ('interval_code','sector_group','volatility_regime','breadth_regime','momentum_regime'):
        if col not in test:continue
        for value,idx in test.groupby(col,dropna=False).indices.items():
            if len(idx)<100:continue
            out.append(_report(y[idx],p[idx],r[idx],fold=fold,model=model,segment=col,
                               segment_value=value,cost_bps=cost_bps))
    return out

def _deciles(test,y,p,r,fold,model,cost_bps):
    # Equal-count confidence bins; no fitted thresholds on the test set.
    order=np.argsort(p[:,1],kind='stable'); buckets=np.empty(len(y),dtype=int)
    for i,ids in enumerate(np.array_split(order,10)):buckets[ids]=i+1
    rows=[]
    for i in range(1,11):
        idx=buckets==i
        rows.append(dict(fold=fold,model=model,decile=i,n=int(idx.sum()),
                         mean_p_up=float(p[idx,1].mean()),observed_up=float((y[idx]==1).mean()),
                         mean_next_return=float(r[idx].mean()),mean_net_long=float(r[idx].mean()-cost_bps/10000),
                         up_brier=float(brier_score_loss((y[idx]==1).astype(int),p[idx,1]))))
    return rows

def _economic_binary(train,test,xtr,xte,seed,cost_bps):
    """Cost-aware binary model: next return > cost; comparison uses different label base."""
    ytr=(train.next_return.to_numpy(dtype=float)>cost_bps/10000).astype(int)
    yte=(test.next_return.to_numpy(dtype=float)>cost_bps/10000).astype(int)
    model=LogisticRegression(C=.1,max_iter=350)
    model.fit(xtr,ytr);pp=model.predict_proba(xte)[:,list(model.classes_).index(1)]
    return dict(positive_base=float(yte.mean()),binary_accuracy=float(accuracy_score(yte,pp>=.5)),
                binary_majority_baseline=float(max(yte.mean(),1-yte.mean())),
                binary_auc=float(roc_auc_score(yte,pp)) if len(np.unique(yte))==2 else np.nan,
                selected_count=int((pp>=.55).sum()),selected_mean_net=float(
                    test.next_return.to_numpy(dtype=float)[pp>=.55].mean()-cost_bps/10000) if (pp>=.55).any() else np.nan)

def run_decision_diagnostics(frame,*,n_folds=3,max_train_rows=50000,tree_train_rows=35000,
                             cost_bps=10,seed=42,output_dir=None,include_volume=True):
    if n_folds!=3:raise ValueError('Use three chronological folds')
    eligible,_=eligible_universe(frame)
    dev,cutoff=development_holdout_split(eligible,.20)
    expanded,hist=add_historical_market_features(dev)
    expanded,technical=add_accuracy_features(expanded)
    volume=[]
    if include_volume:expanded,volume=add_volume_regime_features(expanded)
    cols=list(dict.fromkeys(hist+technical))
    extra=list(dict.fromkeys(cols+volume))
    sector=next((c for c in SECTOR_CANDIDATES if c in expanded),None)
    expanded['sector_group']=expanded[sector].fillna('Unknown').astype(str) if sector else 'Unavailable'
    # Regimes are causal and defined from current-bar features; thresholds use fixed 0.
    expanded['breadth_regime']=np.where(expanded['peer_market_return']>0,'advancing','declining')
    expanded['momentum_regime']=np.where(expanded['return_20bar']>0,'positive','negative')
    scores=[];segments=[];deciles=[];targets=[];confusions=[];ticker=[];timings=[]
    for fold in expanding_folds(expanded,n_folds=n_folds):
        train,test,xb,xv,_=fold_data(expanded,fold,target_q=7)
        oktr=train[cols].replace([np.inf,-np.inf],np.nan).notna().all(axis=1).to_numpy()
        okte=test[cols].replace([np.inf,-np.inf],np.nan).notna().all(axis=1).to_numpy()
        train=train.loc[oktr].copy();test=test.loc[okte].copy();xb=xb[oktr];xv=xv[okte]
        threshold=float(train.realized_vol_20.median())
        for d in (train,test):d['volatility_regime']=np.where(d.realized_vol_20>=threshold,'higher','lower')
        y=train.direction.to_numpy(dtype=int);yt=test.direction.to_numpy(dtype=int)
        r=test.next_return.to_numpy(dtype=float)
        if len(train)<300 or len(test)<100:raise ValueError('Insufficient fold observations')
        rng=np.random.default_rng(seed+fold.number)
        for feature_name,feature_cols in [('price',cols),('price_volume',extra)]:
            if feature_name=='price_volume' and not include_volume:continue
            prep=make_pipeline(SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True),StandardScaler())
            a=prep.fit_transform(train[feature_cols].replace([np.inf,-np.inf],np.nan))
            b=prep.transform(test[feature_cols].replace([np.inf,-np.inf],np.nan))
            xx=np.column_stack([xb,a]);xt=np.column_stack([xv,b])
            for kind in ('forest','logistic','logistic_ticker'):
                if kind=='logistic_ticker':
                    enc=OneHotEncoder(handle_unknown='ignore',sparse_output=False)
                    ta=enc.fit_transform(train[['symbol']]);tb=enc.transform(test[['symbol']])
                    xtrain=np.column_stack([xx,ta]);xtest=np.column_stack([xt,tb])
                else:xtrain,xtest=xx,xt
                cap=tree_train_rows if kind=='forest' else max_train_rows
                ix=np.sort(rng.choice(len(y),size=min(len(y),cap),replace=False))
                model=_fit_model(kind,seed+fold.number)
                t0=time.perf_counter();model.fit(xtrain[ix],y[ix]);elapsed=time.perf_counter()-t0
                p=_aligned(model,xtest);name=f'{kind}_{feature_name}'
                scores.append(_report(yt,p,r,fold=fold.number,model=name,cost_bps=cost_bps))
                segments.extend(_segment_rows(test,yt,p,r,fold.number,name,cost_bps))
                deciles.extend(_deciles(test,yt,p,r,fold.number,name,cost_bps))
                timings.append(dict(fold=fold.number,model=name,train_seconds=elapsed,train_rows=len(ix)))
                for truth in range(3):
                    for prediction in range(3):
                        confusions.append(dict(fold=fold.number,model=name,true_class=truth,pred_class=prediction,
                                               n=int(np.sum((yt==truth)&(p.argmax(axis=1)==prediction)))))
                if kind=='forest' and feature_name=='price':
                    for symbol,indices in test.groupby('symbol').indices.items():
                        if len(indices)>=30:
                            ticker.append(_report(yt[indices],p[indices],r[indices],fold=fold.number,
                                                  model=name,segment='symbol',segment_value=symbol,cost_bps=cost_bps))
                if kind=='logistic' and feature_name=='price':
                    target=_economic_binary(train.iloc[ix],test,xtrain[ix],xtest,seed,cost_bps)
                    targets.append(dict(fold=fold.number,model='economic_binary_logistic',**target))
        # Sector-specific training requires a genuine sector mapping; no sector is inferred.
        if sector:
            prep=make_pipeline(SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True),StandardScaler())
            aa=prep.fit_transform(train[cols].replace([np.inf,-np.inf],np.nan))
            bb=prep.transform(test[cols].replace([np.inf,-np.inf],np.nan))
            aa=np.column_stack([xb,aa]);bb=np.column_stack([xv,bb])
            for group in sorted(test.sector_group.unique()):
                tr=np.flatnonzero(train.sector_group.to_numpy()==group)
                te=np.flatnonzero(test.sector_group.to_numpy()==group)
                if len(tr)<500 or len(te)<100 or len(np.unique(y[tr]))<3:continue
                tr=np.sort(rng.choice(tr,min(len(tr),max_train_rows),replace=False))
                m=LogisticRegression(C=.1,max_iter=350).fit(aa[tr],y[tr])
                p=_aligned(m,bb[te]);scores.append(_report(yt[te],p,r[te],fold=fold.number,
                    model='sector_logistic',segment='sector_group',segment_value=group,cost_bps=cost_bps))
    outputs=dict(scores=pd.DataFrame(scores),segments=pd.DataFrame(segments),deciles=pd.DataFrame(deciles),
                 economic_targets=pd.DataFrame(targets),confusions=pd.DataFrame(confusions),
                 ticker_diagnostics=pd.DataFrame(ticker),timings=pd.DataFrame(timings))
    if output_dir is not None:
        path=Path(output_dir);path.mkdir(parents=True,exist_ok=True)
        for name,data in outputs.items():data.to_csv(path/f'{name}.csv',index=False)
    return outputs,cutoff,sector
