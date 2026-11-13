"""Chronological four-deck, cross-deck Random Forest stacking (research only).

Decks are GLOBAL non-overlapping time periods, not randomly shuffled rows.
Stage k is fitted on deck k and generates out-of-sample probabilities for later
stages. Every upstream model has its own training-fitted preprocessing.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, brier_score_loss
from .outlier_eda import fit_outlier_bounds, apply_outlier_bounds
from .classifier_benchmark import eligible_universe
from .feature_ablation import development_holdout_split
from .historical_market import add_historical_market_features
from .accuracy_features import add_accuracy_features
from .volume_regime_research import add_volume_regime_features
from .walk_forward import expanding_folds, fold_data
from .candle_table import DirectionQuantiles

LABELS=(0,1,2)


def chronological_decks(frame, n=4):
    """Partition by unique outcome timestamps, never separating a market date."""
    t=pd.to_datetime(frame['next_bar_timestamp'],utc=True)
    unique=np.sort(t.dropna().unique())
    if len(unique)<n*12: raise ValueError('Insufficient unique outcome dates for four decks')
    chunks=np.array_split(unique,n)
    decks=[frame.loc[t.isin(chunk)].copy() for chunk in chunks]
    if any(len(d)==0 for d in decks): raise ValueError('Empty deck')
    for left,right in zip(decks,decks[1:]):
        if pd.to_datetime(left.next_bar_timestamp,utc=True).max()>=pd.to_datetime(right.bar_timestamp,utc=True).min():
            # Adjacent completed bars may share an outcome/feature timestamp; this is
            # safe for end-of-day sequential fitting but never use overlapping labels.
            if pd.to_datetime(left.next_bar_timestamp,utc=True).max()>pd.to_datetime(right.bar_timestamp,utc=True).min():
                raise ValueError('Deck time order overlaps')
    return decks


@dataclass
class RFStage:
    columns:list[str]
    imputer:SimpleImputer
    model:RandomForestClassifier
    bounds:pd.DataFrame|None

    def predict(self,frame):
        x=frame.reindex(columns=self.columns).replace([np.inf,-np.inf],np.nan).copy()
        if self.bounds is not None:x,_=apply_outlier_bounds(x,self.bounds)
        p=self.model.predict_proba(self.imputer.transform(x))
        result=np.zeros((len(frame),3),dtype=float)
        for i,label in enumerate(self.model.classes_):result[:,int(label)]=p[:,i]
        return result


def train_stage(frame,cols,*,clip=(.005,.995),cap=60000,trees=200,leaf=8,seed=42):
    if len(frame)<30:raise ValueError('Too few training rows')
    rng=np.random.default_rng(seed)
    indices=np.sort(rng.choice(len(frame),size=min(cap,len(frame)),replace=False))
    sample=frame.iloc[indices]
    x=sample.reindex(columns=cols).replace([np.inf,-np.inf],np.nan).copy()
    bounds=None
    if clip is not None:
        bounds=fit_outlier_bounds(x,cols,np.ones(len(x),dtype=bool),lower=clip[0],upper=clip[1],min_finite=30)
        x,_=apply_outlier_bounds(x,bounds)
    imp=SimpleImputer(strategy='median',keep_empty_features=True)
    x=imp.fit_transform(x)
    model=RandomForestClassifier(n_estimators=trees,min_samples_leaf=leaf,max_depth=12,
                                 max_features='sqrt',n_jobs=-1,random_state=seed)
    model.fit(x,sample.direction.to_numpy(dtype=int))
    return RFStage(list(cols),imp,model,bounds)


def augment(frame,models):
    result=frame.copy()
    for k,model in enumerate(models,1):
        p=model.predict(result)
        for c in LABELS:result[f'model{k}_p{c}']=p[:,c]
        result[f'model{k}_pred']=p.argmax(axis=1)
    return result


def assess(frame,proba,model_name,fold,cost_bps=10,threshold=.45):
    y=frame.direction.to_numpy(dtype=int);ret=frame.next_return.to_numpy(dtype=float)
    pred=proba.argmax(axis=1)
    mask=(pred==1)&(proba[:,1]>=threshold)
    net=ret[mask]-cost_bps/10000
    try:auc=roc_auc_score(y,proba,labels=[0,1,2],multi_class='ovr',average='macro')
    except ValueError:auc=np.nan
    return dict(fold=fold,model=model_name,n=len(y),accuracy=accuracy_score(y,pred),
                majority_baseline=np.bincount(y,minlength=3).max()/len(y),auc=auc,
                logloss=log_loss(y,proba,labels=[0,1,2]),
                brier_up=brier_score_loss(y==1,proba[:,1]),
                coverage=mask.mean(),entry_count=int(mask.sum()),
                up_precision=np.mean(y[mask]==1) if mask.any() else np.nan,
                mean_net=net.mean() if mask.any() else np.nan,
                median_net=np.median(net) if mask.any() else np.nan)


def one_fold(train,valid,base_cols,*,fold=1,clip=(.005,.995),cap=60000,trees=200,
             leaf=8,seed=42,cost_bps=10,threshold=.45):
    """Fit deck1→deck2→deck3→deck4; compare stage4 to two fair baselines."""
    decks=chronological_decks(train)
    # Fix target boundaries using only the earliest deck. No later deck outcome
    # influences the definition of early labels.
    q=DirectionQuantiles(q=7).fit(decks[0].next_return)
    decks=[d.assign(direction=q.transform(d.next_return).astype(int)) for d in decks]
    valid=valid.assign(direction=q.transform(valid.next_return).astype(int))
    stages=[];audit=[]
    start=time.perf_counter()
    for k,d in enumerate(decks,1):
        enhanced=augment(d,stages)
        extra=[f'model{i}_p{c}' for i in range(1,k) for c in LABELS]
        cols=list(base_cols)+extra
        model=train_stage(enhanced,cols,clip=clip,cap=cap,trees=trees,leaf=leaf,seed=seed+fold*10+k)
        stages.append(model)
        audit.append(dict(fold=fold,stage=k,train_rows=len(d),train_start=str(d.bar_timestamp.min()),
                          train_label_end=str(d.next_bar_timestamp.max()),
                          features=len(cols),up_rate=float(np.mean(d.direction==1))))
    evaluation=augment(valid,stages[:3]);p=stages[3].predict(evaluation)
    rows=[assess(valid,p,'sequential_four_deck',fold,cost_bps,threshold)]
    baseline4=train_stage(decks[3],list(base_cols),clip=clip,cap=cap,trees=trees,leaf=leaf,seed=seed+fold*10+5)
    rows.append(assess(valid,baseline4.predict(valid),'deck4_only_rf',fold,cost_bps,threshold))
    all_train=pd.concat(decks,ignore_index=True)
    baselineall=train_stage(all_train,list(base_cols),clip=clip,cap=cap,trees=trees,leaf=leaf,seed=seed+fold*10+6)
    rows.append(assess(valid,baselineall.predict(valid),'all_training_rf',fold,cost_bps,threshold))
    for r in rows:r['seconds_total']=time.perf_counter()-start
    return pd.DataFrame(rows),pd.DataFrame(audit),stages


def run_deck_research(frame,*,n_outer=3,cap=60000,trees=200,leaf=8,seed=42,
                      clip=(.005,.995),threshold=.45,cost_bps=10,output_dir=None):
    eligible,_=eligible_universe(frame)
    dev,cutoff=development_holdout_split(eligible,.2)
    expanded,hist=add_historical_market_features(dev)
    expanded,tech=add_accuracy_features(expanded)
    expanded,vol=add_volume_regime_features(expanded)
    cols=list(dict.fromkeys(hist+tech+vol))
    reports=[];audits=[]
    for outer in expanding_folds(expanded,n_folds=n_outer):
        tr,va,_,_,_=fold_data(expanded,outer,target_q=7)
        result,audit,_=one_fold(tr,va,cols,fold=outer.number,clip=clip,cap=cap,trees=trees,
                                 leaf=leaf,seed=seed,cost_bps=cost_bps,threshold=threshold)
        reports.append(result);audits.append(audit)
    result=pd.concat(reports,ignore_index=True);audit=pd.concat(audits,ignore_index=True)
    if output_dir is not None:
        p=Path(output_dir);p.mkdir(parents=True,exist_ok=True)
        result.to_csv(p/'deck_model_comparison.csv',index=False)
        audit.to_csv(p/'deck_training_audit.csv',index=False)
    return {'outer':result,'audit':audit,'cutoff':cutoff,'feature_count':len(cols)}
