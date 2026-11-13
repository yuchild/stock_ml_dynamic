"""Chronological OOF stacking research. No closed holdout access or ledger writes."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from .sequential_deck_stacking import train_stage, assess
from .classifier_benchmark import eligible_universe
from .feature_ablation import development_holdout_split
from .historical_market import add_historical_market_features
from .accuracy_features import add_accuracy_features
from .volume_regime_research import add_volume_regime_features
from .walk_forward import expanding_folds, fold_data

CLASSES=(0,1,2)

def oof_windows(frame, n_windows=5, initial_fraction=.40):
    """Expanding OOF windows, partitioned on GLOBAL feature timestamps.

    Training labels must be available STRICTLY before the first validation bar.
    """
    ts=pd.to_datetime(frame.bar_timestamp,utc=True)
    outcome=pd.to_datetime(frame.next_bar_timestamp,utc=True)
    unique=np.sort(ts.dropna().unique())
    if len(unique)<n_windows*6: raise ValueError('Insufficient dates for OOF windows')
    bounds=np.linspace(int(len(unique)*initial_fraction),len(unique),n_windows+1,dtype=int)
    for i in range(n_windows):
        start=unique[bounds[i]]
        stop=unique[bounds[i+1]] if i<n_windows-1 else None
        train=np.flatnonzero((outcome<start).to_numpy())
        valid=np.flatnonzero(((ts>=start)&((ts<stop) if stop is not None else True)).to_numpy())
        if len(train)<30 or not len(valid): raise ValueError('Empty or too-small OOF partition')
        assert outcome.iloc[train].max()<ts.iloc[valid].min()
        yield train,valid

def make_oof(train,cols,*,windows=5,cap=60000,trees=200,leaf=20,seed=42,clip=(.005,.995)):
    """OOF predictions on later dates only; early warmup has no OOF predictions."""
    oof=np.full((len(train),3),np.nan)
    audit=[]
    for k,(a,b) in enumerate(oof_windows(train,windows),1):
        model=train_stage(train.iloc[a],cols,cap=cap,trees=trees,leaf=leaf,seed=seed+k,clip=clip)
        oof[b]=model.predict(train.iloc[b])
        audit.append(dict(window=k,fit_rows=len(a),pred_rows=len(b),
                          latest_train_outcome=str(pd.to_datetime(train.iloc[a].next_bar_timestamp,utc=True).max()),
                          earliest_pred_bar=str(pd.to_datetime(train.iloc[b].bar_timestamp,utc=True).min())))
    return oof,pd.DataFrame(audit)

def meta_features(p):
    """Use only probability columns; optional feature augmentation is separately tested."""
    return pd.DataFrame(p,columns=['oof_p0','oof_p1','oof_p2'])

def fit_meta(oof,y,kind,seed=42):
    good=np.isfinite(oof).all(axis=1)
    if good.sum()<100: raise ValueError('Insufficient OOF predictions')
    x=meta_features(oof).iloc[np.flatnonzero(good)]
    labels=np.asarray(y)[good]
    if kind=='logistic':
        model=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),
                            LogisticRegression(C=.1,max_iter=500,random_state=seed))
    elif kind=='rf':
        model=RandomForestClassifier(n_estimators=200,max_depth=4,min_samples_leaf=100,
                                     n_jobs=-1,random_state=seed)
    else:raise ValueError('Unknown meta model')
    model.fit(x,labels)
    return model

def aligned_probs(model,x):
    raw=model.predict_proba(x)
    out=np.zeros((len(x),3))
    for j,c in enumerate(model.classes_):out[:,int(c)]=raw[:,j]
    return out

def run_fold(train,valid,cols,*,fold=1,windows=5,cap=60000,trees=200,leaf=20,
             seed=42,clip=(.005,.995),threshold=.45,cost_bps=10,return_predictions=False):
    """Compare RF full-history, LR/RF OOF meta, and RF on OOF-augmented rows.

    Meta training uses only later training-period OOF rows. Final base model
    is fitted on the entire outer training period, never on outer validation.
    """
    oof,audit=make_oof(train,cols,windows=windows,cap=cap,trees=trees,leaf=leaf,
                       seed=seed+fold*100,clip=clip)
    base=train_stage(train,cols,cap=cap,trees=trees,leaf=leaf,seed=seed+fold*100+50,clip=clip)
    pv=base.predict(valid)
    predictions={'full_history_rf':pv}
    rows=[assess(valid,pv,'full_history_rf',fold,cost_bps,threshold)]
    good=np.isfinite(oof).all(axis=1)
    for kind in ('logistic','rf'):
        meta=fit_meta(oof,train.direction.to_numpy(),kind,seed+fold)
        pm=aligned_probs(meta,meta_features(pv))
        predictions[f'oof_meta_{kind}']=pm
        rows.append(assess(valid,pm,f'oof_meta_{kind}',fold,cost_bps,threshold))
    # A shallow RF on OOF predictions + original features, trained ONLY on OOF rows.
    # Unlike the full-history baseline, its training subset is smaller: report this.
    extra=['oof_p0','oof_p1','oof_p2']
    oof_df=train.iloc[np.flatnonzero(good)].copy()
    for i,c in enumerate(extra):oof_df[c]=oof[good,i]
    val_df=valid.copy()
    for i,c in enumerate(extra):val_df[c]=pv[:,i]
    augmented=train_stage(oof_df,list(cols)+extra,cap=cap,trees=trees,leaf=leaf,
                          seed=seed+fold*100+51,clip=clip)
    augmented_p=augmented.predict(val_df)
    predictions['oof_augmented_rf']=augmented_p
    rows.append(assess(valid,augmented_p,'oof_augmented_rf',fold,cost_bps,threshold))
    for row in rows:row.update(oof_rows=int(good.sum()),outer_train_rows=len(train),
                               outer_valid_rows=len(valid),feature_count=len(cols))
    audit.insert(0,'outer_fold',fold)
    if return_predictions:return pd.DataFrame(rows),audit,predictions
    return pd.DataFrame(rows),audit

def run_research(frame,*,n_outer=3,windows=5,cap=60000,trees=200,leaf=20,
                 seed=42,clip=(.005,.995),threshold=.45,cost_bps=10,output_dir=None):
    eligible,_=eligible_universe(frame)
    dev,cutoff=development_holdout_split(eligible,.2)
    expanded,hist=add_historical_market_features(dev)
    expanded,tech=add_accuracy_features(expanded)
    expanded,vol=add_volume_regime_features(expanded)
    cols=list(dict.fromkeys(hist+tech+vol))
    reports=[];audits=[]
    for fold in expanding_folds(expanded,n_folds=n_outer):
        tr,va,_,_,_=fold_data(expanded,fold,target_q=7)
        report,audit=run_fold(tr,va,cols,fold=fold.number,windows=windows,cap=cap,
                              trees=trees,leaf=leaf,seed=seed,clip=clip,
                              threshold=threshold,cost_bps=cost_bps)
        reports.append(report);audits.append(audit)
    report=pd.concat(reports,ignore_index=True)
    audit=pd.concat(audits,ignore_index=True)
    if output_dir is not None:
        dest=Path(output_dir);dest.mkdir(parents=True,exist_ok=True)
        report.to_csv(dest/'oof_stacking_comparison.csv',index=False)
        audit.to_csv(dest/'oof_stacking_audit.csv',index=False)
    return {'outer':report,'audit':audit,'cutoff':cutoff,'feature_count':len(cols)}
