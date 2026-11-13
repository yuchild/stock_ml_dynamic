"""Notebook 31: nested, chronological RF ablation of robust feature transforms.

Outlier thresholds, imputation and feature selection are fitted only on the
training partition of each fold. Labels, raw prices and P&L remain untouched.
"""
from pathlib import Path
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from .outlier_eda import fit_outlier_bounds, apply_outlier_bounds
from .rf_deep_research import metrics, entry_metrics, probability_matrix, training_indices
from .classifier_benchmark import eligible_universe
from .feature_ablation import development_holdout_split
from .historical_market import add_historical_market_features
from .accuracy_features import add_accuracy_features
from .volume_regime_research import add_volume_regime_features
from .walk_forward import expanding_folds, fold_data

# One feature family is deliberately unchanged to anchor comparisons.
TREATMENTS={'raw':None, 'clip_0p5':(.005,.995), 'clip_1':(.01,.99), 'clip_2':(.02,.98)}
RF_SETTINGS={'leaf8':8,'leaf20':20}
THRESHOLDS=(.40,.45,.50,.55,.60)

def robust_matrices(train, valid, columns, treatment, cap=60000, seed=42):
    """Fit every transformation on sampled training rows, never validation rows."""
    ix=training_indices(train,cap,seed,'uniform')
    a=train.iloc[ix][columns].replace([np.inf,-np.inf],np.nan).copy()
    b=valid[columns].replace([np.inf,-np.inf],np.nan).copy()
    bounds=None
    if treatment is not None:
        bounds=fit_outlier_bounds(a,columns,np.ones(len(a),dtype=bool),lower=treatment[0],upper=treatment[1],min_finite=30)
        a,_=apply_outlier_bounds(a,bounds)
        b,_=apply_outlier_bounds(b,bounds)
    imp=SimpleImputer(strategy='median',keep_empty_features=True)
    x=imp.fit_transform(a); z=imp.transform(b)
    # Constant predictors cannot contribute to a tree split.
    keep=VarianceThreshold(threshold=0)
    x=keep.fit_transform(x);z=keep.transform(z)
    return x,z,ix,keep.get_support(),bounds

def fit_candidate(tr,va,cols,treatment,leaf,cap=60000,trees=200,seed=42):
    x,z,ix,keep,bounds=robust_matrices(tr,va,cols,treatment,cap,seed)
    model=RandomForestClassifier(n_estimators=trees,min_samples_leaf=leaf,max_depth=12,
                                 max_features='sqrt',class_weight=None,n_jobs=-1,random_state=seed)
    model.fit(x,tr.iloc[ix].direction.to_numpy(dtype=int))
    return probability_matrix(model,z),model,keep,bounds

def select_inner(rows,min_count=150,min_coverage=.025):
    tab=pd.DataFrame(rows)
    grouped=tab.groupby(['features','treatment','leaf','threshold'],as_index=False).agg(
        accuracy=('accuracy','mean'),auc=('auc','mean'),mean_net=('mean_net','mean'),
        worst_net=('mean_net','min'),min_count=('entry_count','min'),
        min_coverage=('coverage','min'),folds=('inner_fold','nunique'))
    acc=grouped.sort_values(['accuracy','auc'],ascending=False).iloc[0].to_dict()
    eligible=grouped[(grouped.folds>=2)&(grouped.min_count>=min_count)&(grouped.min_coverage>=min_coverage)].copy()
    if eligible.empty:return {'accuracy':acc,'entry':None},grouped
    eligible['robust_score']=eligible.mean_net+.25*eligible.worst_net
    return {'accuracy':acc,'entry':eligible.sort_values(['robust_score','min_coverage'],ascending=False).iloc[0].to_dict()},grouped

def run_robust_rf(frame,*,n_outer=3,n_inner=2,cap=60000,trees=200,seed=42,
                  cost_bps=10,min_count=150,min_coverage=.025,output_dir=None):
    if n_outer!=3 or n_inner!=2:raise ValueError('Use three outer and two inner chronological folds')
    eligible,_=eligible_universe(frame)
    dev,holdout_cutoff=development_holdout_split(eligible,.2)
    data,hist=add_historical_market_features(dev)
    data,tech=add_accuracy_features(data)
    data,vol=add_volume_regime_features(data)
    sets={'price':list(dict.fromkeys(hist+tech)),
          'price_volume':list(dict.fromkeys(hist+tech+vol)),
          'volume':list(dict.fromkeys(vol))}
    records={'inner':[],'outer':[],'selections':[],'importance':[],'trades':[],'bounds':[]}
    for outer in expanding_folds(data,n_folds=n_outer):
        inner_data=data[pd.to_datetime(data.next_bar_timestamp,utc=True)<=outer.train_end].copy()
        local=[]
        for inner in expanding_folds(inner_data,n_folds=n_inner):
            tr,va,_,_,_=fold_data(inner_data,inner,target_q=7)
            y=va.direction.to_numpy(dtype=int);r=va.next_return.to_numpy(dtype=float)
            for family,cols in sets.items():
                for name,treatment in TREATMENTS.items():
                    for label,leaf in RF_SETTINGS.items():
                        start=time.perf_counter()
                        p,_,_,_=fit_candidate(tr,va,cols,treatment,leaf,cap,trees,seed+outer.number*100+inner.number)
                        core=metrics(y,p)
                        for threshold in THRESHOLDS:
                            local.append(dict(outer_fold=outer.number,inner_fold=inner.number,features=family,
                                              treatment=name,leaf=label,threshold=threshold,seconds=time.perf_counter()-start,
                                              **core,**entry_metrics(y,p,r,threshold,cost_bps)))
        records['inner'].extend(local)
        selections,_=select_inner(local,min_count,min_coverage)
        tr,va,_,_,_=fold_data(data,outer,target_q=7)
        y=va.direction.to_numpy(dtype=int);r=va.next_return.to_numpy(dtype=float)
        for objective,choice in selections.items():
            if choice is None:
                records['selections'].append(dict(fold=outer.number,objective=objective,status='no_eligible_candidate'))
                continue
            family=choice['features'];name=choice['treatment'];leaf=choice['leaf'];threshold=choice['threshold']
            cols=sets[family]
            p,model,keep,bounds=fit_candidate(tr,va,cols,TREATMENTS[name],RF_SETTINGS[leaf],cap,trees,seed+outer.number)
            core=metrics(y,p);entry=entry_metrics(y,p,r,threshold,cost_bps)
            records['outer'].append(dict(fold=outer.number,objective=objective,features=family,treatment=name,
                leaf=leaf,threshold=threshold,n=len(y),majority_baseline=np.bincount(y,minlength=3).max()/len(y),
                up_baseline=np.mean(y==1),**core,**entry))
            records['selections'].append(dict(fold=outer.number,objective=objective,status='selected',**choice))
            for col,importance in zip(np.asarray(cols)[keep],model.feature_importances_):
                records['importance'].append(dict(fold=outer.number,objective=objective,feature=col,importance=importance))
            if bounds is not None:
                for feature,b in bounds.iterrows():
                    records['bounds'].append(dict(fold=outer.number,objective=objective,feature=feature,**b.to_dict()))
            selected=(p.argmax(axis=1)==1)&(p[:,1]>=threshold)
            for j in np.flatnonzero(selected):
                records['trades'].append(dict(fold=outer.number,objective=objective,symbol=str(va.iloc[j].symbol),
                    timestamp=str(va.iloc[j].bar_timestamp),gross_return=float(r[j]),
                    net_return=float(r[j]-cost_bps/10000),prob_up=float(p[j,1])))
    result={k:pd.DataFrame(v) for k,v in records.items()}
    if output_dir:
        path=Path(output_dir);path.mkdir(parents=True,exist_ok=True)
        for key,table in result.items():table.to_csv(path/f'{key}.csv',index=False)
    return result,holdout_cutoff

def daily_returns(trades):
    if trades.empty:return pd.DataFrame()
    x=trades.copy();x['day']=pd.to_datetime(x.timestamp,utc=True).dt.date
    return x.groupby(['fold','objective','day'],as_index=False).agg(n=('net_return','size'),mean_net=('net_return','mean'))
