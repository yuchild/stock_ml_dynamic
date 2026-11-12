"""Notebook 26: point-in-time OHLCV/market-regime research, development only.

Volume must be raw reported volume (not back-adjusted share volume); split days
can distort volume signals. All windows are backward-looking. This module never
loads the historical holdout or prospective ledger.
"""
from __future__ import annotations
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, precision_score, recall_score
from .classifier_benchmark import eligible_universe
from .feature_ablation import development_holdout_split
from .historical_market import add_historical_market_features
from .accuracy_features import add_accuracy_features
from .walk_forward import expanding_folds, fold_data

WINDOWS=(5,10,20,60)

def add_volume_regime_features(frame):
    """Only current/prior bar information. Cross-section uses same-timestamp peers.

    For predictions at the daily close, current day's finalized volume is allowed;
    do not use these features for pre-close entries.
    """
    d=frame.copy()
    d['bar_timestamp']=pd.to_datetime(d['bar_timestamp'],utc=True)
    d=d.sort_values(['symbol','interval_code','bar_timestamp']).reset_index(drop=True)
    keys=['symbol','interval_code']
    if d.duplicated(keys+['bar_timestamp']).any(): raise ValueError('Duplicate OHLCV bars')
    required={'adj_close','volume','high_price','low_price','close_price'}
    if not required.issubset(d.columns): raise ValueError(f'Missing OHLCV: {sorted(required-set(d.columns))}')
    close=pd.to_numeric(d.adj_close,errors='coerce')
    volume=pd.to_numeric(d.volume,errors='coerce').where(lambda s:s>=0)
    high=pd.to_numeric(d.high_price,errors='coerce');low=pd.to_numeric(d.low_price,errors='coerce')
    raw_close=pd.to_numeric(d.close_price,errors='coerce')
    if (close.dropna()<=0).any(): raise ValueError('Nonpositive adjusted close')
    d['_price']=close;d['_volume']=volume
    g=d.groupby(keys,sort=False)
    d['_return']=g['_price'].pct_change(fill_method=None)
    d['_dollar_volume']=raw_close*volume
    d['_log_volume']=np.log1p(volume)
    d['_signed_flow']=np.sign(d['_return'])*volume
    spread=(high-low).where((high-low)>0)
    d['_close_location']=((2*raw_close-high-low)/spread).clip(-1,1)
    d['_money_flow']=d['_close_location']*volume
    names=[]
    for w in WINDOWS:
        vg=d.groupby(keys,sort=False)['_volume']
        past_avg=vg.transform(lambda s:s.shift(1).rolling(w,min_periods=w).mean())
        past_std=vg.transform(lambda s:s.shift(1).rolling(w,min_periods=w).std())
        d[f'volume_relative_{w}']=volume/past_avg.replace(0,np.nan)
        d[f'volume_z_{w}']=(volume-past_avg)/past_std.replace(0,np.nan)
        d[f'volume_cv_{w}']=past_std/past_avg.replace(0,np.nan)
        d[f'dollar_volume_logmean_{w}']=d.groupby(keys,sort=False)['_dollar_volume'].transform(
            lambda s:np.log1p(s.clip(lower=0)).rolling(w,min_periods=w).mean())
        d[f'signed_volume_pressure_{w}']=d.groupby(keys,sort=False)['_signed_flow'].transform(
            lambda s:s.rolling(w,min_periods=w).sum()) / vg.transform(lambda s:s.rolling(w,min_periods=w).sum()).replace(0,np.nan)
        d[f'chaikin_flow_{w}']=d.groupby(keys,sort=False)['_money_flow'].transform(
            lambda s:s.rolling(w,min_periods=w).sum()) / vg.transform(lambda s:s.rolling(w,min_periods=w).sum()).replace(0,np.nan)
        d[f'volume_return_corr_{w}']=d.groupby(keys,sort=False).apply(
            lambda x:x['_return'].rolling(w,min_periods=w).corr(x['_log_volume']),include_groups=False
        ).reset_index(level=[0,1],drop=True).reindex(d.index)
        names.extend([f'{prefix}_{w}' for prefix in ('volume_relative','volume_z','volume_cv',
            'dollar_volume_logmean','signed_volume_pressure','chaikin_flow','volume_return_corr')])
    d['volume_price_impact']=d['_return'].abs()/np.log1p(volume).replace(0,np.nan)
    d['return_x_volume_shock']=d['_return']*d['volume_relative_20'].clip(upper=20)
    names+=['volume_price_impact','return_x_volume_shock']
    by_time=d.groupby(['interval_code','bar_timestamp'])
    d['_up']=d['_return'].gt(0).where(d['_return'].notna()).astype(float)
    # Leave-one-out breadth; no self contribution. Universe is fixed for research.
    n=by_time['_up'].transform('count')-d['_up'].notna().astype(int)
    total=by_time['_up'].transform('sum')-d['_up'].fillna(0)
    d['peer_advance_fraction']=(total/n).where(n>=10)
    names.append('peer_advance_fraction')
    d['_high_volume_up']=(d['_return']>0).astype(float)*volume
    sum_up=by_time['_high_volume_up'].transform('sum')-d['_high_volume_up'].fillna(0)
    sum_vol=by_time['_volume'].transform('sum')-volume.fillna(0)
    d['peer_volume_weighted_advance']=(sum_up/sum_vol.replace(0,np.nan)).where(n>=10)
    names.append('peer_volume_weighted_advance')
    d['peer_breadth_5']=d.groupby(keys,sort=False)['peer_advance_fraction'].transform(
        lambda s:s.rolling(5,min_periods=5).mean())
    names.append('peer_breadth_5')
    d['volume_data_available']=volume.notna().astype(float)
    names.append('volume_data_available')
    d.drop(columns=[c for c in d if c.startswith('_')],inplace=True)
    return d,names

CONFIGS=(
    ('logistic_balanced','logistic',True),('logistic_unweighted','logistic',False),
    ('random_forest_balanced','forest',True),('random_forest_unweighted','forest',False),
    ('extra_trees_balanced','extra',True),('hist_boost_balanced','boost',True),
)

def _model(family,balanced,seed):
    w='balanced' if balanced else None
    if family=='logistic':return LogisticRegression(C=.1,max_iter=400,class_weight=w)
    if family=='forest':return RandomForestClassifier(n_estimators=180,max_depth=14,min_samples_leaf=15,class_weight=w,n_jobs=-1,random_state=seed)
    if family=='extra':return ExtraTreesClassifier(n_estimators=180,max_depth=16,min_samples_leaf=15,class_weight=w,n_jobs=-1,random_state=seed)
    if family=='boost':return HistGradientBoostingClassifier(max_iter=180,max_leaf_nodes=15,learning_rate=.05,random_state=seed)
    raise ValueError(family)

def _metrics(y,p,returns,threshold=.5,cost_bps=10):
    pred=p.argmax(axis=1);entry=(pred==1)&(p[:,1]>=threshold)
    base=float((y==1).mean())
    return dict(n=len(y),accuracy=float(accuracy_score(y,pred)),balanced_accuracy=float(balanced_accuracy_score(y,pred)),
        roc_auc=float(roc_auc_score(y,p,labels=[0,1,2],multi_class='ovr',average='macro')),
        majority_baseline=float(np.bincount(y,minlength=3).max()/len(y)),up_baseline=base,
        up_precision=float(precision_score(y,pred,labels=[1],average='macro',zero_division=0)),
        up_recall=float(recall_score(y,pred,labels=[1],average='macro',zero_division=0)),
        entry_threshold=threshold,entry_coverage=float(entry.mean()),entry_count=int(entry.sum()),
        entry_precision=float((y[entry]==1).mean()) if entry.any() else np.nan,
        entry_precision_lift=float((y[entry]==1).mean()-base) if entry.any() else np.nan,
        mean_entry_gross_return=float(np.mean(returns[entry])) if entry.any() else np.nan,
        mean_entry_net_return=float(np.mean(returns[entry])-cost_bps/10000) if entry.any() else np.nan)

def run_volume_regime_research(frame,*,n_folds=3,max_train_rows=60000,tree_train_rows=40000,
        seed=42,thresholds=(.45,.50,.55,.60),cost_bps=10,output_dir=None,configs=CONFIGS):
    """Predeclared model x feature ablation, with paired test rows.

    All output scores are development-only exploratory. Do not tune thresholds
    from these outer-fold results and then claim an unbiased trading estimate.
    """
    if n_folds!=3:raise ValueError('Research protocol requires three chronological folds')
    eligible,_=eligible_universe(frame)
    dev,cutoff=development_holdout_split(eligible,.2)
    expanded,history=add_historical_market_features(dev)
    expanded,technical=add_accuracy_features(expanded)
    expanded,volume=add_volume_regime_features(expanded)
    features={
        'price_only':list(dict.fromkeys(history+technical)),
        'price_plus_volume':list(dict.fromkeys(history+technical+volume)),
        'volume_only':list(dict.fromkeys(volume)),
        'price_plus_flow':list(dict.fromkeys(history+technical+[x for x in volume if 'flow' in x or 'pressure' in x or 'breadth' in x])),
    }
    # Preserve identical evaluation rows for all configurations; impute missing volume in training only.
    common=features['price_only']
    rows=[];diagnostics=[]
    for fold in expanding_folds(expanded,n_folds=n_folds):
        train,test,xb,xv,_=fold_data(expanded,fold,target_q=7)
        ok_tr=train[common].replace([np.inf,-np.inf],np.nan).notna().all(axis=1).to_numpy()
        ok_te=test[common].replace([np.inf,-np.inf],np.nan).notna().all(axis=1).to_numpy()
        train=train.loc[ok_tr];test=test.loc[ok_te];xb=xb[ok_tr];xv=xv[ok_te]
        if len(train)<200 or len(test)<50:raise ValueError('Insufficient paired observations')
        y=train.direction.to_numpy(dtype=int);yt=test.direction.to_numpy(dtype=int)
        r=test.next_return.to_numpy(dtype=float)
        diagnostics.append(dict(fold=fold.number,n_train=len(train),n_test=len(test),
            test_start=str(test.bar_timestamp.min()),test_end=str(test.bar_timestamp.max()),
            volume_missing_rate=float(test.volume.isna().mean()),up_base=float((yt==1).mean()),
            median_up_return=float(np.nanmedian(r[yt==1])) if (yt==1).any() else np.nan,
            fraction_up_return_above_cost=float(np.mean(r[yt==1]>cost_bps/10000)) if (yt==1).any() else np.nan))
        for feature_name,cols in features.items():
            a=train[cols].replace([np.inf,-np.inf],np.nan).to_numpy(dtype=float)
            b=test[cols].replace([np.inf,-np.inf],np.nan).to_numpy(dtype=float)
            # Missing indicators are intentionally included: missing-volume patterns may carry signal.
            prep=make_pipeline(SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True),StandardScaler())
            a=prep.fit_transform(a);b=prep.transform(b)
            xa=np.column_stack((xb,a));xt=np.column_stack((xv,b))
            for name,family,balanced in configs:
                cap=max_train_rows if family=='logistic' else tree_train_rows
                ix=np.sort(np.random.default_rng(seed+fold.number*1009).choice(len(y),min(len(y),cap),replace=False))
                model=_model(family,balanced,seed+fold.number)
                start=time.perf_counter()
                if family=='boost' and balanced:
                    from sklearn.utils.class_weight import compute_sample_weight
                    model.fit(xa[ix],y[ix],sample_weight=compute_sample_weight('balanced',y[ix]))
                else:model.fit(xa[ix],y[ix])
                elapsed=time.perf_counter()-start
                pp=model.predict_proba(xt)
                p=np.zeros((len(yt),3),dtype=float)
                for j,c in enumerate(model.classes_):p[:,int(c)]=pp[:,j]
                if not np.isfinite(p).all() or not np.allclose(p.sum(axis=1),1,atol=1e-5):raise ValueError('Invalid probabilities')
                for threshold in thresholds:
                    rows.append(dict(fold=fold.number,feature_set=feature_name,model=name,balanced=balanced,
                        train_seconds=elapsed,**_metrics(yt,p,r,threshold,cost_bps)))
    results=pd.DataFrame(rows);diagnostics=pd.DataFrame(diagnostics)
    # A fixed threshold makes leaderboard comparisons interpretable; other thresholds remain diagnostics.
    fixed=results[np.isclose(results.entry_threshold,.5)]
    leaderboard=(fixed.groupby(['model','feature_set'],as_index=False).agg(
        mean_accuracy=('accuracy','mean'),mean_baseline=('majority_baseline','mean'),
        mean_auc=('roc_auc','mean'),mean_balanced_accuracy=('balanced_accuracy','mean'),
        mean_entry_precision=('entry_precision','mean'),mean_entry_coverage=('entry_coverage','mean'),
        mean_net_return=('mean_entry_net_return','mean'),mean_fit_seconds=('train_seconds','mean')))
    leaderboard['accuracy_lift']=leaderboard.mean_accuracy-leaderboard.mean_baseline
    leaderboard=leaderboard.sort_values(['mean_accuracy','mean_auc'],ascending=False)
    if output_dir is not None:
        out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
        results.to_csv(out/'all_fold_model_threshold_results.csv',index=False)
        leaderboard.to_csv(out/'fixed_threshold_leaderboard.csv',index=False)
        diagnostics.to_csv(out/'fold_data_diagnostics.csv',index=False)
        pd.DataFrame([{'feature_set':k,'feature':x} for k,v in features.items() for x in v]).to_csv(out/'feature_manifest.csv',index=False)
    return results,leaderboard,diagnostics,cutoff
