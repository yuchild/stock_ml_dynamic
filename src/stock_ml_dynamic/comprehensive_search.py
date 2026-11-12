"""Notebook 24: accuracy and selective-entry model research, development only.

All choices use inner chronological folds; outer folds evaluate frozen policies.
The historical holdout and prospective ledger are never read or modified.
"""
from __future__ import annotations
import json
import time
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, roc_auc_score, log_loss
from sklearn.utils.class_weight import compute_sample_weight
from .classifier_benchmark import eligible_universe
from .feature_ablation import development_holdout_split
from .historical_market import add_historical_market_features
from .accuracy_features import add_accuracy_features
from .nested_accuracy import feature_sets, _matrix, _predict
from .walk_forward import expanding_folds, fold_data
from .entry_signal import entry_metrics

@dataclass(frozen=True)
class Candidate:
    name: str
    family: str
    feature_set: str
    parameter: float
    balanced: bool = True

# Predeclared search space. No test-period-driven modifications.
CANDIDATES = (
    Candidate('logistic_base_bal','logistic','base',.1),
    Candidate('logistic_all_bal','logistic','all',.1),
    Candidate('logistic_vol_bal','logistic','volatility',.1),
    Candidate('logistic_all_unweighted','logistic','all',.1,False),
    Candidate('forest_all_leaf10','forest','all',10),
    Candidate('forest_all_leaf30','forest','all',30),
    Candidate('forest_vol_leaf20','forest','volatility',20),
    Candidate('extra_trees_all','extra_trees','all',20),
    Candidate('hist_boost_all','hist_boost','all',15),
    Candidate('hist_boost_vol','hist_boost','volatility',15),
)
THRESHOLDS = (.35,.40,.45,.50,.55,.60)

def build_estimator(candidate: Candidate, seed=42):
    weight = 'balanced' if candidate.balanced else None
    if candidate.family=='logistic':
        return LogisticRegression(C=candidate.parameter,max_iter=500,class_weight=weight)
    if candidate.family=='forest':
        return RandomForestClassifier(n_estimators=200,max_depth=14,min_samples_leaf=int(candidate.parameter),class_weight=weight,n_jobs=-1,random_state=seed)
    if candidate.family=='extra_trees':
        return ExtraTreesClassifier(n_estimators=200,max_depth=16,min_samples_leaf=int(candidate.parameter),class_weight=weight,n_jobs=-1,random_state=seed)
    if candidate.family=='hist_boost':
        return HistGradientBoostingClassifier(max_iter=150,max_leaf_nodes=int(candidate.parameter),learning_rate=.05,l2_regularization=1,early_stopping=False,random_state=seed)
    raise ValueError(candidate.family)

def predict_fold(expanded, fold, candidate, schemes, allcols, *, max_train_rows, tree_train_rows, seed):
    tr,te,xb,xv,_=fold_data(expanded,fold,target_q=7)
    # Use identical complete-case population for all candidates.
    mask_a=tr[allcols].replace([np.inf,-np.inf],np.nan).notna().all(axis=1).to_numpy()
    mask_b=te[allcols].replace([np.inf,-np.inf],np.nan).notna().all(axis=1).to_numpy()
    tr=tr.loc[mask_a];te=te.loc[mask_b];xb=xb[mask_a];xv=xv[mask_b]
    if len(tr)<100 or len(te)<20:raise ValueError('Insufficient complete-case rows')
    x,z,y,yt=_matrix(tr,te,xb,xv,schemes[candidate.feature_set])
    cap=max_train_rows if candidate.family=='logistic' else tree_train_rows
    rng=np.random.default_rng(seed+fold.number*1009)
    ix=np.sort(rng.choice(len(y),min(cap,len(y)),replace=False))
    model=build_estimator(candidate,seed)
    t0=time.perf_counter()
    if candidate.family=='hist_boost' and candidate.balanced:
        model.fit(x[ix],y[ix],sample_weight=compute_sample_weight('balanced',y[ix]))
    else:
        model.fit(x[ix],y[ix])
    elapsed=time.perf_counter()-t0
    p=_predict(model,z)
    if not np.isfinite(p).all() or not np.allclose(p.sum(axis=1),1,atol=1e-5):
        raise ValueError('Invalid class probabilities')
    return yt,p,te.next_return.to_numpy(dtype=float),elapsed

def classification_metrics(y,p):
    pred=np.argmax(p,axis=1)
    return dict(n=len(y),accuracy=float(accuracy_score(y,pred)),balanced_accuracy=float(balanced_accuracy_score(y,pred)),
                up_precision=float(precision_score(y,pred,labels=[1],average='macro',zero_division=0)),
                up_recall=float(recall_score(y,pred,labels=[1],average='macro',zero_division=0)),
                up_predicted_fraction=float(np.mean(pred==1)),base_up_rate=float(np.mean(y==1)),
                roc_auc=float(roc_auc_score(y,p,labels=[0,1,2],multi_class='ovr',average='macro')),
                log_loss=float(log_loss(y,p,labels=[0,1,2])))

def summarize_inner(records,min_coverage=.05,min_entries=100):
    df=pd.DataFrame(records)
    if df.empty:raise ValueError('No inner records')
    group=df.groupby('candidate',as_index=False).agg(
        accuracy=('accuracy','mean'),balanced_accuracy=('balanced_accuracy','mean'),
        roc_auc=('roc_auc','mean'),fit_seconds=('fit_seconds','mean'))
    acc=group.sort_values(['accuracy','balanced_accuracy','candidate'],ascending=[False,False,True]).iloc[0]['candidate']
    threshold=df.groupby(['candidate','threshold'],as_index=False).agg(
        n=('n','sum'),entries=('entries','sum'),wins=('up_wins','sum'),
        up_base_wins=('up_base_wins','sum'),net_return_sum=('net_return_sum','sum'),
        valid_inner_folds=('meets_min_coverage','sum'),folds=('meets_min_coverage','size'))
    threshold['coverage']=threshold.entries/threshold.n
    threshold['precision']=threshold.wins/threshold.entries.replace(0,np.nan)
    threshold['baseline_up']=threshold.up_base_wins/threshold.n
    threshold['precision_lift']=threshold.precision-threshold.baseline_up
    threshold['mean_net_return']=threshold.net_return_sum/threshold.entries.replace(0,np.nan)
    eligible=threshold[(threshold.entries>=min_entries)&(threshold.coverage>=min_coverage)&(threshold.valid_inner_folds==threshold.folds)&(threshold.precision_lift>0)]
    if eligible.empty: entry=None
    else:
        best=eligible.sort_values(['precision_lift','mean_net_return','coverage','candidate'],ascending=[False,False,False,True]).iloc[0]
        entry=(str(best.candidate),float(best.threshold))
    return str(acc),entry,group,threshold

def run_comprehensive_search(frame,*,n_outer=3,n_inner=2,max_train_rows=50000,tree_train_rows=35000,
                             min_coverage=.05,min_entries=100,cost_bps=10,seed=42,output_dir=None,
                             candidates=CANDIDATES,thresholds=THRESHOLDS):
    if n_outer!=3 or n_inner<2:raise ValueError('Use three outer folds and at least two inner folds')
    if not candidates or not thresholds:raise ValueError('Candidates and thresholds required')
    eligible,_=eligible_universe(frame)
    development,cutoff=development_holdout_split(eligible,.2)
    expanded,base=add_historical_market_features(development)
    expanded,new=add_accuracy_features(expanded)
    schemes=feature_sets(base,new)
    allcols=schemes['all']
    inner_records=[];outer_records=[];choices=[];summary_records=[];threshold_records=[]
    by_name={c.name:c for c in candidates}
    if len(by_name)!=len(candidates):raise ValueError('Duplicate candidate names')
    for outer in expanding_folds(expanded,n_folds=n_outer):
        history=expanded.loc[pd.to_datetime(expanded.bar_timestamp,utc=True)<=outer.train_end].copy()
        local=[]
        for inner in expanding_folds(history,n_folds=n_inner):
            # fold_data fits quantiles, clustering and baseline preprocessing within this inner fold.
            for c in candidates:
                y,p,r,seconds=predict_fold(history,inner,c,schemes,allcols,max_train_rows=max_train_rows,
                                           tree_train_rows=tree_train_rows,seed=seed)
                common=classification_metrics(y,p)
                for t in thresholds:
                    stats=entry_metrics(y,p,t,min_coverage=min_coverage,cost_bps=cost_bps,next_returns=r)
                    mask=(p[:,1]>=t)&(p.argmax(axis=1)==1)
                    local.append(dict(outer_fold=outer.number,inner_fold=inner.number,candidate=c.name,
                                      fit_seconds=seconds,**common,threshold=t,entries=int(mask.sum()),
                                      up_wins=int(np.sum(y[mask]==1)),up_base_wins=float(np.mean(y==1)*len(y)),
                                      net_return_sum=float(np.sum(r[mask]-cost_bps/10000)),
                                      meets_min_coverage=stats['meets_min_coverage']))
        inner_records.extend(local)
        accuracy_choice,entry_choice,summary,threshold_summary=summarize_inner(local,min_coverage,min_entries)
        summary['outer_fold']=outer.number;threshold_summary['outer_fold']=outer.number
        summary_records.append(summary);threshold_records.append(threshold_summary)
        choices.append(dict(outer_fold=outer.number,policy='accuracy',candidate=accuracy_choice,threshold=np.nan))
        choices.append(dict(outer_fold=outer.number,policy='entry',candidate=entry_choice[0] if entry_choice else 'abstain',
                            threshold=entry_choice[1] if entry_choice else np.nan))
        # One outer evaluation per selected candidate; cache if two policies choose the same model.
        cache={}
        for policy,choice in [('accuracy',(accuracy_choice,None)),('entry',entry_choice)]:
            if choice is None:
                outer_records.append(dict(outer_fold=outer.number,policy=policy,candidate='abstain',entries=0,coverage=0.,
                                          accuracy=np.nan,balanced_accuracy=np.nan,roc_auc=np.nan,
                                          majority_baseline_accuracy=np.nan,precision_up=np.nan,
                                          mean_entry_net_return=np.nan,
                                          note='No inner rule met coverage and precision-lift requirements'))
                continue
            name,t=choice
            if name not in cache:
                cache[name]=predict_fold(expanded,outer,by_name[name],schemes,allcols,
                                         max_train_rows=max_train_rows,tree_train_rows=tree_train_rows,seed=seed)
            y,p,r,seconds=cache[name]
            row=dict(outer_fold=outer.number,policy=policy,candidate=name,fit_seconds=seconds,**classification_metrics(y,p),
                     majority_baseline_accuracy=float(np.max(np.bincount(y,minlength=3))/len(y)),
                     always_up_precision=float(np.mean(y==1)),
                     always_up_net_return=float(np.mean(r-cost_bps/10000)))
            if t is not None:
                row.update(entry_metrics(y,p,t,min_coverage=min_coverage,cost_bps=cost_bps,next_returns=r))
            outer_records.append(row)
        print(f'Outer {outer.number}/{n_outer}: accuracy={accuracy_choice}, entry={entry_choice or "abstain"}',flush=True)
    inner_df=pd.DataFrame(inner_records);outer_df=pd.DataFrame(outer_records);choices_df=pd.DataFrame(choices)
    if output_dir is not None:
        out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
        for filename,df in [('inner_grid.csv',inner_df),('outer_policy_evaluation.csv',outer_df),
                            ('inner_policy_choices.csv',choices_df),('inner_model_rankings.csv',pd.concat(summary_records,ignore_index=True)),
                            ('inner_threshold_rankings.csv',pd.concat(threshold_records,ignore_index=True))]:
            df.to_csv(out/filename,index=False)
        (out/'manifest.json').write_text(json.dumps(dict(target_q=7,target_labels=[2,2,0,0,1,1,1],
             outer_folds=n_outer,inner_folds=n_inner,candidates=[c.__dict__ for c in candidates],thresholds=list(thresholds),
             development_cutoff=str(cutoff),historical_holdout='NOT_ACCESSED',prospective_ledger='NOT_MODIFIED',
             selection='inner accuracy policy and separately inner class-1 precision lift policy',
             min_coverage=min_coverage,min_entries=min_entries,cost_bps=cost_bps,
             limitations='Historical per-stock next-bar returns are not portfolio returns; no liquidity, slippage, fills or position overlap simulation.'),indent=2))
    return inner_df,outer_df,choices_df,cutoff
