"""Notebook 23: q=7 class-1 entry signal, chronological threshold selection.

Outer evaluation is never used to choose a probability threshold or weighting.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, roc_auc_score
from .classifier_benchmark import eligible_universe
from .feature_ablation import development_holdout_split
from .historical_market import add_historical_market_features
from .accuracy_features import add_accuracy_features
from .nested_accuracy import feature_sets, _matrix, _predict
from .walk_forward import expanding_folds, fold_data

THRESHOLDS = (0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65)
CONFIGS = ('logistic_balanced', 'logistic_unweighted', 'forest_balanced', 'forest_unweighted')

def make_estimator(config, seed=42):
    if config not in CONFIGS:
        raise ValueError(config)
    weight = 'balanced' if config.endswith('_balanced') else None
    if config.startswith('logistic'):
        return LogisticRegression(C=.1, max_iter=400, class_weight=weight)
    return RandomForestClassifier(n_estimators=160, max_depth=12, min_samples_leaf=20,
                                  class_weight=weight, n_jobs=-1, random_state=seed)

def entry_metrics(y, p, threshold, *, min_coverage=0.05, cost_bps=10, next_returns=None):
    """Only class 1 can trigger an entry. Report abstentions explicitly."""
    y=np.asarray(y, dtype=int); p=np.asarray(p, dtype=float)
    if p.ndim!=2 or p.shape[1]!=3 or len(y)!=len(p):
        raise ValueError('Expected aligned (n,3) class probabilities')
    if not 0<=threshold<=1 or not 0<=min_coverage<=1:
        raise ValueError('Invalid threshold or coverage')
    chosen=(p[:,1]>=threshold) & (np.argmax(p,axis=1)==1)
    n=int(chosen.sum()); coverage=float(chosen.mean()) if len(y) else 0.
    up_rate=float((y==1).mean()) if len(y) else np.nan
    out=dict(n=int(len(y)),entries=n,coverage=coverage,threshold=float(threshold),
             precision_up=float(np.mean(y[chosen]==1)) if n else np.nan,
             recall_up=float(n and np.sum(y[chosen]==1)/max(1,np.sum(y==1))) if len(y) else np.nan,
             base_up_rate=up_rate,precision_lift=float(np.mean(y[chosen]==1)-up_rate) if n else np.nan,
             meets_min_coverage=bool(coverage>=min_coverage),
             overall_accuracy=float(accuracy_score(y,p.argmax(axis=1))) if len(y) else np.nan,
             balanced_accuracy=float(balanced_accuracy_score(y,p.argmax(axis=1))) if len(y) else np.nan)
    if next_returns is not None:
        r=np.asarray(next_returns,dtype=float)
        if len(r)!=len(y):raise ValueError('Return length mismatch')
        net=r[chosen] - cost_bps/10000
        out['mean_entry_gross_return']=float(np.mean(r[chosen])) if n else np.nan
        out['mean_entry_net_return']=float(np.mean(net)) if n else np.nan
        out['positive_net_return_rate']=float(np.mean(net>0)) if n else np.nan
    return out

def choose_entry_rule(scores, min_coverage=0.05):
    """Maximize pooled inner precision subject to minimum coverage, then recall.

    If no configuration qualifies, return None (abstain) instead of cherry-picking.
    """
    df=pd.DataFrame(scores)
    if df.empty: return None, pd.DataFrame()
    rows=[]
    for (config,threshold),g in df.groupby(['config','threshold']):
        # Aggregate counts across inner folds; avoid averaging unequal fold sizes.
        entries=int(g.entries.sum()); observations=int(g.n.sum())
        wins=float((g.precision_up.fillna(0)*g.entries).sum())
        positives=float((g.base_up_rate*g.n).sum())
        rows.append(dict(config=config,threshold=float(threshold),entries=entries,n=observations,
                         coverage=entries/observations if observations else 0,
                         precision_up=wins/entries if entries else np.nan,
                         baseline_up=positives/observations if observations else np.nan))
    table=pd.DataFrame(rows)
    valid=table.loc[(table.coverage>=min_coverage)&(table.entries>0)].copy()
    if valid.empty:return None,table
    valid=valid.sort_values(['precision_up','coverage','config','threshold'],ascending=[False,False,True,True])
    best=valid.iloc[0]
    return (str(best.config),float(best.threshold)),table

def _fold_predict(expanded, fold, cols, config, max_train_rows, tree_train_rows, seed):
    train,test,xb,xv,_=fold_data(expanded,fold,target_q=7)
    # Same complete-case population for all models.
    mask_train=train[cols].replace([np.inf,-np.inf],np.nan).notna().all(axis=1).to_numpy()
    mask_test=test[cols].replace([np.inf,-np.inf],np.nan).notna().all(axis=1).to_numpy()
    train=train.loc[mask_train];test=test.loc[mask_test];xb=xb[mask_train];xv=xv[mask_test]
    x,z,y,yt=_matrix(train,test,xb,xv,cols)
    cap=tree_train_rows if config.startswith('forest') else max_train_rows
    rng=np.random.default_rng(seed+fold.number*113)
    ids=np.sort(rng.choice(len(y),size=min(cap,len(y)),replace=False))
    model=make_estimator(config,seed)
    model.fit(x[ids],y[ids]);p=_predict(model,z)
    return yt,p,test.next_return.to_numpy(dtype=float)

def run_entry_signal(frame, *, n_outer=3,n_inner=2,min_coverage=.05,max_train_rows=25000,
                     tree_train_rows=15000,seed=42,cost_bps=10,output_dir=None):
    if n_outer!=3 or n_inner<2:raise ValueError('Use three outer and at least two inner folds')
    if not 0<min_coverage<=1:raise ValueError('Invalid min_coverage')
    eligible,_=eligible_universe(frame)
    development,cutoff=development_holdout_split(eligible,.2)
    expanded,base=add_historical_market_features(development)
    expanded,new=add_accuracy_features(expanded)
    cols=feature_sets(base,new)['all']
    inner_records=[];outer_records=[];selection_records=[];threshold_records=[]
    for outer in expanding_folds(expanded,n_folds=n_outer):
        history=expanded.loc[pd.to_datetime(expanded.bar_timestamp,utc=True)<=outer.train_end].copy()
        local=[]
        for inner in expanding_folds(history,n_folds=n_inner):
            for config in CONFIGS:
                yt,p,returns=_fold_predict(history,inner,cols,config,max_train_rows,tree_train_rows,seed)
                for threshold in THRESHOLDS:
                    stats=entry_metrics(yt,p,threshold,min_coverage=min_coverage,cost_bps=cost_bps,next_returns=returns)
                    local.append(dict(outer_fold=outer.number,inner_fold=inner.number,config=config,**stats))
        rule,summary=choose_entry_rule(local,min_coverage)
        inner_records.extend(local)
        summary['outer_fold']=outer.number
        threshold_records.append(summary)
        if rule is None:
            selection_records.append(dict(outer_fold=outer.number,config='abstain',threshold=np.nan,reason='No rule met inner minimum coverage'))
            outer_records.append(dict(outer_fold=outer.number,config='abstain',entries=0,coverage=0.,precision_up=np.nan,reason='No qualifying inner rule'))
            continue
        config,threshold=rule
        selection_records.append(dict(outer_fold=outer.number,config=config,threshold=threshold,reason='Inner-only precision selection'))
        yt,p,returns=_fold_predict(expanded,outer,cols,config,max_train_rows,tree_train_rows,seed)
        stats=entry_metrics(yt,p,threshold,min_coverage=min_coverage,cost_bps=cost_bps,next_returns=returns)
        outer_records.append(dict(outer_fold=outer.number,config=config,**stats))
        print(f'Outer {outer.number}: {config}, threshold {threshold:.2f}, entry precision {stats["precision_up"]:.3f}, coverage {stats["coverage"]:.3f}',flush=True)
    inner_df=pd.DataFrame(inner_records);outer_df=pd.DataFrame(outer_records);selection_df=pd.DataFrame(selection_records)
    if output_dir is not None:
        out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
        inner_df.to_csv(out/'inner_entry_thresholds.csv',index=False)
        outer_df.to_csv(out/'outer_entry_evaluation.csv',index=False)
        selection_df.to_csv(out/'inner_selected_rules.csv',index=False)
        pd.concat(threshold_records,ignore_index=True).to_csv(out/'inner_threshold_summaries.csv',index=False)
        (out/'manifest.json').write_text(json.dumps(dict(target_q=7,target_labels=[2,2,0,0,1,1,1],outer_folds=n_outer,inner_folds=n_inner,
             selection='inner class-1 precision subject to coverage',minimum_coverage=min_coverage,cost_bps=cost_bps,
             development_cutoff=str(cutoff),historical_holdout='NOT_ACCESSED',prospective_ledger='NOT_MODIFIED',
             warning='Net returns are simple one-bar proxies; no execution simulation, slippage or portfolio constraints.'),indent=2))
    return inner_df,outer_df,selection_df,cutoff
