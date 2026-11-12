"""Notebook 25: predeclared expanded search and decision diagnostics.

Do not use outer-fold results to choose a deployable model. Historical holdout
and prospective ledger are outside this workflow.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .comprehensive_search import Candidate, run_comprehensive_search

# All choices fixed before observing Notebook 25 outer-fold results.
CANDIDATES_25 = (
    Candidate('log_all_c003_bal','logistic','all',.03,True),
    Candidate('log_all_c010_bal','logistic','all',.10,True),
    Candidate('log_all_c100_bal','logistic','all',1.0,True),
    Candidate('log_all_c003_plain','logistic','all',.03,False),
    Candidate('log_all_c010_plain','logistic','all',.10,False),
    Candidate('log_all_c100_plain','logistic','all',1.0,False),
    Candidate('log_vol_c010_bal','logistic','volatility',.10,True),
    Candidate('log_vol_c010_plain','logistic','volatility',.10,False),
    Candidate('forest_all_leaf5_bal','forest','all',5,True),
    Candidate('forest_all_leaf15_bal','forest','all',15,True),
    Candidate('forest_all_leaf30_bal','forest','all',30,True),
    Candidate('forest_all_leaf15_plain','forest','all',15,False),
    Candidate('forest_vol_leaf15_bal','forest','volatility',15,True),
    Candidate('extra_all_leaf10_bal','extra_trees','all',10,True),
    Candidate('extra_all_leaf30_bal','extra_trees','all',30,True),
    Candidate('boost_all_leaf15_bal','hist_boost','all',15,True),
    Candidate('boost_all_leaf31_bal','hist_boost','all',31,True),
    Candidate('boost_vol_leaf15_bal','hist_boost','volatility',15,True),
)
THRESHOLDS_25 = (.35,.40,.45,.50,.55,.60,.65)

def summarize_policies(outer: pd.DataFrame, *, min_coverage=.05, min_precision_lift=.03, min_positive_folds=2):
    """Descriptive only: no model selection from outer scores."""
    if outer.empty: raise ValueError('Empty outer evaluation')
    required={'policy','outer_fold','accuracy','majority_baseline_accuracy','precision_lift','coverage','mean_entry_net_return'}
    missing=required-set(outer.columns)
    if missing: raise ValueError(f'Missing evaluation columns: {sorted(missing)}')
    rows=[]
    for policy,g in outer.groupby('policy'):
        acc=(g.accuracy-g.majority_baseline_accuracy).dropna()
        lifts=pd.to_numeric(g.precision_lift,errors='coerce').dropna()
        nets=pd.to_numeric(g.mean_entry_net_return,errors='coerce').dropna()
        covers=pd.to_numeric(g.coverage,errors='coerce').dropna()
        rows.append(dict(policy=policy,folds=len(g),mean_accuracy=float(g.accuracy.mean()),
            mean_majority_baseline=float(g.majority_baseline_accuracy.mean()),
            mean_accuracy_lift=float(acc.mean()) if len(acc) else np.nan,
            accuracy_folds_above_baseline=int((acc>0).sum()),
            mean_up_precision_lift=float(lifts.mean()) if len(lifts) else np.nan,
            up_precision_folds_positive=int((lifts>0).sum()),
            mean_coverage=float(covers.mean()) if len(covers) else np.nan,
            mean_net_return=float(nets.mean()) if len(nets) else np.nan,
            net_positive_folds=int((nets>0).sum()),
            entry_gate_pass=bool(policy=='entry' and len(g)>=3 and len(lifts)==len(g) and
                len(nets)==len(g) and len(covers)==len(g) and
                (lifts>=min_precision_lift).sum()>=min_positive_folds and
                (nets>0).sum()>=min_positive_folds and
                (covers>=min_coverage).all())))
    return pd.DataFrame(rows)

def candidate_stability(inner: pd.DataFrame):
    """Inner-fold model diagnostics, not an unbiased performance estimate."""
    if inner.empty: raise ValueError('Empty inner evaluation')
    dedup=inner.drop_duplicates(['outer_fold','inner_fold','candidate'])
    return (dedup.groupby('candidate',as_index=False)
        .agg(mean_accuracy=('accuracy','mean'),std_accuracy=('accuracy','std'),
             mean_auc=('roc_auc','mean'),mean_balanced_accuracy=('balanced_accuracy','mean'),
             mean_fit_seconds=('fit_seconds','mean'),folds=('accuracy','size'))
        .sort_values(['mean_accuracy','mean_auc'],ascending=False))

def run_decision_research(frame, *, output_dir=None, n_outer=3,n_inner=2,
        max_train_rows=60000,tree_train_rows=40000,min_coverage=.05,
        min_entries=200,cost_bps=10,seed=42,candidates=CANDIDATES_25):
    inner,outer,choices,cutoff=run_comprehensive_search(frame,n_outer=n_outer,n_inner=n_inner,
        max_train_rows=max_train_rows,tree_train_rows=tree_train_rows,
        min_coverage=min_coverage,min_entries=min_entries,cost_bps=cost_bps,
        seed=seed,output_dir=output_dir,candidates=candidates,thresholds=THRESHOLDS_25)
    summary=summarize_policies(outer,min_coverage=min_coverage)
    stability=candidate_stability(inner)
    if output_dir is not None:
        from pathlib import Path
        p=Path(output_dir)
        summary.to_csv(p/'policy_decision_summary.csv',index=False)
        stability.to_csv(p/'inner_candidate_stability.csv',index=False)
    return inner,outer,choices,summary,stability,cutoff
