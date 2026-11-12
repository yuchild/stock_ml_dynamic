"""Notebook 18: development-only repeated-seed robustness diagnostics."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .architecture_ensemble import run_architecture_ensemble

KEY_MODELS=('logistic_l2','random_forest','residual_mlp','soft_vote_lr_rf','soft_vote_all')

def paired_differences(scores,reference='random_forest',metric='roc_auc'):
    required={'seed','fold','model',metric}
    if not required.issubset(scores.columns):raise ValueError(f'Missing columns: {sorted(required-set(scores.columns))}')
    if scores.duplicated(['seed','fold','model']).any():raise ValueError('Duplicate seed/fold/model records')
    wide=scores.pivot(index=['seed','fold'],columns='model',values=metric)
    if reference not in wide:raise ValueError('Reference model absent')
    result=[]
    for model in wide.columns:
        if model==reference:continue
        delta=(wide[model]-wide[reference]).dropna()
        if delta.empty:continue
        result.append({'model':model,'reference':reference,'n_paired_runs':len(delta),'mean_delta':float(delta.mean()),'median_delta':float(delta.median()),'positive_fraction':float((delta>0).mean()),'min_delta':float(delta.min()),'max_delta':float(delta.max())})
    return pd.DataFrame(result).sort_values('mean_delta',ascending=False).reset_index(drop=True)

def summarize_repeats(scores):
    if scores.empty:raise ValueError('No benchmark results')
    metrics=['accuracy','balanced_accuracy','roc_auc','log_loss','fit_seconds','predict_seconds']
    available=[c for c in metrics if c in scores]
    agg=scores.groupby('model',as_index=False)[available].agg(['mean','std'])
    agg.columns=['model' if c[0]=='model' else f'{c[0]}_{c[1]}' for c in agg.columns]
    return agg.sort_values('roc_auc_mean',ascending=False).reset_index(drop=True)

def run_robustness(master,*,seeds=(42,52,62),n_folds=3,neural_epochs=12,max_train_rows=60000,tree_train_rows=45000,output_dir=None):
    if len(set(seeds))!=len(seeds) or not seeds:raise ValueError('Seeds must be nonempty and unique')
    if n_folds<2:raise ValueError('Use at least two chronological folds')
    out=Path(output_dir) if output_dir else None
    if out:out.mkdir(parents=True,exist_ok=True)
    all_scores=[]; boundaries=[]
    for seed in seeds:
        run_dir=out/f'seed_{seed}' if out else None
        scores,_,_,boundary=run_architecture_ensemble(master,n_folds=n_folds,neural_epochs=neural_epochs,max_train_rows=max_train_rows,tree_train_rows=tree_train_rows,include_stacking=False,seed=int(seed),output_dir=run_dir)
        scores=scores.copy();scores.insert(0,'seed',int(seed));all_scores.append(scores);boundaries.append(str(boundary))
        if out:pd.concat(all_scores,ignore_index=True).to_csv(out/'all_fold_metrics.csv',index=False)
    if len(set(boundaries))!=1:raise RuntimeError('Development boundary changed between seeds')
    combined=pd.concat(all_scores,ignore_index=True)
    summary=summarize_repeats(combined)
    paired=paired_differences(combined)
    if out:
        summary.to_csv(out/'repeat_summary.csv',index=False)
        paired.to_csv(out/'paired_auc_differences.csv',index=False)
        (out/'manifest.json').write_text(json.dumps({'status':'DEVELOPMENT_ONLY','seeds':list(seeds),'folds':n_folds,'neural_epochs':neural_epochs,'development_end':boundaries[0],'final_holdout':'NOT_ACCESSED','prospective_ledger':'NOT_ACCESSED','caveat':'Repeated folds and seeds are correlated; deltas are descriptive, not independent statistical significance tests.'},indent=2))
    return combined,summary,paired,boundaries[0]
