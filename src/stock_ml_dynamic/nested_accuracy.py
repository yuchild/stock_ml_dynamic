"""Notebook 22: nested, development-only, accuracy-prioritized model selection."""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, log_loss, precision_score, recall_score
from .classifier_benchmark import eligible_universe
from .feature_ablation import development_holdout_split
from .historical_market import add_historical_market_features
from .accuracy_features import add_accuracy_features
from .walk_forward import expanding_folds, fold_data

# Fixed before running: no searching the outer-fold outcomes for configurations.
CANDIDATES = (
    ('logistic_base', 'base', 'logistic', .10),
    ('logistic_all_c03', 'all', 'logistic', .03),
    ('logistic_all_c10', 'all', 'logistic', .10),
    ('logistic_vol_c10', 'volatility', 'logistic', .10),
    ('forest_all_leaf20', 'all', 'forest', 20),
    ('forest_all_leaf50', 'all', 'forest', 50),
)

def feature_sets(base, new):
    vol = [c for c in new if 'volatility' in c or 'downside' in c or 'return_z' in c]
    return {'base': list(base), 'volatility': list(dict.fromkeys(base+vol)), 'all': list(dict.fromkeys(base+new))}

def _model(family, parameter, seed):
    if family == 'logistic': return LogisticRegression(C=float(parameter),max_iter=400,class_weight='balanced')
    if family == 'forest': return RandomForestClassifier(n_estimators=160,max_depth=12,min_samples_leaf=int(parameter),class_weight="balanced",n_jobs=-1,random_state=seed)
    raise ValueError(family)

def _matrix(train,test,xb,xv,cols):
    a=train[cols].replace([np.inf,-np.inf],np.nan)
    b=test[cols].replace([np.inf,-np.inf],np.nan)
    ok_a=a.notna().all(axis=1).to_numpy()
    ok_b=b.notna().all(axis=1).to_numpy()
    if ok_a.sum()<100 or ok_b.sum()<20: raise ValueError('Insufficient paired rows')
    scaler=StandardScaler().fit(a.loc[ok_a])
    x=np.column_stack([xb[ok_a],scaler.transform(a.loc[ok_a])]);z=np.column_stack([xv[ok_b],scaler.transform(b.loc[ok_b])])
    return x,z,train.direction.to_numpy(dtype=int)[ok_a],test.direction.to_numpy(dtype=int)[ok_b]

def _predict(model,z):
    p=np.zeros((len(z),3),dtype=float)
    for j,c in enumerate(model.classes_):p[:,int(c)]=model.predict_proba(z)[:,j]
    return p

def _score(y,p):
    pred=p.argmax(axis=1)
    return dict(accuracy=float(accuracy_score(y,pred)),balanced_accuracy=float(balanced_accuracy_score(y,pred)),
                roc_auc=float(roc_auc_score(y,p,labels=[0,1,2],multi_class='ovr',average='macro')),
                log_loss=float(log_loss(y,p,labels=[0,1,2])),up_precision=float(precision_score(y,pred,labels=[1],average="macro",zero_division=0)),up_recall=float(recall_score(y,pred,labels=[1],average="macro",zero_division=0)),up_prediction_rate=float(np.mean(pred==1)),actual_up_rate=float(np.mean(y==1)),n_test=int(len(y)))

def choose_candidate(scores):
    """Select on inner accuracy, tie-break balanced accuracy; never use outer scores."""
    summary=scores.groupby('candidate',as_index=False).agg(accuracy=('accuracy','mean'),balanced_accuracy=('balanced_accuracy','mean'))
    summary=summary.sort_values(['accuracy','balanced_accuracy','candidate'],ascending=[False,False,True])
    return str(summary.iloc[0].candidate),summary

def run_nested_accuracy(frame, *, n_outer=3,n_inner=2,max_train_rows=25000,tree_train_rows=15000,seed=42,output_dir=None,target_q=7,balance_classes=True):
    if n_outer!=3: raise ValueError('Notebook 22 is configured for exactly three outer folds')
    if target_q != 7: raise ValueError('This experiment requires q=7')
    if not balance_classes: raise ValueError('This experiment requires class balancing')
    eligible,_=eligible_universe(frame)
    development,boundary=development_holdout_split(eligible,.2)
    expanded,base=add_historical_market_features(development)
    expanded,new=add_accuracy_features(expanded)
    schemes=feature_sets(base,new)
    # Require identical complete-case populations across every candidate, preventing selection by row filtering.
    allcols=schemes['all']
    inner_records=[];outer_records=[];selection=[]
    for outer in expanding_folds(expanded,n_folds=n_outer):
        # Inner model selection sees ONLY rows whose outcomes were known at outer.train_end.
        outer_history=expanded.loc[pd.to_datetime(expanded.bar_timestamp,utc=True)<=outer.train_end].copy()
        for inner in expanding_folds(outer_history,n_folds=n_inner):
            tr,te,xb,xv,_=fold_data(outer_history,inner,target_q=target_q)
            paired_a=tr[allcols].replace([np.inf,-np.inf],np.nan).notna().all(axis=1).to_numpy()
            paired_b=te[allcols].replace([np.inf,-np.inf],np.nan).notna().all(axis=1).to_numpy()
            tr=tr.loc[paired_a];te=te.loc[paired_b];xb=xb[paired_a];xv=xv[paired_b]
            if len(tr)<100 or len(te)<20:raise ValueError('Insufficient complete-case rows')
            for name,scheme,family,param in CANDIDATES:
                x,z,y,yt=_matrix(tr,te,xb,xv,schemes[scheme])
                rng=np.random.default_rng(seed+outer.number*100+inner.number)
                cap=max_train_rows if family=='logistic' else tree_train_rows
                ix=np.sort(rng.choice(len(y),min(cap,len(y)),replace=False))
                model=_model(family,param,seed);start=time.perf_counter();model.fit(x[ix],y[ix]);elapsed=time.perf_counter()-start
                inner_records.append(dict(outer_fold=outer.number,inner_fold=inner.number,candidate=name,fit_seconds=elapsed,majority_baseline_accuracy=float(np.mean(yt==np.bincount(y[ix],minlength=3).argmax())),**_score(yt,_predict(model,z))))
        subset=pd.DataFrame([r for r in inner_records if r['outer_fold']==outer.number])
        winner,ranking=choose_candidate(subset)
        name,scheme,family,param=next(c for c in CANDIDATES if c[0]==winner)
        selection.append(dict(outer_fold=outer.number,winner=winner,inner_accuracy=float(ranking.iloc[0].accuracy),outer_train_end=str(outer.train_end)))
        tr,te,xb,xv,_=fold_data(expanded,outer,target_q=target_q)
        paired_a=tr[allcols].replace([np.inf,-np.inf],np.nan).notna().all(axis=1).to_numpy()
        paired_b=te[allcols].replace([np.inf,-np.inf],np.nan).notna().all(axis=1).to_numpy()
        tr=tr.loc[paired_a];te=te.loc[paired_b];xb=xb[paired_a];xv=xv[paired_b]
        x,z,y,yt=_matrix(tr,te,xb,xv,schemes[scheme]);cap=max_train_rows if family=='logistic' else tree_train_rows
        rng=np.random.default_rng(seed+outer.number*1000);ix=np.sort(rng.choice(len(y),min(cap,len(y)),replace=False))
        model=_model(family,param,seed);start=time.perf_counter();model.fit(x[ix],y[ix]);elapsed=time.perf_counter()-start
        outer_records.append(dict(outer_fold=outer.number,winner=winner,fit_seconds=elapsed,train_rows=int(len(ix)),majority_baseline_accuracy=float(np.mean(yt==np.bincount(y[ix],minlength=3).argmax())),**_score(yt,_predict(model,z))))
        print(f'Outer fold {outer.number}/{n_outer}: selected {winner}; outer accuracy {outer_records[-1]["accuracy"]:.4f}',flush=True)
    inner_df=pd.DataFrame(inner_records);outer_df=pd.DataFrame(outer_records);selection_df=pd.DataFrame(selection)
    if output_dir is not None:
        out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
        inner_df.to_csv(out/'inner_candidate_scores.csv',index=False)
        outer_df.to_csv(out/'outer_unbiased_scores.csv',index=False)
        selection_df.to_csv(out/'outer_selections.csv',index=False)
        (out/'manifest.json').write_text(json.dumps({'outer_folds':n_outer,'inner_folds':n_inner,'development_cutoff':str(boundary),'historical_holdout':'NOT_ACCESSED','prospective_ledger':'NOT_ACCESSED','selection_metric':'inner_accuracy','target_q':7,'septile_labels':[2,2,0,0,1,1,1],'class_weight':'balanced','candidates':[c[0] for c in CANDIDATES],'warning':'Outer scores estimate nested selection performance; do not select hyperparameters from them.'},indent=2))
    return inner_df,outer_df,selection_df,boundary
