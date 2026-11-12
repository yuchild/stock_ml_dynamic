"""Development-only paired feature benchmark; model selection prioritizes accuracy."""
from __future__ import annotations
from pathlib import Path
import json,time
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score,balanced_accuracy_score,roc_auc_score,log_loss
from .classifier_benchmark import eligible_universe
from .historical_market import add_historical_market_features
from .accuracy_features import add_accuracy_features,selective_metrics
from .feature_ablation import development_holdout_split
from .walk_forward import expanding_folds,fold_data
from sklearn.preprocessing import StandardScaler


def run_accuracy_benchmark(frame,*,n_folds=2,max_train_rows=30000,tree_train_rows=18000,seed=42,output_dir=None):
    eligible,_=eligible_universe(frame)
    development,boundary=development_holdout_split(eligible,.2)
    expanded,base=add_historical_market_features(development)
    expanded,new=add_accuracy_features(expanded)
    # Fair paired sample: both feature sets use the same rows and labels.
    all_features=list(dict.fromkeys(base+new))
    records=[];selected=[]
    for fold in expanding_folds(expanded,n_folds=n_folds):
        tr,te,xb,xv,_=fold_data(expanded,fold)
        a=tr[all_features].replace([np.inf,-np.inf],np.nan)
        b=te[all_features].replace([np.inf,-np.inf],np.nan)
        good_a=a.notna().all(axis=1).to_numpy();good_b=b.notna().all(axis=1).to_numpy()
        if good_a.sum()<100 or good_b.sum()<20:raise ValueError('Too few complete feature rows')
        y=tr.direction.to_numpy(dtype=int)[good_a];yt=te.direction.to_numpy(dtype=int)[good_b]
        rng=np.random.default_rng(seed+fold.number)
        ix=np.sort(rng.choice(len(y),min(max_train_rows,len(y)),replace=False))
        for scheme,cols in [('historical',base),('historical_plus_volatility',base+[x for x in new if 'volatility' in x or 'downside' in x or 'return_z' in x]),('historical_plus_all',all_features)]:
            scaler=StandardScaler().fit(a.loc[good_a,cols]);
            x=np.column_stack([xb[good_a],scaler.transform(a.loc[good_a,cols])]);z=np.column_stack([xv[good_b],scaler.transform(b.loc[good_b,cols])])
            for name in ('logistic_l2','random_forest','hist_boosting'):
                if name=='logistic_l2':model=LogisticRegression(C=.1,max_iter=400)
                elif name=='random_forest':model=RandomForestClassifier(n_estimators=160,max_depth=12,min_samples_leaf=30,n_jobs=-1,random_state=seed)
                else:model=HistGradientBoostingClassifier(max_iter=150,max_leaf_nodes=15,learning_rate=.05,l2_regularization=1,random_state=seed)
                fitix=ix if name=='logistic_l2' else ix[:min(len(ix),tree_train_rows)]
                t=time.perf_counter();model.fit(x[fitix],y[fitix]);elapsed=time.perf_counter()-t
                p=np.zeros((len(yt),3));raw=model.predict_proba(z)
                for j,c in enumerate(model.classes_):p[:,int(c)]=raw[:,j]
                pred=p.argmax(axis=1)
                records.append(dict(fold=fold.number,scheme=scheme,model=name,accuracy=accuracy_score(yt,pred),balanced_accuracy=balanced_accuracy_score(yt,pred),roc_auc=roc_auc_score(yt,p,labels=[0,1,2],multi_class='ovr',average='macro'),log_loss=log_loss(yt,p,labels=[0,1,2]),fit_seconds=elapsed,fit_rows=len(fitix),test_rows=len(yt)))
                if name in ('logistic_l2','random_forest'):
                    s=selective_metrics(yt,p);s['fold']=fold.number;s['scheme']=scheme;s['model']=name;selected.append(s)
        print(f'Completed development fold {fold.number}/{n_folds}',flush=True)
    scores=pd.DataFrame(records);selective=pd.concat(selected,ignore_index=True)
    summary=scores.groupby(['scheme','model'],as_index=False).agg(accuracy=('accuracy','mean'),balanced_accuracy=('balanced_accuracy','mean'),roc_auc=('roc_auc','mean'),log_loss=('log_loss','mean'),fit_seconds=('fit_seconds','mean')).sort_values(['accuracy','balanced_accuracy'],ascending=False)
    if output_dir:
        out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
        scores.to_csv(out/'fold_scores.csv',index=False);summary.to_csv(out/'summary.csv',index=False);selective.to_csv(out/'selective_accuracy.csv',index=False)
        (out/'manifest.json').write_text(json.dumps({'development_boundary':str(boundary),'holdout':'NOT_ACCESSED','prospective_ledger':'NOT_ACCESSED','selection_metric':'accuracy','feature_sets':['historical','historical_plus_volatility','historical_plus_all'],'note':'Threshold curves are diagnostic only; select thresholds in nested training, not evaluation folds.'},indent=2))
    return scores,summary,selective,boundary
