"""Development-only comparison of next-bar target definitions.

Tertiles use training-only cutoffs; volatility-adjusted neutral labels use the
*current* bar's trailing volatility (known at prediction time). Binary labels
exclude exact zero returns and record how many observations were removed.
The final 20% chronological holdout is never evaluated here.
"""
from __future__ import annotations
from pathlib import Path
import json, time
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, roc_curve, log_loss, confusion_matrix
from .classifier_benchmark import eligible_universe
from .historical_market import add_historical_market_features
from .feature_ablation import development_holdout_split
from .walk_forward import expanding_folds, fold_data

TARGETS = ('tertiles', 'volatility_adjusted', 'binary_direction')
MODELS = ('prior_baseline', 'logistic_l2', 'logistic_l1')


def make_labels(target, train_returns, test_returns, train_vol, test_vol, *,
                volatility_multiplier=0.5):
    """Return labels for training/test, with no test-derived thresholds.

    Class IDs: three-class 2=down, 0=neutral, 1=up; binary 0=down, 1=up.
    Binary zero-return rows are excluded, not arbitrarily assigned a sign.
    """
    tr=np.asarray(train_returns,dtype=float); te=np.asarray(test_returns,dtype=float)
    vt=np.asarray(train_vol,dtype=float); ve=np.asarray(test_vol,dtype=float)
    if target == 'tertiles':
        q=np.quantile(tr,[1/3,2/3]);
        if q[0]>=q[1]: raise ValueError('Degenerate training tertiles')
        def label(r): return np.where(r<q[0],2,np.where(r>q[1],1,0))
        return label(tr),label(te),np.ones(len(tr),bool),np.ones(len(te),bool),{'cutoffs':q.tolist()}
    if target == 'volatility_adjusted':
        if volatility_multiplier<=0: raise ValueError('volatility_multiplier must be positive')
        def label(r,v):
            threshold=volatility_multiplier*v
            return np.where(r < -threshold,2,np.where(r > threshold,1,0))
        return label(tr,vt),label(te,ve),np.ones(len(tr),bool),np.ones(len(te),bool),{'volatility_multiplier':volatility_multiplier,'volatility_window':20}
    if target == 'binary_direction':
        keep_tr=tr!=0; keep_te=te!=0
        return (tr>0).astype(int),(te>0).astype(int),keep_tr,keep_te,{'zero_return_policy':'exclude'}
    raise ValueError(target)


def _model(name):
    if name=='prior_baseline': return DummyClassifier(strategy='prior')
    if name=='logistic_l2': return LogisticRegression(C=.1,max_iter=500,solver='lbfgs')
    if name=='logistic_l1': return LogisticRegression(C=.1,l1_ratio=1,max_iter=500,solver='saga',tol=.002,random_state=42)
    raise ValueError(name)


def _probability(model,x,labels):
    raw=model.predict_proba(x)
    p=np.zeros((len(x),len(labels)),dtype=float)
    for j,c in enumerate(model.classes_): p[:,labels.index(int(c))]=raw[:,j]
    return p


def evaluate_predictions(y, p, probability, labels):
    labels=list(labels)
    if len(labels)==2:
        try: auc=roc_auc_score(y,probability[:,labels.index(1)])
        except ValueError: auc=np.nan
    else:
        try: auc=roc_auc_score(y,probability,labels=labels,multi_class='ovr',average='macro')
        except ValueError: auc=np.nan
    return dict(accuracy=float(accuracy_score(y,p)),balanced_accuracy=float(balanced_accuracy_score(y,p)),
                roc_auc=float(auc),log_loss=float(log_loss(y,probability,labels=labels)))


def run_target_diagnostics(frame, *, n_folds=5, holdout_fraction=.2, max_train_rows=120000,
                           selected_k=4,cluster_fit_limit=15000,volatility_multiplier=.5,
                           output_dir=None,models=MODELS,targets=TARGETS):
    eligible,eligibility=eligible_universe(frame)
    expanded,extra=add_historical_market_features(eligible)
    dev,boundary=development_holdout_split(expanded,holdout_fraction)
    out=Path(output_dir) if output_dir else None
    if out:
        out.mkdir(parents=True,exist_ok=True)
        eligibility.to_csv(out/'universe_eligibility.csv',index=False)
        (out/'holdout_lock.json').write_text(json.dumps({'status':'NOT_EVALUATED','development_end':str(boundary),
            'holdout_fraction':holdout_fraction,'folds':n_folds,'volatility_multiplier':volatility_multiplier,
            'warning':'Targets differ: compare improvements against target-specific baselines; do not compare raw accuracies across targets.'},indent=2))
    rows=[];curves=[];prevalence=[];confusions=[]
    for fold in expanding_folds(dev,n_folds=n_folds):
        t0=time.perf_counter()
        train,test,base_train,base_test,_=fold_data(dev,fold,selected_k=selected_k,cluster_fit_limit=cluster_fit_limit)
        et=train[extra].replace([np.inf,-np.inf],np.nan)
        ev=test[extra].replace([np.inf,-np.inf],np.nan)
        good_tr=et.notna().all(axis=1).to_numpy();good_te=ev.notna().all(axis=1).to_numpy()
        if good_tr.sum()<100 or good_te.sum()<20: raise ValueError('Insufficient complete paired rows')
        tr=train.loc[good_tr];te=test.loc[good_te]
        et=et.loc[good_tr];ev=ev.loc[good_te]
        scaler=StandardScaler().fit(et)
        # Use identical full historical/market features for all targets; remove 10-bar vol
        # based on prior development diagnostics. All transforms are training-only.
        cols=[c for c in extra if c!='realized_vol_10']
        xt=np.column_stack([base_train[good_tr],pd.DataFrame(scaler.transform(et),columns=extra)[cols].to_numpy()])
        xv=np.column_stack([base_test[good_te],pd.DataFrame(scaler.transform(ev),columns=extra)[cols].to_numpy()])
        prep=time.perf_counter()-t0
        print(f'Fold {fold.number}/{n_folds}: train={len(tr):,} test={len(te):,} preprocess={prep:.1f}s',flush=True)
        for target in targets:
            ytr,yte,ktr,kte,definition=make_labels(target,tr.next_return,te.next_return,
                et.realized_vol_20,ev.realized_vol_20,volatility_multiplier=volatility_multiplier)
            labels=[0,1] if target=='binary_direction' else [0,1,2]
            xtrain=xt[ktr];xtest=xv[kte];ytrain=ytr[ktr];ytest=yte[kte]
            if len(np.unique(ytrain))!=len(labels):raise ValueError(f'{target}: missing training class')
            if max_train_rows and len(ytrain)>max_train_rows:
                idx=np.sort(np.random.default_rng(42+fold.number).choice(len(ytrain),max_train_rows,replace=False))
            else:idx=np.arange(len(ytrain))
            for cls in labels:
                prevalence.append({'fold':fold.number,'target':target,'class_id':cls,
                    'train_share':float(np.mean(ytrain==cls)),'test_share':float(np.mean(ytest==cls))})
            for name in models:
                model=_model(name)
                start=time.perf_counter();model.fit(xtrain[idx],ytrain[idx]);fit=time.perf_counter()-start
                start=time.perf_counter();p=model.predict(xtest);prob=_probability(model,xtest,labels);predict=time.perf_counter()-start
                metrics=evaluate_predictions(ytest,p,prob,labels)
                rows.append({'fold':fold.number,'target':target,'model':name,'train_rows':len(ytrain),
                    'fit_rows':len(idx),'test_rows':len(ytest),'zero_train_excluded':int((~ktr).sum()),
                    'zero_test_excluded':int((~kte).sum()),'features':xt.shape[1],
                    'preprocess_seconds':prep,'fit_seconds':fit,'predict_seconds':predict,
                    'train_end':str(fold.train_end),'test_end':str(fold.test_end),**metrics})
                cm=confusion_matrix(ytest,p,labels=labels)
                for i,a in enumerate(labels):
                    for j,b in enumerate(labels):
                        confusions.append({'fold':fold.number,'target':target,'model':name,
                                           'actual':a,'predicted':b,'count':int(cm[i,j])})
                for cls in labels:
                    truth=(ytest==cls).astype(int)
                    if len(np.unique(truth))==2:
                        fpr,tpr,_=roc_curve(truth,prob[:,labels.index(cls)])
                        curves.append(pd.DataFrame({'fold':fold.number,'target':target,'model':name,
                            'class_id':cls,'fpr':fpr,'tpr':tpr}))
                print(f'  {target:20} {name:15} acc={metrics["accuracy"]:.4f} '
                      f'auc={metrics["roc_auc"]:.4f} fit={fit:.1f}s',flush=True)
                if out:pd.DataFrame(rows).to_csv(out/'fold_metrics.csv',index=False)
    scores=pd.DataFrame(rows);roc=pd.concat(curves,ignore_index=True) if curves else pd.DataFrame()
    shares=pd.DataFrame(prevalence);cm=pd.DataFrame(confusions)
    if out:
        scores.to_parquet(out/'fold_metrics.parquet',index=False)
        roc.to_parquet(out/'roc_curves.parquet',index=False)
        shares.to_csv(out/'class_prevalence.csv',index=False)
        cm.to_csv(out/'confusion_matrices.csv',index=False)
        summary_targets(scores).to_csv(out/'summary.csv',index=False)
    return scores,roc,shares,cm,eligibility,boundary


def summary_targets(scores):
    summary=scores.groupby(['target','model'],as_index=False).agg(
        accuracy=('accuracy','mean'),accuracy_sd=('accuracy','std'),
        balanced_accuracy=('balanced_accuracy','mean'),auc=('roc_auc','mean'),
        auc_sd=('roc_auc','std'),log_loss=('log_loss','mean'),
        fit_seconds=('fit_seconds','sum'),predict_seconds=('predict_seconds','sum'))
    return summary.sort_values(['target','auc'],ascending=[True,False])


def plot_target_diagnostics(scores,curves,shares,output_dir=None):
    import matplotlib.pyplot as plt
    out=Path(output_dir) if output_dir else None
    summary=summary_targets(scores)
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for i,metric in enumerate(('accuracy','auc')):
        pivot=summary.pivot(index='target',columns='model',values=metric)
        pivot.plot.bar(ax=axes[i]);axes[i].set_title(('CV accuracy','CV ROC-AUC')[i]);axes[i].set_ylabel(metric)
        axes[i].tick_params(axis='x',rotation=20)
    fig.tight_layout()
    if out:fig.savefig(out/'target_comparison.png',dpi=160)
    plt.show()
    grid=np.linspace(0,1,201)
    for target in scores.target.unique():
        fig,ax=plt.subplots(figsize=(7,6))
        for model in scores.model.unique():
            sub=curves[(curves.target==target)&(curves.model==model)]
            folds=[]
            for _,fd in sub.groupby('fold'):
                class_curves=[np.interp(grid,c.fpr,c.tpr) for _,c in fd.groupby('class_id')]
                if class_curves:folds.append(np.mean(class_curves,axis=0))
            if folds:ax.plot(grid,np.mean(folds,axis=0),label=model)
        ax.plot([0,1],[0,1],'k--',alpha=.5)
        ax.set(xlabel='False positive rate',ylabel='True positive rate',title=f'{target}: mean OVR ROC (development folds)')
        ax.legend();fig.tight_layout()
        if out:fig.savefig(out/f'roc_{target}.png',dpi=160)
        plt.show()
    return summary
