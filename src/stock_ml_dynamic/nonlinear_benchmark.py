"""Notebook 16: leakage-controlled ensemble and custom PyTorch development benchmark.

Never reads the prospective ledger or evaluates the already-opened final holdout.
"""
from __future__ import annotations
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, log_loss
from .classifier_benchmark import eligible_universe
from .historical_market import add_historical_market_features
from .feature_ablation import development_holdout_split
from .regularization_diagnostics import matrices, sample_rows
from .walk_forward import expanding_folds

CLASSES=np.array([0,1,2])

def aligned_proba(model, X):
    raw=model.predict_proba(X)
    out=np.zeros((len(X),3),dtype=float)
    for i,c in enumerate(model.classes_):out[:,int(c)]=raw[:,i]
    return out

def score_probabilities(y,probs):
    y=np.asarray(y,dtype=int);p=np.asarray(probs,dtype=float)
    if p.shape!=(len(y),3) or not np.isfinite(p).all() or (p<0).any():
        raise ValueError('Invalid three-class probabilities')
    if not np.allclose(p.sum(axis=1),1,atol=1e-5):raise ValueError('Probabilities must sum to one')
    pred=CLASSES[p.argmax(axis=1)]
    return {'accuracy':float(accuracy_score(y,pred)),
            'balanced_accuracy':float(balanced_accuracy_score(y,pred)),
            'roc_auc':float(roc_auc_score(y,p,labels=CLASSES,multi_class='ovr',average='macro')) if len(np.unique(y))==3 else float('nan'),
            'log_loss':float(log_loss(y,p,labels=CLASSES))}

def build_estimators(seed=42, n_jobs=-1):
    return {
        'prior_baseline':DummyClassifier(strategy='prior'),
        'logistic_l2_fixed':LogisticRegression(C=.1,max_iter=600,solver='lbfgs'),
        'random_forest':RandomForestClassifier(n_estimators=160,max_depth=12,min_samples_leaf=30,max_features='sqrt',n_jobs=n_jobs,random_state=seed),
        'extra_trees':ExtraTreesClassifier(n_estimators=160,max_depth=12,min_samples_leaf=30,max_features='sqrt',n_jobs=n_jobs,random_state=seed),
        'hist_gradient_boosting':HistGradientBoostingClassifier(max_iter=150,max_leaf_nodes=15,learning_rate=.05,l2_regularization=1,early_stopping=False,random_state=seed),
    }

def _torch_model(n_features,hidden,dropout):
    import torch
    from torch import nn
    return nn.Sequential(nn.Linear(n_features,hidden),nn.ReLU(),nn.Dropout(dropout),
                         nn.Linear(hidden,hidden//2),nn.ReLU(),nn.Dropout(dropout),
                         nn.Linear(hidden//2,3))

def train_custom_mlp(X,y,X_val,y_val,*,seed=42,epochs=35,batch_size=2048,hidden=128,dropout=.20,
                     lr=1e-3,weight_decay=1e-3,patience=6,device='auto'):
    """Fresh neural net, explicit minibatch training, AdamW, validation early stopping.

    The supplied validation must be chronologically earlier than the evaluation fold.
    """
    import torch
    from torch import nn
    from torch.utils.data import DataLoader,TensorDataset
    torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch_device='cuda' if device=='auto' and torch.cuda.is_available() else ('cpu' if device=='auto' else device)
    net=_torch_model(X.shape[1],hidden,dropout).to(torch_device)
    opt=torch.optim.AdamW(net.parameters(),lr=lr,weight_decay=weight_decay)
    loss_fn=nn.CrossEntropyLoss()
    loader=DataLoader(TensorDataset(torch.as_tensor(np.asarray(X,dtype=np.float32)),torch.as_tensor(np.asarray(y,dtype=np.int64))),
                      batch_size=batch_size,shuffle=True,num_workers=0)
    xv=torch.as_tensor(np.asarray(X_val,dtype=np.float32),device=torch_device)
    yv=torch.as_tensor(np.asarray(y_val,dtype=np.int64),device=torch_device)
    best=float('inf');best_state=None;wait=0;history=[]
    for epoch in range(1,epochs+1):
        net.train();total=0.;count=0
        for xb,yb in loader:
            xb=xb.to(torch_device);yb=yb.to(torch_device)
            opt.zero_grad(set_to_none=True)
            loss=loss_fn(net(xb),yb)
            loss.backward();opt.step()
            total+=float(loss.detach())*len(yb);count+=len(yb)
        net.eval()
        with torch.no_grad():val_loss=float(loss_fn(net(xv),yv))
        history.append({'epoch':epoch,'train_loss':total/count,'val_loss':val_loss})
        if val_loss<best-1e-5:
            best=val_loss;best_state={k:v.detach().cpu().clone() for k,v in net.state_dict().items()};wait=0
        else:
            wait+=1
            if wait>=patience:break
    net.load_state_dict(best_state);net.eval()
    return net,torch_device,pd.DataFrame(history)

def predict_custom_mlp(net,X,device,batch_size=8192):
    import torch
    probs=[]
    net.eval()
    with torch.no_grad():
        for start in range(0,len(X),batch_size):
            xb=torch.as_tensor(np.asarray(X[start:start+batch_size],dtype=np.float32),device=device)
            probs.append(torch.softmax(net(xb),dim=1).cpu().numpy())
    return np.vstack(probs)

def chronological_neural_split(train_rows,*,validation_fraction=.15):
    """Split training by outcome availability, not row order or random shuffle."""
    timestamps=pd.to_datetime(train_rows['next_bar_timestamp'],utc=True)
    unique=pd.DatetimeIndex(timestamps.unique()).sort_values()
    if len(unique)<20:raise ValueError('Too few timestamps for neural validation')
    cutoff=unique[max(0,int(len(unique)*(1-validation_fraction))-1)]
    is_fit=(timestamps<=cutoff).to_numpy()
    if is_fit.sum()<100 or (~is_fit).sum()<30:raise ValueError('Insufficient neural train/validation rows')
    return is_fit,cutoff

def run_nonlinear_benchmark(master,*,n_folds=5,max_train_rows=60000,tree_train_rows=45000,
                            include_neural=True,neural_epochs=35,seed=42,output_dir=None):
    """Run fixed-config models on development-only expanding folds.

    Each fold refits feature transformations and target quantiles on past data.
    The neural early-stop subset is strictly within that fold's training period.
    """
    if n_folds<1 or max_train_rows<500 or tree_train_rows<500:raise ValueError('Invalid training configuration')
    eligible,eligibility=eligible_universe(master)
    expanded,extra=add_historical_market_features(eligible)
    dev,boundary=development_holdout_split(expanded,.2)
    output=Path(output_dir) if output_dir is not None else None
    if output:output.mkdir(parents=True,exist_ok=True)
    results=[];histories=[]
    for fold in expanding_folds(dev,n_folds=n_folds):
        X,y,Z,yt,tr,te=matrices(dev,fold,extra)
        # Sample once, with corresponding timestamps retained for neural early stopping.
        if len(y)>max_train_rows:
            idx=np.sort(np.random.default_rng(seed+fold.number).choice(len(y),max_train_rows,replace=False))
            X=X[idx];y=y[idx];tr=tr.iloc[idx].copy()
        print(f'Fold {fold.number}/{n_folds}: train={len(y):,} test={len(yt):,}',flush=True)
        for name,model in build_estimators(seed+fold.number).items():
            if name in ('random_forest','extra_trees','hist_gradient_boosting') and len(y)>tree_train_rows:
                idx=np.sort(np.random.default_rng(seed+100+fold.number).choice(len(y),tree_train_rows,replace=False))
                xfit,yfit=X[idx],y[idx]
            else:xfit,yfit=X,y
            start=time.perf_counter();model.fit(xfit,yfit);fit_s=time.perf_counter()-start
            start=time.perf_counter();probs=aligned_proba(model,Z);infer_s=time.perf_counter()-start
            row={'fold':fold.number,'model':name,'train_rows':len(yfit),'test_rows':len(yt),
                 'train_end':str(fold.train_end),'test_end':str(fold.test_end),
                 'fit_seconds':fit_s,'predict_seconds':infer_s,**score_probabilities(yt,probs)}
            results.append(row)
            print(f'  {name}: accuracy={row["accuracy"]:.4f}, auc={row["roc_auc"]:.4f}, fit={fit_s:.1f}s',flush=True)
        if include_neural:
            # IMPORTANT: matrices() fits its preprocessing on the entire outer train,
            # including neural early-stop rows. These rows are used only for early
            # stopping, not an unbiased neural validation estimate.
            mask,cutoff=chronological_neural_split(tr)
            start=time.perf_counter()
            net,device,hist=train_custom_mlp(X[mask],y[mask],X[~mask],y[~mask],seed=seed+fold.number,epochs=neural_epochs)
            fit_s=time.perf_counter()-start
            start=time.perf_counter();probs=predict_custom_mlp(net,Z,device);infer_s=time.perf_counter()-start
            row={'fold':fold.number,'model':'custom_pytorch_mlp','train_rows':int(mask.sum()),'test_rows':len(yt),
                 'train_end':str(fold.train_end),'test_end':str(fold.test_end),
                 'fit_seconds':fit_s,'predict_seconds':infer_s,'device':device,
                 'early_stop_cutoff':str(cutoff),**score_probabilities(yt,probs)}
            results.append(row)
            hist['fold']=fold.number;histories.append(hist)
            print(f'  custom_pytorch_mlp: accuracy={row["accuracy"]:.4f}, auc={row["roc_auc"]:.4f}, fit={fit_s:.1f}s',flush=True)
        if output:pd.DataFrame(results).to_csv(output/'fold_metrics.csv',index=False)
    scores=pd.DataFrame(results)
    summary=summarize_benchmark(scores)
    if output:
        summary.to_csv(output/'summary.csv',index=False)
        eligibility.to_csv(output/'universe_eligibility.csv',index=False)
        if histories:pd.concat(histories,ignore_index=True).to_csv(output/'neural_training_history.csv',index=False)
        (output/'experiment_manifest.json').write_text(json.dumps({'status':'DEVELOPMENT_ONLY','holdout':'NOT_REUSED',
           'development_end':str(boundary),'folds':n_folds,'max_train_rows':max_train_rows,
           'tree_train_rows':tree_train_rows,'neural_epochs':neural_epochs,'seed':seed,
           'models':sorted(scores.model.unique().tolist()),'prospective_ledger':'NOT_TOUCHED'},indent=2))
    return scores,summary,eligibility,boundary

def summarize_benchmark(scores):
    return scores.groupby('model',as_index=False).agg(
        accuracy=('accuracy','mean'),accuracy_sd=('accuracy','std'),
        balanced_accuracy=('balanced_accuracy','mean'),roc_auc=('roc_auc','mean'),
        auc_sd=('roc_auc','std'),log_loss=('log_loss','mean'),
        total_fit_seconds=('fit_seconds','sum'),total_predict_seconds=('predict_seconds','sum')
    ).sort_values(['roc_auc','log_loss'],ascending=[False,True]).reset_index(drop=True)
