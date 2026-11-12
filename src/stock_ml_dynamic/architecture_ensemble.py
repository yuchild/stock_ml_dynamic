"""Notebook 17: development-only residual neural networks and temporal ensembles."""
from __future__ import annotations
import json,time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from .nonlinear_benchmark import aligned_proba,score_probabilities,summarize_benchmark,chronological_neural_split,predict_custom_mlp
from .classifier_benchmark import eligible_universe
from .historical_market import add_historical_market_features
from .feature_ablation import development_holdout_split
from .regularization_diagnostics import matrices
from .walk_forward import expanding_folds

CLASSES=np.array([0,1,2])

def normalize_probabilities(p):
    p=np.asarray(p,dtype=float)
    if p.ndim!=2 or p.shape[1]!=3 or not np.isfinite(p).all():raise ValueError('Expected finite (n,3) probabilities')
    p=np.clip(p,1e-9,None)
    return p/p.sum(axis=1,keepdims=True)

def blend_probabilities(predictions,weights):
    if len(predictions)!=len(weights) or not predictions:raise ValueError('Missing predictions or weights')
    w=np.asarray(weights,dtype=float)
    if not np.isfinite(w).all() or (w<0).any() or w.sum()<=0:raise ValueError('Invalid weights')
    shapes={np.asarray(p).shape for p in predictions}
    if len(shapes)!=1:raise ValueError('Probability shapes differ')
    return normalize_probabilities(sum(float(a)*normalize_probabilities(p) for a,p in zip(w/w.sum(),predictions)))

class ResidualMLP:
    """Fresh PyTorch residual network; no pretrained weights or sklearn MLP."""
    def __init__(self,n_features,hidden=128,dropout=.2):
        import torch
        from torch import nn
        class Network(nn.Module):
            def __init__(self):
                super().__init__()
                self.stem=nn.Sequential(nn.Linear(n_features,hidden),nn.LayerNorm(hidden),nn.GELU())
                self.block=nn.Sequential(nn.Linear(hidden,hidden),nn.GELU(),nn.Dropout(dropout),nn.Linear(hidden,hidden))
                self.norm=nn.LayerNorm(hidden)
                self.head=nn.Sequential(nn.GELU(),nn.Dropout(dropout),nn.Linear(hidden,3))
            def forward(self,x):
                z=self.stem(x)
                return self.head(self.norm(z+self.block(z)))
        self.net=Network()

def train_residual_mlp(x,y,xv,yv,*,seed=42,epochs=35,batch_size=2048,hidden=128,dropout=.2,lr=8e-4,weight_decay=.01,patience=6,device='auto'):
    import torch
    from torch import nn
    from torch.utils.data import TensorDataset,DataLoader
    torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    dev=('cuda' if torch.cuda.is_available() else 'cpu') if device=='auto' else device
    net=ResidualMLP(x.shape[1],hidden,dropout).net.to(dev)
    opt=torch.optim.AdamW(net.parameters(),lr=lr,weight_decay=weight_decay)
    train=TensorDataset(torch.as_tensor(np.asarray(x,dtype=np.float32)),torch.as_tensor(np.asarray(y,dtype=np.int64)))
    loader=DataLoader(train,batch_size=batch_size,shuffle=True,num_workers=0)
    vx=torch.as_tensor(np.asarray(xv,dtype=np.float32),device=dev)
    vy=torch.as_tensor(np.asarray(yv,dtype=np.int64),device=dev)
    best=float('inf');state=None;wait=0;history=[]
    for epoch in range(1,epochs+1):
        net.train();running=0.;n=0
        for bx,by in loader:
            bx=bx.to(dev);by=by.to(dev)
            opt.zero_grad(set_to_none=True)
            loss=nn.functional.cross_entropy(net(bx),by)
            loss.backward();nn.utils.clip_grad_norm_(net.parameters(),1.)
            opt.step();running+=loss.item()*len(by);n+=len(by)
        net.eval()
        with torch.no_grad():vl=nn.functional.cross_entropy(net(vx),vy).item()
        history.append({'epoch':epoch,'train_loss':running/n,'val_loss':vl})
        if vl<best-1e-5:
            best=vl;state={k:v.detach().cpu().clone() for k,v in net.state_dict().items()};wait=0
        else:
            wait+=1
            if wait>=patience:break
    net.load_state_dict(state);net.eval()
    return net,dev,pd.DataFrame(history)

def sample_aligned(x,y,rows,cap,seed):
    if cap and len(y)>cap:
        idx=np.sort(np.random.default_rng(seed).choice(len(y),cap,replace=False))
        return x[idx],y[idx],rows.iloc[idx].copy()
    return x,y,rows

def chronological_meta_indices(rows,fraction=.2):
    mask,cutoff=chronological_neural_split(rows,validation_fraction=fraction)
    return mask,cutoff

def run_architecture_ensemble(master,*,n_folds=5,max_train_rows=60000,tree_train_rows=45000,neural_epochs=35,seed=42,include_stacking=True,output_dir=None):
    """Temporal outer CV. Stacking meta-fit uses inner chronological OOF slices only.

    Feature preprocessing is fit on each outer training fold. Inner stacking fits
    reuse that outer-train-only preprocessing; inner validation therefore has
    preprocessing exposure, but never outer-test exposure. Inner stacking is
    exploratory, not an unbiased inner validation estimate.
    """
    if n_folds<1 or max_train_rows<500 or tree_train_rows<500:raise ValueError('Invalid fold/cap settings')
    eligible,eligibility=eligible_universe(master)
    expanded,extra=add_historical_market_features(eligible)
    dev,boundary=development_holdout_split(expanded,.2)
    out=Path(output_dir) if output_dir else None
    if out:out.mkdir(parents=True,exist_ok=True)
    rows=[];histories=[];oof=[]
    for fold in expanding_folds(dev,n_folds=n_folds):
        x,y,z,yt,tr,te=matrices(dev,fold,extra)
        x,y,tr=sample_aligned(x,y,tr,max_train_rows,seed+fold.number)
        models={
          'logistic_l2':LogisticRegression(C=.1,max_iter=600),
          'random_forest':RandomForestClassifier(n_estimators=160,max_depth=12,min_samples_leaf=30,max_features='sqrt',n_jobs=-1,random_state=seed+fold.number)
        }
        pred={};timings={}
        for name,model in models.items():
            xf,yf=(x,y)
            if name=='random_forest':xf,yf,_=sample_aligned(x,y,tr,tree_train_rows,seed+fold.number+100)
            start=time.perf_counter();model.fit(xf,yf);fit=time.perf_counter()-start
            start=time.perf_counter();pred[name]=normalize_probabilities(aligned_proba(model,z));infer=time.perf_counter()-start
            timings[name]=(fit,infer)
        mask,cutoff=chronological_neural_split(tr)
        start=time.perf_counter()
        net,device,history=train_residual_mlp(x[mask],y[mask],x[~mask],y[~mask],seed=seed+fold.number,epochs=neural_epochs)
        fit=time.perf_counter()-start
        start=time.perf_counter();pred['residual_mlp']=normalize_probabilities(predict_custom_mlp(net,z,device));infer=time.perf_counter()-start
        timings['residual_mlp']=(fit,infer)
        history['fold']=fold.number;histories.append(history)
        pred['soft_vote_lr_rf']=blend_probabilities([pred['logistic_l2'],pred['random_forest']],[.5,.5])
        pred['soft_vote_all']=blend_probabilities([pred['logistic_l2'],pred['random_forest'],pred['residual_mlp']],[1,1,1])
        timings['soft_vote_lr_rf']=(sum(timings[k][0] for k in ('logistic_l2','random_forest')),sum(timings[k][1] for k in ('logistic_l2','random_forest')))
        timings['soft_vote_all']=(sum(timings[k][0] for k in ('logistic_l2','random_forest','residual_mlp')),sum(timings[k][1] for k in ('logistic_l2','random_forest','residual_mlp')))
        if include_stacking:
            # Two strictly chronological inner OOF blocks; meta is trained on
            # predictions from models fit only on earlier observations.
            ts=pd.to_datetime(tr['next_bar_timestamp'],utc=True)
            unique=pd.DatetimeIndex(ts.unique()).sort_values()
            if len(unique)<40:raise ValueError('Not enough unique dates for stacking')
            boundaries=[(.50,.65),(.65,.80)]
            mx=[];my=[]
            for j,(a,b) in enumerate(boundaries):
                left=unique[int(len(unique)*a)-1];right=unique[int(len(unique)*b)-1]
                trainmask=(ts<=left).to_numpy();testmask=((ts>left)&(ts<=right)).to_numpy()
                if trainmask.sum()<100 or testmask.sum()<30:raise ValueError('Too few stacking observations')
                ip=[]
                for name in ('logistic_l2','random_forest'):
                    m=LogisticRegression(C=.1,max_iter=600) if name=='logistic_l2' else RandomForestClassifier(n_estimators=160,max_depth=12,min_samples_leaf=30,max_features='sqrt',n_jobs=-1,random_state=seed+fold.number+j)
                    xi,yi,_=sample_aligned(x[trainmask],y[trainmask],tr.loc[trainmask],tree_train_rows if name=='random_forest' else max_train_rows,seed+j+fold.number)
                    m.fit(xi,yi);ip.append(normalize_probabilities(aligned_proba(m,x[testmask])))
                mx.append(np.hstack(ip));my.append(y[testmask])
            meta=LogisticRegression(C=.1,max_iter=500)
            start=time.perf_counter();meta.fit(np.vstack(mx),np.concatenate(my));meta_fit=time.perf_counter()-start
            start=time.perf_counter();pred['temporal_stacking']=normalize_probabilities(aligned_proba(meta,np.hstack([pred['logistic_l2'],pred['random_forest']])));meta_infer=time.perf_counter()-start
            timings['temporal_stacking']=(meta_fit+timings['soft_vote_lr_rf'][0],meta_infer+timings['soft_vote_lr_rf'][1])
            oof.append({'fold':fold.number,'meta_rows':sum(len(v) for v in my),'meta_train_end':str(right),'outer_test_end':str(fold.test_end)})
        for name,p in pred.items():
            metrics=score_probabilities(yt,p)
            rows.append({'fold':fold.number,'model':name,'train_rows':len(y),'test_rows':len(yt),'train_end':str(fold.train_end),'test_end':str(fold.test_end),'fit_seconds':timings[name][0],'predict_seconds':timings[name][1],**metrics})
            print(f'Fold {fold.number}/{n_folds} {name}: AUC={metrics["roc_auc"]:.4f} accuracy={metrics["accuracy"]:.4f}',flush=True)
        if out:pd.DataFrame(rows).to_csv(out/'fold_metrics.csv',index=False)
    scores=pd.DataFrame(rows);summary=summarize_benchmark(scores)
    if out:
        summary.to_csv(out/'summary.csv',index=False)
        eligibility.to_csv(out/'universe_eligibility.csv',index=False)
        pd.concat(histories,ignore_index=True).to_csv(out/'residual_training_history.csv',index=False)
        if oof:pd.DataFrame(oof).to_csv(out/'stacking_oof_manifest.csv',index=False)
        (out/'experiment_manifest.json').write_text(json.dumps({'status':'DEVELOPMENT_ONLY','holdout':'NOT_REUSED','prospective_ledger':'NOT_TOUCHED','development_end':str(boundary),'folds':n_folds,'max_train_rows':max_train_rows,'tree_train_rows':tree_train_rows,'neural_epochs':neural_epochs,'stacking':include_stacking,'seed':seed,'note':'Exploratory outer-fold preprocessing is fit on outer train only; inner stacking shares outer training preprocessing.'},indent=2))
    return scores,summary,eligibility,boundary
