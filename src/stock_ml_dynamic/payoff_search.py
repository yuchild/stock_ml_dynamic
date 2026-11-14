"""Notebook 38: zero-cost, chronological, multi-model payoff and ranking research."""
from pathlib import Path
import time
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor, HistGradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import roc_auc_score, r2_score
from sklearn.preprocessing import StandardScaler
from .walk_forward import expanding_folds, fold_data
from .rf_deep_research import probability_matrix
from .economic_target_research import prepare_extras

MODELS=('direction_rf','direction_extra_trees','positive_hgb','positive_logistic','return_hgb','return_rf','return_extra_trees','return_ridge','random_control','oracle_upper_bound')

def payoff_stats(returns):
    """Realized gross-return statistics; no hypothetical stop/target fills."""
    x=np.asarray(returns,dtype=float);x=x[np.isfinite(x)]
    if not len(x):return dict(n=0,win_rate=np.nan,avg_win=np.nan,avg_loss=np.nan,payoff_ratio=np.nan,profit_factor=np.nan,expectancy=np.nan,breakeven_win_rate=np.nan,meets_2_to_1=False)
    w=x[x>0];l=x[x<0];aw=w.mean() if len(w) else np.nan;al=-l.mean() if len(l) else np.nan
    ratio=aw/al if len(w) and len(l) else np.nan
    pf=w.sum()/(-l.sum()) if len(l) and len(w) else np.nan
    return dict(n=len(x),win_rate=len(w)/len(x),avg_win=aw,avg_loss=al,payoff_ratio=ratio,profit_factor=pf,expectancy=x.mean(),breakeven_win_rate=1/(1+ratio) if np.isfinite(ratio) else np.nan,meets_2_to_1=bool(np.isfinite(ratio) and ratio>=2))

def daily_top_k(frame,score,k,model,fold):
    """Ex-ante rank each day's candidates, no future return used in selection."""
    x=frame[['bar_timestamp','symbol','next_return']].copy().reset_index(drop=True)
    x['score']=np.asarray(score,dtype=float);x['score']=x.score.replace([np.inf,-np.inf],np.nan).fillna(-1e20)
    x=x.sort_values(['bar_timestamp','score','symbol'],ascending=[True,False,True])
    x['rank']=x.groupby('bar_timestamp').cumcount()+1
    x['selected']=x['rank']<=k;x['model']=model;x['fold']=fold;x['k']=k
    return x

def date_block_ci(daily_diff,block=10,reps=400,seed=42):
    a=np.asarray(daily_diff,float);a=a[np.isfinite(a)]
    if len(a)<2:return (np.nan,np.nan)
    rng=np.random.default_rng(seed);n=len(a);b=min(block,n);res=np.empty(reps)
    for i in range(reps):
        ids=[]
        while len(ids)<n:
            start=int(rng.integers(0,n-b+1));ids.extend(range(start,start+b))
        res[i]=a[np.asarray(ids[:n])].mean()
    return tuple(np.quantile(res,[.025,.975]))

def _ic_by_date(x):
    vals=[]
    for _,g in x.groupby('bar_timestamp'):
        if len(g)>=8 and g.score.nunique()>1 and g.next_return.nunique()>1:
            v=spearmanr(g.score,g.next_return).statistic
            if np.isfinite(v):vals.append(v)
    return np.asarray(vals)

def run_payoff_search(frame,feature_cols,*,folds=3,cap=60000,trees=180,positions=(5,10,20),seed=42,output_dir=None):
    """No final holdout: caller must supply development-only frame."""
    if folds!=3:raise ValueError('Three predeclared development folds required')
    metrics=[];trades=[];daily=[];audit=[];timing=[];ic=[]
    for f in expanding_folds(frame,n_folds=folds):
        tr,va,xb,xv,_=fold_data(frame,f,target_q=7)
        assert pd.to_datetime(tr.next_bar_timestamp,utc=True).max()<=f.train_end
        assert pd.to_datetime(va.bar_timestamp,utc=True).min()>f.train_end
        ax,az=prepare_extras(tr,va,feature_cols)
        X=np.column_stack([xb.astype('float32'),ax]);Z=np.column_stack([xv.astype('float32'),az])
        rng=np.random.default_rng(seed+f.number)
        idx=np.sort(rng.choice(len(X),min(cap,len(X)),replace=False));X=X[idx]
        y=tr.direction.to_numpy(dtype=int)[idx];r=tr.next_return.to_numpy(float)[idx]
        qlo,qhi=np.quantile(r,[.005,.995]);rc=np.clip(r,qlo,qhi)
        spec={
          'direction_rf':(RandomForestClassifier(n_estimators=trees,max_depth=12,min_samples_leaf=20,n_jobs=-1,random_state=seed+f.number),'direction'),
          'direction_extra_trees':(ExtraTreesClassifier(n_estimators=trees,max_depth=16,min_samples_leaf=12,n_jobs=-1,random_state=seed+f.number),'direction'),
          'positive_hgb':(HistGradientBoostingClassifier(max_iter=180,max_leaf_nodes=15,l2_regularization=1,learning_rate=.05,random_state=seed+f.number),'binary'),
          'positive_logistic':(LogisticRegression(max_iter=250,solver='lbfgs',C=.1),'binary_scaled'),
          'return_hgb':(HistGradientBoostingRegressor(max_iter=180,max_leaf_nodes=15,l2_regularization=1,learning_rate=.05,random_state=seed+f.number),'regression'),
          'return_rf':(RandomForestRegressor(n_estimators=trees,max_depth=12,min_samples_leaf=20,n_jobs=-1,random_state=seed+f.number),'regression'),
          'return_extra_trees':(ExtraTreesRegressor(n_estimators=trees,max_depth=16,min_samples_leaf=12,n_jobs=-1,random_state=seed+f.number),'regression'),
          'return_ridge':(Ridge(alpha=100.),'regression_scaled')}
        scores={};
        for name,(model,kind) in spec.items():
            t=time.perf_counter()
            xx,zz=X,Z
            if kind.endswith('_scaled'):
                scaler=StandardScaler();xx=scaler.fit_transform(X);zz=scaler.transform(Z)
            target=y if kind=='direction' else ((r>0).astype(int) if kind.startswith('binary') else rc)
            model.fit(xx,target)
            if kind=='direction':
                probs=probability_matrix(model,zz);score=probs[:,1]
            elif kind.startswith('binary'):
                score=model.predict_proba(zz)[:,list(model.classes_).index(1)]
            else:score=model.predict(zz)
            scores[name]=np.asarray(score,float)
            rec=dict(fold=f.number,model=name,fit_seconds=time.perf_counter()-t)
            if kind.startswith('regression'):rec['r2']=r2_score(va.next_return,score)
            if kind.startswith('binary'):
                try:rec['binary_auc']=roc_auc_score((va.next_return>0).astype(int),score)
                except ValueError:rec['binary_auc']=np.nan
            timing.append(rec)
        # Negative control is independent random scores; oracle is explicitly NON-deployable.
        scores['random_control']=np.random.default_rng(seed+1000+f.number).random(len(va))
        scores['oracle_upper_bound']=va.next_return.to_numpy(float)
        for name,score in scores.items():
            ranked=daily_top_k(va,score,max(positions),name,f.number)
            vals=_ic_by_date(ranked)
            ic.append(dict(fold=f.number,model=name,mean_daily_spearman_ic=vals.mean() if len(vals) else np.nan,ic_days=len(vals),ic_positive_fraction=(vals>0).mean() if len(vals) else np.nan))
            for k in positions:
                chosen=ranked[ranked['rank']<=k].copy();chosen['k']=k
                trades.append(chosen[['fold','model','k','bar_timestamp','symbol','next_return','score','rank']])
                d=chosen.groupby('bar_timestamp',sort=True).next_return.mean()
                # Daily equal-weight portfolio among selected names; all k slots filled if enough names.
                for date,value in d.items():daily.append(dict(fold=f.number,model=name,k=k,bar_timestamp=date,daily_return=value))
                p=payoff_stats(chosen.next_return)
                p.update(dict(fold=f.number,model=name,k=k,mean_daily_return=d.mean(),daily_volatility=d.std(),days=len(d),compounded_return=np.prod(1+d)-1,max_drawdown=((1+d).cumprod()/(1+d).cumprod().cummax()-1).min()))
                metrics.append(p)
        audit.append(dict(fold=f.number,train_rows=len(tr),validation_rows=len(va),sampled_train=len(idx),last_train_label=pd.to_datetime(tr.next_bar_timestamp,utc=True).max(),first_validation_bar=pd.to_datetime(va.bar_timestamp,utc=True).min()))
    result=dict(metrics=pd.DataFrame(metrics),trades=pd.concat(trades,ignore_index=True),daily=pd.DataFrame(daily),ic=pd.DataFrame(ic),timing=pd.DataFrame(timing),audit=pd.DataFrame(audit))
    pairs=[]
    for (fold,k),g in result['daily'].groupby(['fold','k']):
        p=g.pivot(index='bar_timestamp',columns='model',values='daily_return')
        for name in MODELS:
            if name=='direction_rf':continue
            delta=(p[name]-p['direction_rf']).dropna();lo,hi=date_block_ci(delta,seed=seed+fold)
            pairs.append(dict(fold=fold,k=k,challenger=name,mean_daily_lift=delta.mean(),ci_low=lo,ci_high=hi,days=len(delta)))
    result['paired']=pd.DataFrame(pairs)
    if output_dir:
        dest=Path(output_dir);dest.mkdir(parents=True,exist_ok=True)
        for key,df in result.items():df.to_csv(dest/f'{key}.csv',index=False)
    return result
