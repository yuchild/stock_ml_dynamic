"""Notebook 40: source-of-edge audit. Development-only, chronological, zero-cost.

Feature-group assignments are exploratory, and validation-only diagnostics must
never be used as deployment filters. No final historical holdout is accessed.
"""
from pathlib import Path
import re
import time
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from .walk_forward import expanding_folds, fold_data
from .economic_target_research import prepare_extras
from .payoff_search import payoff_stats, daily_top_k, date_block_ci

GROUP_PATTERNS = {
    'volume': r'volume|vol_|obv|money_flow|vwap|turnover|liquidity|dollar_vol|accum',
    'volatility': r'volatility|atr|true_range|std_|variance|boll|range|dispersion|drawdown',
    'market_breadth': r'market_|breadth|peer_|cross_section|universe_|index_|equal_weight',
    'momentum': r'momentum|return_|ret_|pct|roc|rsi|macd|trend|slope|moving_avg|sma|ema',
    'price_structure': r'price|close|open|high|low|gap|candle|wick|body|shadow',
}

def feature_groups(names):
    """Disjoint, explicit first-match groups; review uncategorized features."""
    groups={k:[] for k in GROUP_PATTERNS};groups['other']=[]
    for i,name in enumerate(names):
        found=next((k for k,p in GROUP_PATTERNS.items() if re.search(p,str(name),re.I)), 'other')
        groups[found].append(i)
    return groups

def _daily_metrics(frame, score, k, model, fold):
    chosen=daily_top_k(frame,score,k,model,fold)
    chosen=chosen.loc[chosen['selected']].copy()
    daily=chosen.groupby('bar_timestamp',sort=True).next_return.mean()
    market=frame.groupby('bar_timestamp',sort=True).next_return.mean()
    payoff=payoff_stats(chosen.next_return)
    return chosen,dict(fold=fold,model=model,k=k,mean_daily=float(daily.mean()),market_excess=float((daily-market.reindex(daily.index)).mean()),**payoff),daily

def _group_names(original_names,extra_names):
    return list(original_names)+list(extra_names)

def _ic(frame, score):
    z=frame[['bar_timestamp','next_return']].copy();z['score']=score
    vals=[]
    for _,g in z.groupby('bar_timestamp',sort=True):
        if len(g)>=8 and g.score.nunique()>1 and g.next_return.nunique()>1:
            x=spearmanr(g.score,g.next_return).statistic
            if np.isfinite(x):vals.append(x)
    return float(np.mean(vals)) if vals else np.nan

def run_feature_sources(frame,extra_cols,*,cap=60000,seed=42,output_dir=None):
    """Ablate and isolate feature families on the same three chronological folds.

    Model selection is NOT performed here. All tested configurations are
    exploratory and their validation results must not be recycled as test data.
    """
    metrics=[];trades=[];weights=[];audit=[];importance=[];daily_rows=[];timings=[]
    for fold in expanding_folds(frame,n_folds=3):
        tr,va,xb,xv,objects=fold_data(frame,fold,target_q=7)
        last=pd.to_datetime(tr.next_bar_timestamp,utc=True).max()
        first=pd.to_datetime(va.bar_timestamp,utc=True).min()
        assert last<=fold.train_end<first
        ax,az=prepare_extras(tr,va,extra_cols)
        X=np.column_stack([np.asarray(xb,dtype='float32'),ax]); Z=np.column_stack([np.asarray(xv,dtype='float32'),az])
        pre=objects['preprocessing']
        base_names=list(pre.get_feature_names_out())
        names=_group_names(base_names,extra_cols)
        assert len(names)==X.shape[1]
        groups=feature_groups(names)
        rng=np.random.default_rng(seed+fold.number)
        idx=np.sort(rng.choice(len(X),size=min(cap,len(X)),replace=False))
        x=X[idx]; y=tr.next_return.to_numpy(float)[idx]
        lo,hi=np.quantile(y,[.005,.995]);y=np.clip(y,lo,hi)
        scale=StandardScaler();xs=scale.fit_transform(x);zs=scale.transform(Z)
        configs={'ridge_all':np.arange(X.shape[1])}
        for group,ids in groups.items():
            if ids:
                configs['only_'+group]=np.asarray(ids)
                other=np.asarray([i for i in range(X.shape[1]) if i not in set(ids)])
                if len(other):configs['without_'+group]=other
        # Training-only feature selection and coefficient diagnostics.
        reference=Ridge(alpha=100.).fit(xs,y)
        coef=np.asarray(reference.coef_)
        top=np.argsort(np.abs(coef))[-min(10,len(coef)):]
        for n in (10,30,60):
            configs[f'top_{n}_coef']=np.argsort(np.abs(coef))[-min(n,len(coef)):]
        for i,(name,c) in enumerate(zip(names,coef)):
            weights.append(dict(fold=fold.number,feature=name,group=next((g for g,ids in groups.items() if i in ids),'other'),coefficient=float(c),absolute=float(abs(c))))
        audit.append(dict(fold=fold.number,train_rows=len(tr),validation_rows=len(va),train_sample=len(idx),feature_count=X.shape[1],train_last_label=last,validation_first_bar=first,groups={k:len(v) for k,v in groups.items()}.__str__()))
        for model,cols in configs.items():
            start=time.perf_counter()
            fitted=Ridge(alpha=100.).fit(xs[:,cols],y)
            score=fitted.predict(zs[:,cols])
            timings.append(dict(fold=fold.number,model=model,fit_seconds=time.perf_counter()-start,mean_daily_ic=_ic(va,score)))
            for k in (5,10,20):
                picks,row,daily=_daily_metrics(va,score,k,model,fold.number)
                metrics.append(row)
                picks=picks.copy();picks['k']=k;trades.append(picks)
                daily_rows.extend(dict(fold=fold.number,model=model,k=k,bar_timestamp=d,daily_return=float(r)) for d,r in daily.items())
        # Matched-ranking diagnostics: score-neutralization against contemporaneous
        # *observable* factor proxies. Validation returns are not used to rank.
        proxy_names=[(i,n) for i,n in enumerate(names) if re.search(r'market_|momentum|volume|volatility',n,re.I)]
        for j,n in proxy_names[:8]:
            s=reference.predict(zs)
            proxy=zs[:,j]
            z=va[['bar_timestamp']].copy();z['score']=s;z['proxy']=proxy
            residual=np.empty(len(z))
            for _,ix in z.groupby('bar_timestamp').indices.items():
                a=proxy[ix];b=s[ix];den=float(np.dot(a-a.mean(),a-a.mean()))
                beta=float(np.dot(a-a.mean(),b-b.mean())/den) if den>1e-12 else 0.
                residual[ix]=b-beta*(a-a.mean())
            model='neutral_'+re.sub('[^a-zA-Z0-9_]+','_',n)[:55]
            for k in (5,10,20):
                picks,row,daily=_daily_metrics(va,residual,k,model,fold.number)
                metrics.append(row);picks=picks.copy();picks['k']=k;trades.append(picks)
                daily_rows.extend(dict(fold=fold.number,model=model,k=k,bar_timestamp=d,daily_return=float(r)) for d,r in daily.items())
    out=dict(metrics=pd.DataFrame(metrics),trades=pd.concat(trades,ignore_index=True),weights=pd.DataFrame(weights),audit=pd.DataFrame(audit),timings=pd.DataFrame(timings),daily=pd.DataFrame(daily_rows))
    if output_dir:
        dest=Path(output_dir);dest.mkdir(parents=True,exist_ok=True)
        for k,v in out.items():v.to_csv(dest/f'{k}.csv',index=False)
    return out

def trade_diagnostics(trades,*,model='ridge_a100',k=5):
    """Analyze previously frozen selections; no retrospective selection changes."""
    x=trades[(trades.model==model)&(trades.k==k)].copy()
    if x.empty:raise ValueError('No matching frozen trades')
    x['bar_timestamp']=pd.to_datetime(x.bar_timestamp,utc=True)
    x['year']=x.bar_timestamp.dt.year
    x['month']=x.bar_timestamp.dt.to_period('M').astype(str)
    x['positive']=x.next_return>0
    x['contribution']=x.next_return
    by_symbol=x.groupby('symbol').agg(trades=('next_return','size'),mean_return=('next_return','mean'),sum_return=('next_return','sum')).sort_values('sum_return',ascending=False)
    by_year=x.groupby(['fold','year']).next_return.agg(['count','mean','median','std'])
    by_month=x.groupby(['fold','month']).next_return.agg(['count','mean'])
    by_date=x.groupby(['fold','bar_timestamp']).next_return.mean().reset_index(name='daily_return')
    by_date['equity']=(1+by_date.daily_return).groupby(by_date.fold).cumprod()
    by_date['drawdown']=by_date.equity/by_date.groupby('fold').equity.cummax()-1
    top=x.sort_values('next_return',ascending=False).head(30)
    bottom=x.sort_values('next_return').head(30)
    concentration=[]
    for fold,g in x.groupby('fold'):
        positives=g.loc[g.next_return>0,'next_return'].sort_values(ascending=False)
        total=g.next_return.sum()
        for frac in (.001,.005,.01,.025,.05):
            n=max(1,int(np.ceil(len(g)*frac)))
            concentration.append(dict(fold=fold,fraction=frac,top_n=n,top_winner_sum=float(positives.head(n).sum()),total_sum=float(total),share_of_net=float(positives.head(n).sum()/total) if abs(total)>1e-12 else np.nan))
    return dict(by_symbol=by_symbol.reset_index(),by_year=by_year.reset_index(),by_month=by_month.reset_index(),by_date=by_date,top_winners=top,bottom_losers=bottom,concentration=pd.DataFrame(concentration))

def paired_block_intervals(daily,*,reference='ridge_all',k=5,block=10,reps=1500,seed=42):
    x=daily[daily.k==k]
    out=[]
    for fold,g in x.groupby('fold'):
        p=g.pivot(index='bar_timestamp',columns='model',values='daily_return')
        for model in p.columns:
            if model==reference:continue
            delta=(p[model]-p[reference]).dropna()
            if len(delta)<5:continue
            lo,hi=date_block_ci(delta,block=block,reps=reps,seed=seed+fold)
            out.append(dict(fold=fold,model=model,k=k,mean_delta=float(delta.mean()),ci_low=lo,ci_high=hi,days=len(delta)))
    return pd.DataFrame(out)

def reality_check(daily,*,baseline='ridge_all',k=5,reps=1500,block=10,seed=42):
    """Exploratory familywise max-t bootstrap of paired *daily* performance.

    Center each challenger-vs-baseline series under the joint null, resample
    common contiguous date blocks, and compare the largest observed mean lift
    to the bootstrap distribution of the maximum lift. Fold dates stay separate.
    This is a development-data diagnostic, not an independent confirmatory test.
    """
    x=daily[daily.k==k]
    p=x.pivot(index=['fold','bar_timestamp'],columns='model',values='daily_return').sort_index()
    challengers=[c for c in p.columns if c!=baseline]
    if not challengers:return pd.DataFrame()
    diffs=p[challengers].sub(p[baseline],axis=0).dropna()
    obs=diffs.mean();rng=np.random.default_rng(seed)
    arrays=[g.to_numpy(float) for _,g in diffs.groupby(level=0)]
    draws=np.empty(reps)
    for rep in range(reps):
        samples=[]
        for a in arrays:
            n=len(a);b=min(block,n);indices=[]
            while len(indices)<n:
                start=int(rng.integers(0,n-b+1));indices.extend(range(start,start+b))
            centered=a-a.mean(axis=0)
            samples.append(centered[np.asarray(indices[:n])])
        draws[rep]=np.concatenate(samples).mean(axis=0).max()
    return pd.DataFrame([dict(model=c,observed_mean_lift=float(obs[c]),familywise_p=float((1+(draws>=obs[c]).sum())/(reps+1)),n_strategies=len(challengers)) for c in challengers]).sort_values('observed_mean_lift',ascending=False)
