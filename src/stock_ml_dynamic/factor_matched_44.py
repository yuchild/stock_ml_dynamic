"""Notebook 44: ex-ante factor-matched controls and cross-sectional alpha diagnostics.
All ranking/matching uses contemporaneous features; outcomes only for evaluation.
The final historical holdout must be removed upstream.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from .edge_robustness_41 import rank_with_exclusions, _metrics
from .payoff_search import date_block_ci

FACTORS=('volatility_20','relative_volume_20','market_momentum_5','volume')

def add_factor_bins(scored, factors=FACTORS, n_bins=3):
    x=scored.copy().reset_index(drop=True)
    x['bar_timestamp']=pd.to_datetime(x['bar_timestamp'],utc=True)
    available=[c for c in factors if c in x and pd.api.types.is_numeric_dtype(x[c]) and x[c].notna().any()]
    if not available:raise ValueError('No observable matching factors available; inspect scored columns')
    for c in available:
        v=pd.to_numeric(x[c],errors='coerce')
        # rank within date, NOT on outcome; NaNs assigned explicit -1 stratum
        pct=v.groupby([x['fold'],x['bar_timestamp']]).rank(pct=True,method='average')
        x['bin_'+c]=np.where(pct.notna(),np.minimum((pct.fillna(0)*n_bins).astype(int),n_bins-1),-1)
    return x,available

def _matched_day(group, selected_idx, match_factors, rng, exclude_selected=True):
    """Fast within-date nearest-bin sampling without outcomes or replacement."""
    all_idx=group.index.to_numpy(dtype=int)
    selected_idx=np.asarray(selected_idx,dtype=int)
    bins=group[match_factors].to_numpy(dtype=np.int16)
    index_pos={int(ix):i for i,ix in enumerate(all_idx)}
    selected_pos=np.array([index_pos[int(ix)] for ix in selected_idx],dtype=int)
    eligible=np.ones(len(all_idx),dtype=bool)
    if exclude_selected and len(all_idx)-len(selected_pos)>=len(selected_pos):
        eligible[selected_pos]=False
    picked=[]
    for pos in selected_pos:
        candidates=np.flatnonzero(eligible)
        if not len(candidates):break
        distance=np.abs(bins[candidates]-bins[pos]).sum(axis=1)
        nearest=candidates[distance==distance.min()]
        choice=int(rng.choice(nearest));picked.append(int(all_idx[choice]));eligible[choice]=False
    return picked

def matched_random_experiment(scored, k=5, reps=300, seed=42, factors=FACTORS, n_bins=3):
    """Date-specific, matched randomized portfolios. Equal-weight, no transaction costs.
    Returns observed, random distribution, and daily difference for block bootstrap.
    Matched samples exclude selected stocks when feasible; thus random-control
    population differs from selected set (a deliberate, documented contrast).
    """
    x,available=add_factor_bins(scored,factors,n_bins)
    cols=['bin_'+c for c in available]
    rng=np.random.default_rng(seed)
    selected_ids=[]
    for fold,group in x.groupby('fold',sort=True):
        ranked=rank_with_exclusions(group,group['score'].to_numpy(),k)
        chosen=group.reset_index().merge(ranked[['bar_timestamp','symbol']],on=['bar_timestamp','symbol'],how='inner',validate='one_to_one')
        selected_ids.extend(chosen['index'].tolist())
    selected_ids=np.asarray(selected_ids,dtype=int)
    selected_flag=np.zeros(len(x),dtype=bool);selected_flag[selected_ids]=True
    dates=[];observed=[];random_rows=[];matched_gap=[]
    for (fold,dt),g in x.groupby(['fold','bar_timestamp'],sort=True):
        ix=g.index.to_numpy();sel=ix[selected_flag[ix]]
        if not len(sel):continue
        observed_return=float(x.loc[sel,'next_return'].mean())
        dates.append((int(fold),dt));observed.append(observed_return)
        selected_bins=g.loc[sel,cols].to_numpy(float)
        # For audit, calculate average nearest-bin distance (0 means exact match).
        choices=[];gaps=[]
        for r in range(reps):
            choice=_matched_day(g,sel,cols,rng)
            choices.append(float(x.loc[choice,'next_return'].mean()))
            gaps.append(float(np.abs(g.loc[choice,cols].to_numpy(float)-selected_bins[:len(choice)]).sum(axis=1).mean()))
        random_rows.append(choices);matched_gap.append(float(np.mean(gaps)))
    random_array=np.asarray(random_rows,float)
    daily=pd.DataFrame({'fold':[d[0] for d in dates], 'bar_timestamp':[d[1] for d in dates],
                        'observed':observed,'matched_mean':random_array.mean(axis=1),
                        'market_mean':[float(x.loc[(x.fold==f)&(x.bar_timestamp==dt),'next_return'].mean()) for f,dt in dates],
                        'mean_bin_distance':matched_gap})
    daily['matched_excess']=daily.observed-daily.matched_mean
    daily['market_excess']=daily.observed-daily.market_mean
    sim=[]
    for f,g in daily.groupby('fold',sort=True):
        pos=np.where(daily.fold.to_numpy()==f)[0]
        obs=float(g.observed.mean());null=random_array[pos].mean(axis=0)
        sim.append(dict(fold=int(f),k=k,days=len(g),observed=obs,matched_random_mean=float(null.mean()),
                        excess=float(obs-null.mean()),null_sd=float(null.std(ddof=1)),
                        monte_carlo_p=float((1+np.sum(null>=obs))/(reps+1)),
                        null_p025=float(np.quantile(null,.025)),null_p975=float(np.quantile(null,.975)),
                        mean_bin_distance=float(g.mean_bin_distance.mean())))
    return daily,pd.DataFrame(sim),pd.DataFrame({'rep':np.arange(reps),**{f'fold_{f}':random_array[np.where(daily.fold.to_numpy()==f)[0]].mean(axis=0) for f in sorted(daily.fold.unique())}}),available

def within_bin_directionality(scored,factors=FACTORS,n_bins=3):
    x,available=add_factor_bins(scored,factors,n_bins)
    rows=[]
    for factor in available:
        col='bin_'+factor
        for (fold,b),g in x.groupby(['fold',col]):
            if b<0 or len(g)<100:continue
            auc=spearmanr(g.score,g.next_return).statistic
            # conditional tail asymmetry: compare top and bottom score deciles within bin
            lo=g.score.quantile(.1);hi=g.score.quantile(.9)
            top=g.loc[g.score>=hi];bottom=g.loc[g.score<=lo]
            rows.append(dict(fold=int(fold),factor=factor,bin=int(b),n=len(g),ic=float(auc),
                             top_mean=float(top.next_return.mean()),bottom_mean=float(bottom.next_return.mean()),
                             directional_spread=float(top.next_return.mean()-bottom.next_return.mean()),
                             top_up5=float((top.next_return>.05).mean()),top_down5=float((top.next_return<-.05).mean())))
    return pd.DataFrame(rows)

def stratified_topk(scored,k=5,factor='volatility_20',n_bins=3):
    x,available=add_factor_bins(scored,(factor,),n_bins)
    if factor not in available:raise ValueError(f'Missing factor {factor}')
    x=x.sort_values(['fold','bar_timestamp','score','symbol'],ascending=[True,True,False,True]).copy()
    # distribute k across observable factor bins; remainder allocated by highest score
    col='bin_'+factor;selected=[]
    for (_,dt),g in x.groupby(['fold','bar_timestamp'],sort=True):
        parts=[];quota=k//n_bins
        for b in range(n_bins):parts.append(g.loc[g[col].eq(b)].head(quota))
        first=pd.concat(parts) if parts else g.iloc[:0]
        remaining=g.loc[~g.index.isin(first.index)].head(k-len(first))
        selected.append(pd.concat([first,remaining]))
    return pd.concat(selected,ignore_index=True)

def portfolio_summary(selected,scored,k=5):
    rows=[];daily=[]
    for f,g in selected.groupby('fold'):
        market=scored.loc[scored.fold.eq(f)].groupby('bar_timestamp').next_return.mean()
        m,d=_metrics(g,'factor_balanced',int(f),k,market);rows.append(m)
        daily.extend(dict(fold=int(f),bar_timestamp=dt,balanced=float(ret)) for dt,ret in d.items())
    return pd.DataFrame(rows),pd.DataFrame(daily)

def block_inference(daily,columns=('matched_excess','market_excess'),reps=1200,seed=42):
    rows=[]
    for f,g in daily.groupby('fold'):
        for c in columns:
            values=g.sort_values('bar_timestamp')[c].to_numpy(float)
            low,high=date_block_ci(values,block=10,reps=reps,seed=seed+int(f))
            rows.append(dict(fold=int(f),comparison=c,n=len(values),mean=float(values.mean()),ci_low=low,ci_high=high))
    return pd.DataFrame(rows)

def leave_symbol_out(scored,symbols=('CVNA','PLTR','ECHO','NFLX','ABNB'),k=5):
    rows=[]
    for banned in [(),('CVNA',),('PLTR',),tuple(symbols)]:
        name='none' if not banned else '+'.join(banned)
        for f,g in scored.groupby('fold'):
            selected=rank_with_exclusions(g,g.score.to_numpy(),k,excluded=banned)
            market=g.groupby('bar_timestamp').next_return.mean()
            m,_=_metrics(selected,name,int(f),k,market);rows.append(m)
    return pd.DataFrame(rows)

def save_tables(directory,**tables):
    out=Path(directory);out.mkdir(parents=True,exist_ok=True)
    for name,df in tables.items():df.to_csv(out/(name+'.csv'),index=False)
