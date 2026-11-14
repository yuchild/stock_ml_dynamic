"""Notebook 45: read-only point-in-time integrity and sensitivity diagnostics.
No validation outcomes are used to select positions. No claims of external verification.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from .edge_robustness_41 import rank_with_exclusions
from .payoff_search import date_block_ci

KEYS=['symbol','interval_code','bar_timestamp']

def master_integrity(master, latest, manifest):
    x=master.copy(); x['bar_timestamp']=pd.to_datetime(x.bar_timestamp,utc=True)
    x['next_bar_timestamp']=pd.to_datetime(x.next_bar_timestamp,utc=True)
    checks=[]
    def add(name,value,interpretation):checks.append(dict(check=name,value=value,interpretation=interpretation))
    add('master_rows',len(x),'population')
    add('master_symbols',x.symbol.nunique(),'not point-in-time membership')
    add('duplicate_keys',int(x.duplicated(KEYS).sum()),'must equal zero')
    add('nonmonotonic_target_timestamps',int((x.next_bar_timestamp<=x.bar_timestamp).sum()),'must equal zero')
    add('nonfinite_targets',int((~np.isfinite(pd.to_numeric(x.next_return,errors='coerce'))).sum()),'must equal zero')
    add('target_abs_over_20pct',int((x.next_return.abs()>.2).sum()),'requires source-level review')
    add('target_abs_over_50pct',int((x.next_return.abs()>.5).sum()),'requires source-level review')
    add('missing_latest_symbols',int(len(set(x.symbol)-set(latest.symbol))),'may be delistings, gaps, or filtering')
    for col in ['adj_close','close','volume','next_adj_close']:
        if col in x:
            s=pd.to_numeric(x[col],errors='coerce')
            add('nonfinite_'+col,int((~np.isfinite(s)).sum()),'requires review')
            if col!='volume':add('nonpositive_'+col,int((s<=0).sum()),'requires review')
            else:add('negative_volume',int((s<0).sum()),'requires review')
    if 'adj_close' in x and 'next_adj_close' in x:
        implied=x.next_adj_close/x.adj_close-1
        diff=(implied-x.next_return).abs()
        add('target_identity_over_1e-8',int((diff>1e-8).sum()),'next_return must equal next_adj_close / adj_close - 1')
        add('max_target_identity_error',float(diff.max()),'rounding tolerance')
    return pd.DataFrame(checks)

def chronology_audit(master,scored,fold_audit):
    x=master.copy();x['bar_timestamp']=pd.to_datetime(x.bar_timestamp,utc=True)
    x['next_bar_timestamp']=pd.to_datetime(x.next_bar_timestamp,utc=True)
    rows=[]
    for r in fold_audit.itertuples(index=False):
        train_end=pd.Timestamp(r.train_last_label);first=pd.Timestamp(r.validation_first)
        rows.append(dict(fold=int(r.fold),train_last_label=train_end,validation_first=first,
                         purged=train_end<first,train_rows=int(r.train_rows),validation_rows=int(r.validation_rows)))
    return pd.DataFrame(rows)

def source_inventory(master,latest):
    x=master.copy();x['bar_timestamp']=pd.to_datetime(x.bar_timestamp,utc=True)
    latest_symbols=set(latest.symbol)
    rows=[]
    for symbol,g in x.groupby('symbol',sort=True):
        g=g.sort_values('bar_timestamp');delta=g.bar_timestamp.diff().dt.total_seconds()/86400
        rows.append(dict(symbol=symbol,rows=len(g),first=g.bar_timestamp.iloc[0],last=g.bar_timestamp.iloc[-1],
                         max_calendar_gap_days=float(delta.max()) if len(g)>1 else np.nan,
                         gaps_over_7_calendar_days=int((delta>7).sum()),
                         in_latest=symbol in latest_symbols,
                         extreme_20pct=int((g.next_return.abs()>.2).sum())))
    return pd.DataFrame(rows)

def extreme_observations(master,threshold=.2):
    x=master.copy();x['bar_timestamp']=pd.to_datetime(x.bar_timestamp,utc=True)
    cols=[c for c in ['symbol','bar_timestamp','next_bar_timestamp','close','adj_close','next_adj_close','volume','next_return'] if c in x]
    return x.loc[x.next_return.abs()>threshold,cols].sort_values('next_return',key=lambda s:s.abs(),ascending=False).reset_index(drop=True)

def score_integrity(scored):
    x=scored.copy();checks=[]
    for f,g in x.groupby('fold'):
        checks.append(dict(fold=int(f),rows=len(g),symbols=g.symbol.nunique(),dates=g.bar_timestamp.nunique(),
                           duplicate_stock_dates=int(g.duplicated(['bar_timestamp','symbol']).sum()),
                           missing_scores=int((~np.isfinite(g.score)).sum()),
                           nonfinite_returns=int((~np.isfinite(g.next_return)).sum()),
                           extreme_20pct=int((g.next_return.abs()>.2).sum())))
    return pd.DataFrame(checks)

def frozen_sensitivity(scored,ks=(5,10,20),exclude_symbols=(),thresholds=(.1,.2,.5)):
    """Outcome masking only: keeps ex-ante selections fixed; no re-optimization."""
    rows=[];daily=[]
    for fold,g in scored.groupby('fold'):
        market=g.groupby('bar_timestamp').next_return.mean()
        for k in ks:
            selected=rank_with_exclusions(g,g.score.to_numpy(),k,excluded=exclude_symbols)
            for threshold in (None,*thresholds):
                for method in ('original','winsorize_both','drop_extremes','zero_extremes'):
                    if threshold is None and method!='original':continue
                    if threshold is not None and method=='original':continue
                    s=selected.copy();y=s.next_return.astype(float)
                    if method=='winsorize_both':s['tested_return']=y.clip(-threshold,threshold)
                    elif method=='drop_extremes':s['tested_return']=y.where(y.abs()<=threshold)
                    elif method=='zero_extremes':s['tested_return']=y.where(y.abs()<=threshold,0.)
                    else:s['tested_return']=y
                    day=s.groupby('bar_timestamp').tested_return.mean().reindex(market.index)
                    name=method if threshold is None else f'{method}_{threshold:g}'
                    rows.append(dict(fold=int(fold),k=int(k),scenario=name,days=int(day.notna().sum()),
                                     mean_daily=float(day.mean()),market_excess=float((day-market).mean()),
                                     observations=int(s.tested_return.notna().sum())))
                    daily.extend(dict(fold=int(fold),k=int(k),scenario=name,bar_timestamp=t,daily_return=float(v)) for t,v in day.dropna().items())
    return pd.DataFrame(rows),pd.DataFrame(daily)

def symbol_influence(scored,k=5,top_n=15):
    """Leave-symbol-out *diagnostic*, selected symbols are identified ex post; not an OOS improvement."""
    base=[]
    for f,g in scored.groupby('fold'):
        s=rank_with_exclusions(g,g.score.to_numpy(),k);s['fold']=f;base.append(s)
    base=pd.concat(base,ignore_index=True)
    impact=base.groupby('symbol').agg(n=('next_return','size'),sum_return=('next_return','sum')).reset_index()
    return impact.sort_values('sum_return',ascending=False).head(top_n),base

def bootstrap_sensitivity(daily,reference='original',reps=1200,seed=42):
    rows=[]
    for (fold,k),g in daily.groupby(['fold','k']):
        pivot=g.pivot(index='bar_timestamp',columns='scenario',values='daily_return').sort_index()
        if reference not in pivot:continue
        for scenario in pivot.columns:
            if scenario==reference:continue
            diff=(pivot[scenario]-pivot[reference]).dropna().to_numpy(float)
            if len(diff)<20:continue
            lo,hi=date_block_ci(diff,block=10,reps=reps,seed=seed+int(fold))
            rows.append(dict(fold=int(fold),k=int(k),scenario=scenario,n=len(diff),mean_diff=float(diff.mean()),ci_low=lo,ci_high=hi))
    return pd.DataFrame(rows)

def save_tables(directory,**tables):
    out=Path(directory);out.mkdir(parents=True,exist_ok=True)
    for name,df in tables.items():df.to_csv(out/(name+'.csv'),index=False)
