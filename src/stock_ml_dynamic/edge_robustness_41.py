"""Notebook 41: chronological leave-symbol-out, exposure, score and tail robustness.
Development data only. Zero cost. Validation labels never enter ranking decisions.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from .walk_forward import expanding_folds,fold_data
from .economic_target_research import prepare_extras
from .payoff_search import payoff_stats,date_block_ci

K_VALUES=(5,10,20)
EXCLUDED=('CVNA','PLTR','ECHO','NFLX','ABNB')

def rank_with_exclusions(frame,score,k,excluded=()):
    """Exclude symbols BEFORE ranking; refill each date using next eligible names."""
    x=frame[['bar_timestamp','symbol','next_return']].copy().reset_index(drop=True)
    x['score']=np.asarray(score,float)
    x=x.loc[~x.symbol.isin(excluded)&np.isfinite(x.score)].copy()
    x=x.sort_values(['bar_timestamp','score','symbol'],ascending=[True,False,True])
    x['rank']=x.groupby('bar_timestamp').cumcount()+1
    return x.loc[x['rank']<=k].copy()

def _metrics(x,model,fold,k,market):
    daily=x.groupby('bar_timestamp').next_return.mean().sort_index()
    market=market.reindex(daily.index)
    p=payoff_stats(x.next_return)
    equity=(1+daily).cumprod()
    return dict(model=model,fold=fold,k=k,days=len(daily),mean_daily=float(daily.mean()),market_excess=float((daily-market).mean()),daily_std=float(daily.std()),max_drawdown=float((equity/equity.cummax()-1).min()),**p),daily

def fit_and_score(frame,extra_cols,cap=60000,seed=42):
    """Exactly one frozen Ridge fit per fold; score diagnostics reuse predictions."""
    records=[];coefficients=[];audit=[]
    for f in expanding_folds(frame,n_folds=3):
        tr,va,xb,xv,obj=fold_data(frame,f,target_q=7)
        assert pd.to_datetime(tr.next_bar_timestamp,utc=True).max()<=f.train_end
        assert pd.to_datetime(va.bar_timestamp,utc=True).min()>f.train_end
        ax,az=prepare_extras(tr,va,extra_cols)
        X=np.column_stack([np.asarray(xb,dtype='float32'),ax]);Z=np.column_stack([np.asarray(xv,dtype='float32'),az])
        rng=np.random.default_rng(seed+f.number)
        ids=np.sort(rng.choice(len(X),min(cap,len(X)),replace=False))
        y=tr.next_return.to_numpy(float)[ids]
        lo,hi=np.quantile(y,[.005,.995]);y=np.clip(y,lo,hi)
        scaler=StandardScaler();xs=scaler.fit_transform(X[ids]);zs=scaler.transform(Z)
        model=Ridge(alpha=100.).fit(xs,y)
        score=model.predict(zs)
        names=list(obj['preprocessing'].get_feature_names_out())+list(extra_cols)
        for name,coef in zip(names,model.coef_):coefficients.append(dict(fold=f.number,feature=name,coef=float(coef)))
        v=va[['bar_timestamp','symbol','next_return']].copy().reset_index(drop=True)
        v['score']=score;v['fold']=f.number
        # Diagnostics: strictly observable price/volume fields, if present.
        for col in ('close','adj_close','volume','market_return','market_momentum_5','relative_volume_20','volatility_20'):
            if col in va.columns:v[col]=va[col].to_numpy()
        records.append(v)
        audit.append(dict(fold=f.number,train_rows=len(tr),validation_rows=len(va),sample_rows=len(ids),train_last_label=str(pd.to_datetime(tr.next_bar_timestamp,utc=True).max()),validation_first=str(pd.to_datetime(va.bar_timestamp,utc=True).min()),feature_count=X.shape[1],alpha=100))
    return pd.concat(records,ignore_index=True),pd.DataFrame(coefficients),pd.DataFrame(audit)

def score_variants(frame):
    """Date-local, ex-ante transforms of frozen scores. No realized returns used."""
    x=frame.copy();g=x.groupby(['fold','bar_timestamp'])
    x['score_rank']=g.score.rank(pct=True)
    x['score_z']=g.score.transform(lambda v:(v-v.mean())/(v.std()+1e-12))
    # Confidence is relative to cross-sectional score dispersion, not fitted probability.
    x['score_centered']=x.score-g.score.transform('median')
    return x

def run_portfolio_stress(scored,excluded=EXCLUDED,k_values=K_VALUES):
    """All variants rank before inspecting realized returns. No ex-post winners removed."""
    # Compute ex-ante score transformations here so callers may pass raw
    # fit_and_score() output without preparing diagnostic columns manually.
    scored=score_variants(scored)
    metrics=[];daily_rows=[];trades=[]
    for fold,v in scored.groupby('fold',sort=True):
        v=v.reset_index(drop=True)
        market=v.groupby('bar_timestamp').next_return.mean()
        specs={'ridge_all':(), 'exclude_CVNA':('CVNA',),'exclude_PLTR':('PLTR',),
               'exclude_CVNA_PLTR':('CVNA','PLTR'),'exclude_top5_contributors':tuple(excluded)}
        for name,ban in specs.items():
            for k in k_values:
                chosen=rank_with_exclusions(v,v.score.to_numpy(),k,ban)
                if chosen.empty:continue
                row,d=_metrics(chosen,name,fold,k,market);metrics.append(row)
                chosen['fold']=fold;chosen['model']=name;chosen['k']=k;trades.append(chosen)
                daily_rows.extend(dict(fold=fold,model=name,k=k,bar_timestamp=t,daily_return=float(r)) for t,r in d.items())
        # No future-return-informed selection. Scores are fixed from frozen Ridge.
        for name,method in [('score_above_daily_median','median'),('score_above_daily_q75','q75'),('score_positive','positive')]:
            cutoff=(v.groupby('bar_timestamp').score.transform('median') if method=='median' else
                    v.groupby('bar_timestamp').score.transform(lambda z:z.quantile(.75)) if method=='q75' else 0.)
            subset=v.loc[v.score>cutoff].copy()
            for k in k_values:
                chosen=rank_with_exclusions(subset,subset.score.to_numpy(),k)
                if chosen.empty:continue
                row,d=_metrics(chosen,name,fold,k,market);row['coverage']=len(d)/len(market);metrics.append(row)
                chosen['fold']=fold;chosen['model']=name;chosen['k']=k;trades.append(chosen)
                daily_rows.extend(dict(fold=fold,model=name,k=k,bar_timestamp=t,daily_return=float(r)) for t,r in d.items())
        # Ex-ante score transformations: ranking invariant within date (negative control).
        for name,col in [('rank_pct','score_rank'),('score_z','score_z')]:
            for k in k_values:
                chosen=rank_with_exclusions(v,v[col].to_numpy(),k)
                row,d=_metrics(chosen,name,fold,k,market);metrics.append(row)
                chosen['fold']=fold;chosen['model']=name;chosen['k']=k;trades.append(chosen)
                daily_rows.extend(dict(fold=fold,model=name,k=k,bar_timestamp=t,daily_return=float(r)) for t,r in d.items())
    return dict(metrics=pd.DataFrame(metrics),daily=pd.DataFrame(daily_rows),trades=pd.concat(trades,ignore_index=True))

def exposure_diagnostics(trades,scored):
    x=trades.copy();x['bar_timestamp']=pd.to_datetime(x.bar_timestamp,utc=True)
    base=x[(x.model=='ridge_all')&(x.k==5)].copy()
    sym=base.groupby('symbol').agg(trades=('next_return','size'),gross_sum=('next_return','sum'),mean_return=('next_return','mean')).sort_values('gross_sum',ascending=False).reset_index()
    counts=base.groupby(['fold','symbol']).size().reset_index(name='n')
    counts['share_of_days']=counts.n/counts.groupby('fold').n.transform('sum')*5
    # Date-level market-relative and volatility proxies from available columns.
    factor=[]
    for col in ('volume','market_return','market_momentum_5','relative_volume_20','volatility_20','close','adj_close'):
        if col not in scored.columns:continue
        tmp=scored[['fold','bar_timestamp','symbol',col]].copy()
        tmp['date_rank']=tmp.groupby(['fold','bar_timestamp'])[col].rank(pct=True)
        selected=base[['fold','bar_timestamp','symbol']].merge(tmp,on=['fold','bar_timestamp','symbol'],how='left')
        factor.append(dict(factor=col,selected_percentile=float(selected.date_rank.mean()),universe_percentile=float(tmp.date_rank.mean()),selected_median=float(selected[col].median())))
    return dict(symbol=sym,frequency=counts,factor=pd.DataFrame(factor))

def tail_and_integrity(scored,trades):
    """Data-quality flags are *diagnostics*, never silently excluded from backtests."""
    x=scored.copy();x['bar_timestamp']=pd.to_datetime(x.bar_timestamp,utc=True)
    integrity=dict(rows=len(x),symbols=x.symbol.nunique(),duplicate_symbol_dates=int(x.duplicated(['fold','bar_timestamp','symbol']).sum()),nonfinite_returns=int((~np.isfinite(x.next_return)).sum()),abs_return_over_20pct=int((x.next_return.abs()>.2).sum()),abs_return_over_50pct=int((x.next_return.abs()>.5).sum()),abs_return_over_100pct=int((x.next_return.abs()>1).sum()),zero_returns=int((x.next_return==0).sum()))
    base=trades[(trades.model=='ridge_all')&(trades.k==5)].copy()
    base['abs_return']=base.next_return.abs()
    outliers=base.sort_values('abs_return',ascending=False).head(100)
    rows=[]
    for fold,g in base.groupby('fold'):
        total=g.next_return.sum()
        for frac in (.001,.005,.01,.025,.05):
            n=max(1,int(np.ceil(len(g)*frac)))
            wins=g.next_return.nlargest(n).sum()
            rows.append(dict(fold=fold,top_fraction=frac,top_n=n,winner_sum=float(wins),net_sum=float(total),share_of_net=float(wins/total) if abs(total)>1e-12 else np.nan))
    return pd.DataFrame([integrity]),outliers,pd.DataFrame(rows)

def paired_inference(daily,baseline='ridge_all',k=5,reps=1200,seed=42):
    out=[];x=daily[daily.k==k]
    for fold,g in x.groupby('fold'):
        p=g.pivot(index='bar_timestamp',columns='model',values='daily_return')
        for name in p.columns:
            if name==baseline:continue
            delta=(p[name]-p[baseline]).dropna()
            if len(delta)<10:continue
            low,high=date_block_ci(delta,block=10,reps=reps,seed=seed+fold)
            out.append(dict(fold=fold,model=name,k=k,days=len(delta),mean_lift=float(delta.mean()),ci_low=low,ci_high=high))
    return pd.DataFrame(out)

def score_quality(scored):
    rows=[]
    for fold,g in scored.groupby('fold'):
        for date,day in g.groupby('bar_timestamp'):
            if len(day)<10:continue
            ic=spearmanr(day.score,day.next_return).statistic
            rows.append(dict(fold=fold,bar_timestamp=date,ic=float(ic) if np.isfinite(ic) else np.nan,score_spread=float(day.score.std()),return_spread=float(day.next_return.std()),positive_fraction=float((day.next_return>0).mean())))
    return pd.DataFrame(rows)

def save_outputs(output_dir,**frames):
    path=Path(output_dir);path.mkdir(parents=True,exist_ok=True)
    for name,frame in frames.items():frame.to_csv(path/(name+'.csv'),index=False)
