"""Notebook 36: fixed prior-calibration thresholds and constrained daily portfolios.

Uses Notebook 34 *development validation* predictions and prior-calibration thresholds.
No model fitting, final holdout access, or prospective ledger writes.
"""
from pathlib import Path
import numpy as np
import pandas as pd

MODELS=('full_history_rf','oof_meta_logistic','oof_meta_rf')


def load_inputs(predictions_path, policies_path, *, calibration='raw', coverage=.10):
    from .exact_coverage_regime_research import load_predictions
    predictions=load_predictions(predictions_path)
    policies=pd.read_csv(policies_path)
    required={'fold','model','calibration','target_coverage','threshold'}
    if not required.issubset(policies.columns):
        raise ValueError(f'Notebook 34 policy columns missing: {required-set(policies.columns)}')
    if not 0<coverage<1:raise ValueError('coverage must be between zero and one')
    p=predictions[predictions.calibration.eq(calibration)].copy()
    thresholds=policies[policies.calibration.eq(calibration)&np.isclose(policies.target_coverage,coverage)].copy()
    if p.empty or thresholds.empty:raise ValueError('Missing predictions or training-calibrated thresholds')
    if thresholds.duplicated(['fold','model']).any():raise ValueError('Ambiguous thresholds')
    pairs=set(zip(p.fold,p.model))
    if pairs!=set(zip(thresholds.fold,thresholds.model)):
        raise ValueError('Threshold/model/fold pairs differ from prediction pairs')
    if not set(MODELS).issubset(set(p.model)):
        raise ValueError('Missing benchmark or challenger models')
    if not np.isfinite(thresholds.threshold.to_numpy(dtype=float)).all():
        raise ValueError('Nonfinite threshold')
    return p,thresholds


def add_sector_map(predictions, sector_csv=None):
    p=predictions.copy()
    if sector_csv is None:
        p['sector']=pd.NA
        return p
    lookup=pd.read_csv(sector_csv)
    if not {'symbol','sector'}.issubset(lookup):
        raise ValueError('Sector CSV must have symbol and sector columns')
    lookup=lookup[['symbol','sector']].drop_duplicates()
    if lookup.symbol.duplicated().any():raise ValueError('Conflicting sector mappings')
    p=p.merge(lookup,on='symbol',how='left',validate='many_to_one')
    return p


def select_portfolio(predictions,thresholds,*,max_positions=10,max_per_sector=None,
                     cooldown_days=0,cost_bps=10,turnover_bps=5):
    """Chronological threshold and daily-cap simulation; no validation-label selection.

    Positions last one observation interval. Turnover is a daily symbol-set change
    proxy, not a fill-level execution simulation. Missing sector mappings prohibit
    sector caps rather than silently treating them as valid sectors.
    """
    if max_positions<1 or cooldown_days<0:raise ValueError('Invalid portfolio limits')
    if max_per_sector is not None and max_per_sector<1:raise ValueError('Invalid sector cap')
    if cost_bps<0 or turnover_bps<0:raise ValueError('Costs must be nonnegative')
    p=predictions.copy()
    if 'sector' not in p:p['sector']=pd.NA
    if max_per_sector is not None and p.sector.isna().any():
        raise ValueError('Sector cap requires complete, point-in-time sector mapping')
    t=thresholds[['fold','model','threshold']]
    p=p.merge(t,on=['fold','model'],how='left',validate='many_to_one')
    if p.threshold.isna().any():raise ValueError('Some thresholds were not found')
    p['eligible']=p.up_probability>=p.threshold
    selected=[]; daily=[]
    for (fold,model),group in p.groupby(['fold','model'],sort=True):
        prev=set();last_seen={}; day_number=0
        for date,g in group.groupby('bar_timestamp',sort=True):
            candidates=g[g.eligible].sort_values(['up_probability','symbol'],ascending=[False,True],kind='mergesort')
            chosen=[];sector_counts={}
            for _,row in candidates.iterrows():
                symbol=row.symbol
                if symbol in last_seen and day_number-last_seen[symbol]<=cooldown_days:continue
                sector=row.sector
                if max_per_sector is not None and sector_counts.get(sector,0)>=max_per_sector:continue
                chosen.append(row)
                sector_counts[sector]=sector_counts.get(sector,0)+1
                if len(chosen)==max_positions:break
            current={r.symbol for r in chosen}
            # Count new positions relative to the previous day. All positions are
            # assumed to close after one bar; this is a churn proxy only.
            new_names=len(current-prev)
            per_entry_extra=turnover_bps/10000*(new_names/len(current)) if current else 0.
            for r in chosen:
                selected.append(dict(fold=fold,model=model,bar_timestamp=date,symbol=r.symbol,
                    sector=r.sector,up_probability=float(r.up_probability),direction=int(r.direction),
                    next_return=float(r.next_return),net_return=float(r.next_return)-cost_bps/10000-per_entry_extra))
                last_seen[r.symbol]=day_number
            gross=float(sum(r.next_return for r in chosen))/max_positions
            net=(sum(r.next_return-cost_bps/10000 for r in chosen)-turnover_bps/10000*new_names)/max_positions
            daily.append(dict(fold=fold,model=model,bar_timestamp=date,eligible=int(g.eligible.sum()),
                entries=len(chosen),universe=len(g),new_names=new_names,
                gross_portfolio_return=gross,net_portfolio_return=float(net),
                invested_fraction=len(chosen)/max_positions))
            prev=current;day_number+=1
    return pd.DataFrame(selected),pd.DataFrame(daily)


def summarize(trades,daily):
    rows=[]
    for (fold,model),d in daily.groupby(['fold','model']):
        t=trades[(trades.fold==fold)&(trades.model==model)] if len(trades) else pd.DataFrame()
        vals=d.net_portfolio_return.to_numpy(dtype=float)
        wealth=np.cumprod(1+vals)
        peak=np.maximum.accumulate(np.r_[1.,wealth])[1:]
        drawdown=float(np.min(wealth/peak-1)) if len(wealth) else np.nan
        rows.append(dict(fold=fold,model=model,days=len(d),entries=len(t),
            coverage=float(len(t)/d.universe.sum()),active_days=int((d.entries>0).sum()),
            avg_invested_fraction=float(d.invested_fraction.mean()),
            mean_net_per_trade=float(t.net_return.mean()) if len(t) else np.nan,
            up_precision=float((t.direction==1).mean()) if len(t) else np.nan,
            mean_daily_net=float(vals.mean()),total_compounded_net=float(wealth[-1]-1) if len(wealth) else np.nan,
            max_drawdown=drawdown,mean_daily_new_names=float(d.new_names.mean())))
    return pd.DataFrame(rows)


def paired_daily(daily,challenger):
    a=daily[daily.model.eq('full_history_rf')][['fold','bar_timestamp','net_portfolio_return']].rename(columns={'net_portfolio_return':'baseline'})
    b=daily[daily.model.eq(challenger)][['fold','bar_timestamp','net_portfolio_return']].rename(columns={'net_portfolio_return':'challenger'})
    merged=a.merge(b,on=['fold','bar_timestamp'],how='inner',validate='one_to_one')
    if len(merged)!=len(a) or len(merged)!=len(b):raise ValueError('Different validation dates')
    merged['difference']=merged.challenger-merged.baseline
    return merged


def bootstrap_daily(daily,*,reps=1000,block_days=10,seed=42):
    """Moving-block bootstrap of paired daily portfolio returns, separately by fold."""
    if reps<10 or block_days<1:raise ValueError('Invalid bootstrap settings')
    rng=np.random.default_rng(seed);rows=[]
    for challenger in MODELS[1:]:
        for fold,part in paired_daily(daily,challenger).groupby('fold'):
            x=part.sort_values('bar_timestamp').difference.to_numpy(dtype=float)
            n=len(x)
            if n<2:continue
            draws=[]
            for _ in range(reps):
                indices=[]
                while len(indices)<n:
                    start=int(rng.integers(0,n))
                    indices.extend((start+np.arange(min(block_days,n)))%n)
                draws.append(float(x[np.array(indices[:n])].mean()))
            rows.append(dict(fold=fold,challenger=challenger,days=n,
                mean_daily_net_difference=float(x.mean()),
                ci_low=float(np.quantile(draws,.025)),ci_high=float(np.quantile(draws,.975)),
                block_days=block_days,reps=reps))
    return pd.DataFrame(rows)


def run_research(predictions_path,policies_path,*,coverage=.10,calibration='raw',
                 max_positions=10,max_per_sector=None,sector_csv=None,cooldown_days=0,
                 cost_bps=10,turnover_bps=5,reps=1000,block_days=10,output_dir=None):
    p,thresholds=load_inputs(predictions_path,policies_path,calibration=calibration,coverage=coverage)
    p=add_sector_map(p,sector_csv)
    trades,daily=select_portfolio(p,thresholds,max_positions=max_positions,
        max_per_sector=max_per_sector,cooldown_days=cooldown_days,
        cost_bps=cost_bps,turnover_bps=turnover_bps)
    scores=summarize(trades,daily)
    confidence=bootstrap_daily(daily,reps=reps,block_days=block_days)
    result={'thresholds':thresholds,'scores':scores,'trades':trades,'daily':daily,'confidence':confidence}
    if output_dir is not None:
        dest=Path(output_dir);dest.mkdir(parents=True,exist_ok=True)
        for name,data in result.items():
            if name=='trades':data.to_parquet(dest/f'{name}.parquet',index=False)
            else:data.to_csv(dest/f'{name}.csv',index=False)
    return result
