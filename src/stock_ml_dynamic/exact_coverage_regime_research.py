"""Notebook 35: retrospective exact top-k, paired selections, and date-block uncertainty.

Only uses Notebook 34 saved *development-validation* predictions. Exact top-k on
validation is diagnostic, not a deployable threshold policy.
"""
from pathlib import Path
import numpy as np
import pandas as pd

MODELS=('full_history_rf','oof_meta_logistic','oof_meta_rf')


def load_predictions(path):
    path=Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Missing Notebook 34 predictions: {path}. Run Notebook 34 first.')
    df=pd.read_parquet(path)
    required={'fold','model','calibration','bar_timestamp','symbol','direction','next_return','up_probability'}
    if not required.issubset(df):raise ValueError(f'Missing columns: {required-set(df)}')
    df=df.copy()
    df['bar_timestamp']=pd.to_datetime(df.bar_timestamp,utc=True)
    if df.duplicated(['fold','model','calibration','bar_timestamp','symbol']).any():
        raise ValueError('Duplicate fold/model/calibration/date/symbol predictions')
    if not np.isfinite(df[['next_return','up_probability']].to_numpy(dtype=float)).all():
        raise ValueError('Nonfinite predictions or returns')
    return df


def aligned_fold(frame,fold,calibration='raw'):
    parts=[]
    for model in MODELS:
        g=frame.loc[(frame.fold==fold)&(frame.calibration==calibration)&(frame.model==model),
                    ['bar_timestamp','symbol','direction','next_return','up_probability']].copy()
        if g.empty:raise ValueError(f'No predictions for {model}, fold {fold}, {calibration}')
        g=g.rename(columns={'up_probability':model})
        parts.append(g)
    out=parts[0]
    for part in parts[1:]:
        out=out.merge(part,on=['bar_timestamp','symbol'],how='inner',validate='one_to_one',suffixes=('','_check'))
        for col in ['direction','next_return']:
            other=col+'_check'
            if col=='direction' and not (out[col]==out[other]).all():raise ValueError('Target mismatch across models')
            if col=='next_return' and not np.allclose(out[col],out[other],rtol=0,atol=1e-12):raise ValueError('Return mismatch across models')
            out=out.drop(columns=other)
    if len(out)!=len(parts[0]) or any(len(out)!=len(g) for g in parts):
        raise ValueError('Prediction universes do not match: cannot claim exact paired comparison')
    return out.sort_values(['bar_timestamp','symbol']).reset_index(drop=True)


def top_k_mask(frame,model,k):
    if not 0<k<=len(frame):raise ValueError('Invalid k')
    # Stable tie break on date and symbol, identical across models.
    ix=frame.sort_values([model,'bar_timestamp','symbol'],ascending=[False,True,True],kind='mergesort').index[:k]
    return frame.index.isin(ix)


def score_subset(frame,mask,cost_bps=10):
    chosen=frame.loc[mask]
    n=len(chosen)
    return {'entries':n,'actual_coverage':n/len(frame),
            'up_precision':float((chosen.direction==1).mean()) if n else np.nan,
            'mean_net':float((chosen.next_return-cost_bps/10000).mean()) if n else np.nan,
            'net_hit_rate':float(((chosen.next_return-cost_bps/10000)>0).mean()) if n else np.nan}


def exact_coverage(frame,coverages=(.05,.10,.15,.20,.25),calibration='raw',cost_bps=10):
    rows=[];trades=[]
    for fold in sorted(frame.fold.unique()):
        g=aligned_fold(frame,fold,calibration)
        for coverage in coverages:
            k=max(1,int(round(len(g)*coverage)))
            for model in MODELS:
                mask=top_k_mask(g,model,k)
                rows.append({'fold':fold,'model':model,'calibration':calibration,
                             'target_coverage':coverage,'k':k,**score_subset(g,mask,cost_bps)})
                t=g.loc[mask,['bar_timestamp','symbol','direction','next_return']].copy()
                t['fold']=fold;t['model']=model;t['target_coverage']=coverage
                t['net_return']=t.next_return-cost_bps/10000
                trades.append(t)
    return pd.DataFrame(rows),pd.concat(trades,ignore_index=True)


def paired_selection(frame,coverages=(.05,.10,.15,.20,.25),calibration='raw',cost_bps=10):
    rows=[];trade_rows=[]
    for fold in sorted(frame.fold.unique()):
        g=aligned_fold(frame,fold,calibration)
        for coverage in coverages:
            k=max(1,int(round(len(g)*coverage)))
            base=top_k_mask(g,'full_history_rf',k)
            for challenger in MODELS[1:]:
                other=top_k_mask(g,challenger,k)
                groups={'shared':base&other,'baseline_only':base&~other,'challenger_only':~base&other}
                for group,mask in groups.items():
                    rows.append({'fold':fold,'target_coverage':coverage,'challenger':challenger,
                                 'selection':group,**score_subset(g,mask,cost_bps)})
                    t=g.loc[mask,['bar_timestamp','symbol','direction','next_return']].copy()
                    t['fold']=fold;t['target_coverage']=coverage;t['challenger']=challenger;t['selection']=group
                    t['net_return']=t.next_return-cost_bps/10000
                    trade_rows.append(t)
    return pd.DataFrame(rows),pd.concat(trade_rows,ignore_index=True)


def lagged_regimes(frame,lookback=20):
    """Descriptive proxy: lagged cross-sectional average realized next-bar return.

    For each date t, use only outcomes from dates <= t-2 (one extra
    date of embargo). Warm-up dates are excluded from regime tables.
    """
    if lookback<5:raise ValueError('lookback must be >= 5')
    keys=['fold','bar_timestamp']
    daily=frame.groupby(keys,as_index=False).next_return.mean().sort_values(keys)
    daily['lagged_market_return']=daily.groupby('fold').next_return.shift(2)
    daily['trend']=daily.groupby('fold').lagged_market_return.transform(
        lambda s:s.rolling(lookback,min_periods=lookback).mean())
    daily['volatility']=daily.groupby('fold').lagged_market_return.transform(
        lambda s:s.rolling(lookback,min_periods=lookback).std())
    # Past-only 60-day volatility reference, shifted to exclude current estimate.
    daily['prior_vol_reference']=daily.groupby('fold').volatility.transform(
        lambda s:s.shift(1).rolling(60,min_periods=20).median())
    daily['trend_regime']=np.where(daily.trend.isna(),None,np.where(daily.trend>=0,'bullish','bearish'))
    daily['vol_regime']=np.where(daily.prior_vol_reference.isna(),None,
                                  np.where(daily.volatility>=daily.prior_vol_reference,'high_vol','low_vol'))
    return daily[['fold','bar_timestamp','trend_regime','vol_regime','trend','volatility']]


def regime_metrics(trades,regimes):
    merged=trades.merge(regimes,on=['fold','bar_timestamp'],how='left',validate='many_to_one')
    rows=[]
    for kind in ['trend_regime','vol_regime']:
        g=merged.dropna(subset=[kind])
        for key,part in g.groupby(['fold','model','target_coverage',kind],dropna=False):
            fold,model,coverage,regime=key
            rows.append({'fold':fold,'model':model,'target_coverage':coverage,
                         'regime_type':kind,'regime':regime,'entries':len(part),
                         'up_precision':float((part.direction==1).mean()),
                         'mean_net':float(part.net_return.mean()),
                         'distinct_days':part.bar_timestamp.nunique()})
    return pd.DataFrame(rows)


def date_block_bootstrap(trades,coverages=(.05,.10,.15,.20,.25),reps=1000,seed=42):
    """Paired resampling of trading *dates* within fold, preserving cross-stock dependence.

    Differences compare per-selected-trade precision/net return. Folds are
    reported separately; bootstrap does not establish prospective significance.
    """
    rng=np.random.default_rng(seed)
    rows=[]
    for (fold,coverage),part in trades.groupby(['fold','target_coverage']):
        day=part.groupby(['model','bar_timestamp']).agg(n=('direction','size'),
                 up=('direction',lambda x:int((x==1).sum())),net=('net_return','sum')).reset_index()
        dates=np.sort(part.bar_timestamp.unique())
        # Every model has same k, but dates with zero entries are represented by zeros.
        vectors={}
        for model in MODELS:
            d=day[day.model==model].set_index('bar_timestamp').reindex(dates,fill_value=0)
            vectors[model]=d[['n','up','net']].to_numpy(dtype=float)
        for challenger in MODELS[1:]:
            a=vectors[challenger];b=vectors['full_history_rf']
            point=np.array([a[:,1].sum()/a[:,0].sum()-b[:,1].sum()/b[:,0].sum(),
                            a[:,2].sum()/a[:,0].sum()-b[:,2].sum()/b[:,0].sum()])
            boot=np.empty((reps,2))
            for r in range(reps):
                ix=rng.integers(0,len(dates),size=len(dates))
                aa=a[ix].sum(axis=0);bb=b[ix].sum(axis=0)
                boot[r]=[aa[1]/aa[0]-bb[1]/bb[0],aa[2]/aa[0]-bb[2]/bb[0]] if aa[0] and bb[0] else [np.nan,np.nan]
            for j,metric in enumerate(['up_precision_difference','mean_net_difference']):
                valid=boot[:,j][np.isfinite(boot[:,j])]
                rows.append({'fold':fold,'target_coverage':coverage,'challenger':challenger,
                             'metric':metric,'point_estimate':point[j],
                             'ci_low':np.quantile(valid,.025) if len(valid) else np.nan,
                             'ci_high':np.quantile(valid,.975) if len(valid) else np.nan,
                             'days':len(dates),'bootstrap_reps':reps})
    return pd.DataFrame(rows)


def run_research(path,output_dir=None,coverages=(.05,.10,.15,.20,.25),reps=1000,seed=42):
    frame=load_predictions(path)
    # Ranking-preserving Platt transforms should select the same top-k; raw is primary.
    scores,trades=exact_coverage(frame,coverages)
    paired,paired_trades=paired_selection(frame,coverages)
    regimes=lagged_regimes(aligned_all(frame))
    regime=regime_metrics(trades,regimes)
    bootstrap=date_block_bootstrap(trades,coverages,reps,seed)
    results={'exact_scores':scores,'selected_trades':trades,'paired':paired,
             'paired_trades':paired_trades,'regimes':regimes,'regime_scores':regime,
             'bootstrap':bootstrap}
    if output_dir:
        output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
        for name,data in results.items():
            if name.endswith('trades'):data.to_parquet(output_dir/f'{name}.parquet',index=False)
            else:data.to_csv(output_dir/f'{name}.csv',index=False)
    return results


def aligned_all(frame):
    return pd.concat([aligned_fold(frame,fold).assign(fold=fold) for fold in sorted(frame.fold.unique())],ignore_index=True)
