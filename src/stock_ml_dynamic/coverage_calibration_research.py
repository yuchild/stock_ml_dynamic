"""Notebook 34: nested chronological calibration and matched-coverage diagnostics.

No final historical holdout access, no prospective ledger writes.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from .oof_stacking_research import run_fold
from .classifier_benchmark import eligible_universe
from .feature_ablation import development_holdout_split
from .historical_market import add_historical_market_features
from .accuracy_features import add_accuracy_features
from .volume_regime_research import add_volume_regime_features
from .walk_forward import expanding_folds, fold_data

MODELS=('full_history_rf','oof_meta_logistic','oof_meta_rf')


def chronological_calibration_split(frame, fraction=.25):
    """Split by distinct bar dates; exclude training rows whose labels overlap calibration."""
    if not 0<fraction<.5:raise ValueError('fraction must be between 0 and .5')
    ts=pd.to_datetime(frame.bar_timestamp,utc=True)
    outcome=pd.to_datetime(frame.next_bar_timestamp,utc=True)
    dates=np.sort(ts.dropna().unique())
    if len(dates)<50:raise ValueError('Insufficient dates')
    start=dates[int(len(dates)*(1-fraction))]
    fit=frame.loc[outcome<start].copy()
    cal=frame.loc[ts>=start].copy()
    if len(fit)<100 or len(cal)<100:raise ValueError('Insufficient chronological fit/calibration rows')
    assert pd.to_datetime(fit.next_bar_timestamp,utc=True).max()<pd.to_datetime(cal.bar_timestamp,utc=True).min()
    return fit,cal


def calibrate_up(p_cal,y_cal,p_eval):
    """Train-only Platt mapping of class-1 probability; evaluation labels are never used."""
    p_cal=np.asarray(p_cal,dtype=float)
    p_eval=np.asarray(p_eval,dtype=float)
    y=np.asarray(y_cal,dtype=int)==1
    def logit(x):
        x=np.clip(x,1e-6,1-1e-6)
        return np.log(x/(1-x)).reshape(-1,1)
    if np.unique(y).size<2:raise ValueError('Calibration labels contain one class')
    calibrator=LogisticRegression(C=1.0,max_iter=500)
    calibrator.fit(logit(p_cal),y)
    return calibrator.predict_proba(logit(p_eval))[:,1],calibrator.predict_proba(logit(p_cal))[:,1]


def threshold_for_coverage(scores,coverage):
    """Choose score threshold using calibration-period scores only."""
    scores=np.asarray(scores,dtype=float)
    if not 0<coverage<1:raise ValueError('coverage must be in (0,1)')
    if not np.isfinite(scores).all():raise ValueError('Nonfinite score')
    return float(np.quantile(scores,1-coverage))


def policy_metrics(frame,score,threshold,*,cost_bps=10):
    y=frame.direction.to_numpy(dtype=int)
    returns=frame.next_return.to_numpy(dtype=float)
    score=np.asarray(score,dtype=float)
    mask=score>=threshold
    n=int(mask.sum())
    net=returns[mask]-cost_bps/10000
    return dict(n=len(frame),entry_count=n,coverage=float(mask.mean()),
                up_precision=float(np.mean(y[mask]==1)) if n else np.nan,
                mean_net=float(net.mean()) if n else np.nan,
                median_net=float(np.median(net)) if n else np.nan,
                hit_rate_net=float(np.mean(net>0)) if n else np.nan)


def reliability_table(frame,predictions,*,fold,model,subset,bins=10):
    y=(frame.direction.to_numpy(dtype=int)==1)
    p=np.asarray(predictions,dtype=float)
    # fixed probability bins avoid peeking at evaluation outcomes for boundaries
    ids=np.minimum(np.floor(np.clip(p,0,1)*bins).astype(int),bins-1)
    rows=[]
    for i in range(bins):
        mask=ids==i
        rows.append(dict(fold=fold,model=model,subset=subset,bin=i,
                         lower=i/bins,upper=(i+1)/bins,n=int(mask.sum()),
                         predicted_up=float(p[mask].mean()) if mask.any() else np.nan,
                         observed_up=float(y[mask].mean()) if mask.any() else np.nan))
    return rows


def run_research(frame,*,n_outer=3,windows=5,cap=60000,trees=200,leaf=20,
                 seed=42,clip=(.005,.995),coverages=(.05,.10,.15,.20,.25),
                 cost_bps=10,calibration_fraction=.25,output_dir=None):
    eligible,_=eligible_universe(frame)
    dev,cutoff=development_holdout_split(eligible,.2)
    expanded,hist=add_historical_market_features(dev)
    expanded,tech=add_accuracy_features(expanded)
    expanded,vol=add_volume_regime_features(expanded)
    cols=list(dict.fromkeys(hist+tech+vol))
    rows=[];audit=[];reliability=[];pred_records=[]
    for outer in expanding_folds(expanded,n_folds=n_outer):
        tr,va,_,_,_=fold_data(expanded,outer,target_q=7)
        fit,cal=chronological_calibration_split(tr,calibration_fraction)
        # Inner models see only the pre-calibration portion of outer training.
        _,inner_audit,inner_preds=run_fold(fit,cal,cols,fold=outer.number,windows=windows,
                cap=cap,trees=trees,leaf=leaf,seed=seed,clip=clip,
                cost_bps=cost_bps,return_predictions=True)
        # Outer models see all outer-training history, but no validation labels.
        _,outer_audit,outer_preds=run_fold(tr,va,cols,fold=outer.number,windows=windows,
                cap=cap,trees=trees,leaf=leaf,seed=seed,clip=clip,
                cost_bps=cost_bps,return_predictions=True)
        audit.append(dict(fold=outer.number,fit_rows=len(fit),cal_rows=len(cal),
                          outer_train_rows=len(tr),outer_valid_rows=len(va),
                          latest_fit_outcome=str(pd.to_datetime(fit.next_bar_timestamp,utc=True).max()),
                          earliest_cal_bar=str(pd.to_datetime(cal.bar_timestamp,utc=True).min()),
                          latest_outer_train_outcome=str(pd.to_datetime(tr.next_bar_timestamp,utc=True).max()),
                          earliest_outer_valid_bar=str(pd.to_datetime(va.bar_timestamp,utc=True).min()),
                          inner_oof_windows=len(inner_audit),outer_oof_windows=len(outer_audit)))
        for name in MODELS:
            cal_raw=inner_preds[name][:,1]
            eval_raw=outer_preds[name][:,1]
            eval_platt,cal_platt=calibrate_up(cal_raw,cal.direction,eval_raw)
            for variant,cal_scores,eval_scores in [('raw',cal_raw,eval_raw),('platt',cal_platt,eval_platt)]:
                reliability.extend(reliability_table(va,eval_scores,fold=outer.number,
                                                     model=name,subset=variant))
                brier=brier_score_loss(va.direction.to_numpy()==1,eval_scores)
                binary_ll=log_loss((va.direction.to_numpy()==1).astype(int),
                                   np.clip(eval_scores,1e-8,1-1e-8),labels=[0,1])
                for coverage in coverages:
                    cutoff_score=threshold_for_coverage(cal_scores,coverage)
                    result=policy_metrics(va,eval_scores,cutoff_score,cost_bps=cost_bps)
                    rows.append(dict(fold=outer.number,model=name,calibration=variant,
                                     target_coverage=coverage,threshold=cutoff_score,
                                     brier_up=brier,binary_logloss=binary_ll,**result))
                pred_records.append(pd.DataFrame(dict(fold=outer.number,model=name,
                      calibration=variant,bar_timestamp=va.bar_timestamp.to_numpy(),
                      symbol=va.symbol.to_numpy(),direction=va.direction.to_numpy(),
                      next_return=va.next_return.to_numpy(),up_probability=eval_scores)))
    result={'policies':pd.DataFrame(rows),'audit':pd.DataFrame(audit),
            'reliability':pd.DataFrame(reliability),
            'predictions':pd.concat(pred_records,ignore_index=True),
            'cutoff':cutoff,'feature_count':len(cols)}
    if output_dir is not None:
        dest=Path(output_dir);dest.mkdir(parents=True,exist_ok=True)
        for name in ('policies','audit','reliability'):
            result[name].to_csv(dest/f'{name}.csv',index=False)
        result['predictions'].to_parquet(dest/'validation_predictions.parquet',index=False)
    return result
