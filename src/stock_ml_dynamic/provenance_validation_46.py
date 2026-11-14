"""Notebook 46: non-mutating evidence audit. No access to final holdout."""
import ast
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd

CRITICAL=('master_table.py','candle_table.py','historical_market.py','accuracy_features.py','volume_regime_research.py','walk_forward.py','economic_target_research.py','edge_robustness_41.py')

def fingerprint(paths):
    rows=[]
    for p in map(Path,paths):
        if p.is_file():
            rows.append(dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    return pd.DataFrame(rows)

def code_inventory(package_dir):
    rows=[]
    for name in CRITICAL:
        path=Path(package_dir)/name
        if not path.exists():continue
        source=path.read_text();tree=ast.parse(source)
        for node in ast.walk(tree):
            if not isinstance(node,ast.Call):continue
            func=ast.unparse(node.func)
            interesting=('shift','rolling','fit','fit_transform','bfill','merge_asof','pct_change','quantile','transform','groupby')
            if not any(x in func for x in interesting):continue
            rows.append(dict(file=name,line=node.lineno,call=func,expression=ast.get_source_segment(source,node)[:220],review=('REVIEW' if any(x in func for x in ('shift','bfill','fit','merge_asof')) else 'INVENTORY')))
    return pd.DataFrame(rows)

def master_alignment(master):
    x=master.copy();x['bar_timestamp']=pd.to_datetime(x.bar_timestamp,utc=True);x['next_bar_timestamp']=pd.to_datetime(x.next_bar_timestamp,utc=True)
    x=x.sort_values(['symbol','bar_timestamp'])
    actual_next=x.groupby('symbol').bar_timestamp.shift(-1)
    valid=actual_next.notna()
    diff=(x.loc[valid,'next_bar_timestamp']!=actual_next.loc[valid])
    rows=[dict(check='duplicate_symbol_time',failures=int(x.duplicated(['symbol','bar_timestamp']).sum()),scope='master'),dict(check='nonfuture_next_timestamp',failures=int((x.next_bar_timestamp<=x.bar_timestamp).sum()),scope='master'),dict(check='next_timestamp_mismatch_internal',failures=int(diff.sum()),scope=f'{int(valid.sum())} rows with a following master row'),dict(check='nonfinite_next_return',failures=int((~np.isfinite(x.next_return.to_numpy(float))).sum()),scope='master')]
    return pd.DataFrame(rows)

def boundary_checks(frame,folds):
    out=[]
    for f in folds:
        tr=frame.loc[frame.bar_timestamp<=f.train_end]
        val=frame.loc[(frame.bar_timestamp>=f.validation_start)&(frame.bar_timestamp<=f.validation_end)] if hasattr(f,'validation_start') else frame.loc[frame.bar_timestamp>f.train_end]
        out.append(dict(fold=f.number,train_end=str(f.train_end),training_rows=len(tr),validation_rows=len(val),train_labels_after_boundary=int((pd.to_datetime(tr.next_bar_timestamp,utc=True)>f.train_end).sum()),train_max_label=str(pd.to_datetime(tr.next_bar_timestamp,utc=True).max()),first_validation=str(pd.to_datetime(val.bar_timestamp,utc=True).min())))
    return pd.DataFrame(out)

def universe_coverage(master):
    x=master.copy();x['bar_timestamp']=pd.to_datetime(x.bar_timestamp,utc=True)
    start=x.groupby('symbol').bar_timestamp.min();end=x.groupby('symbol').bar_timestamp.max()
    return pd.DataFrame({'symbol':start.index,'first_seen':start.values,'last_seen':end.values,'bars':x.groupby('symbol').size().values}).sort_values(['bars','symbol'])

def event_candidates(master,threshold=.2):
    x=master.copy();x=x.loc[x.next_return.abs()>threshold].copy()
    columns=[c for c in ['symbol','bar_timestamp','next_bar_timestamp','adj_close','next_adj_close','close','volume','next_return'] if c in x]
    return x[columns].sort_values('next_return',key=lambda s:s.abs(),ascending=False).reset_index(drop=True)

def compare_raw_to_master(master,raw):
    """Compare raw source rows to engineered master using timestamps and adjusted close."""
    x=master[['symbol','bar_timestamp','adj_close']].copy();x['bar_timestamp']=pd.to_datetime(x.bar_timestamp,utc=True)
    y=raw[['symbol','bar_timestamp','adj_close_price']].copy();y['bar_timestamp']=pd.to_datetime(y.bar_timestamp,utc=True)
    z=x.merge(y,on=['symbol','bar_timestamp'],how='left',indicator=True)
    a=pd.to_numeric(z.adj_close,errors='coerce');b=pd.to_numeric(z.adj_close_price,errors='coerce')
    delta=(a-b).abs()
    return pd.DataFrame([dict(master_rows=len(z),missing_raw_rows=int(z.adj_close_price.isna().sum()),price_mismatches_gt_1e_8=int((delta>1e-8).sum()),max_abs_difference=float(delta.max()) if delta.notna().any() else np.nan)])

def decision_gate(alignment,source_available,verified_universe=False,verified_corporate_actions=False,verified_feature_timing=False):
    issues=int(alignment.failures.sum())
    return pd.DataFrame([dict(check='internal_alignment',status='PASS' if issues==0 else 'FAIL',evidence=str(issues)+' violations'),dict(check='raw_source_comparison',status='REVIEWED' if source_available else 'UNVERIFIED',evidence='local raw PostgreSQL comparison only; vendor correctness not established'),dict(check='historical_universe',status='VERIFIED' if verified_universe else 'UNVERIFIED',evidence='point-in-time membership and delistings required'),dict(check='corporate_actions',status='VERIFIED' if verified_corporate_actions else 'UNVERIFIED',evidence='independent split/dividend source required'),dict(check='feature_timing',status='VERIFIED' if verified_feature_timing else 'UNVERIFIED',evidence='manual decision-time audit required'),dict(check='final_holdout',status='CLOSED',evidence='not read')])
