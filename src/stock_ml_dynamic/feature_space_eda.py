"""Read-only, development-only feature-space EDA utilities for Notebook 30."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, r2_score, mean_absolute_error

IDENTIFIERS={'symbol','interval_code','bar_timestamp','next_bar_timestamp'}
OUTCOMES={'direction','next_return','next_adj_close','next_close','target','label'}
RAW={'open_price','high_price','low_price','close_price','adj_close','volume'}

def feature_inventory(frame, engineered=None):
    """Separate model features from identifiers, outcomes, and raw prices."""
    if engineered is not None:
        chosen=[c for c in dict.fromkeys(engineered) if c in frame.columns]
    else:
        chosen=[c for c in frame if c not in IDENTIFIERS|OUTCOMES|RAW and not c.startswith('_')]
    numeric=[c for c in chosen if pd.api.types.is_numeric_dtype(frame[c]) and not pd.api.types.is_bool_dtype(frame[c])]
    categorical=[c for c in chosen if c not in numeric]
    return numeric,categorical

def numeric_profile(frame,cols):
    x=frame[cols].apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan)
    out=x.describe(percentiles=[.01,.05,.25,.5,.75,.95,.99]).T
    out['missing_pct']=x.isna().mean()*100
    out['zero_pct']=x.eq(0).mean()*100
    out['unique']=x.nunique()
    out['skew']=x.skew()
    return out.sort_values('missing_pct',ascending=False)

def correlation_pairs(corr, top=30):
    """Rank unique off-diagonal correlations without relying on index level names.

    Stock correlation matrices often have both axis names set to ``symbol``;
    ``stack().reset_index()`` fails because it attempts to insert that name twice.
    """
    if corr.shape[0] != corr.shape[1]:
        raise ValueError('Correlation matrix must be square')
    i, j = np.triu_indices(len(corr), k=1)
    values = corr.to_numpy()[i, j]
    valid = np.isfinite(values)
    pairs = pd.DataFrame({
        'feature_a': corr.index.to_numpy()[i[valid]],
        'feature_b': corr.columns.to_numpy()[j[valid]],
        'pearson_r': values[valid],
    })
    pairs['abs_r'] = pairs['pearson_r'].abs()
    return pairs.sort_values('abs_r', ascending=False).head(top).reset_index(drop=True)

def time_split(frame,train_frac=.75):
    """Strictly chronological split with one-bar label embargo via next_bar_timestamp."""
    if not 0<train_frac<1:raise ValueError('train_frac must be between 0 and 1')
    d=frame.copy();d['bar_timestamp']=pd.to_datetime(d.bar_timestamp,utc=True)
    unique=np.sort(d.bar_timestamp.unique());cutoff=pd.Timestamp(unique[max(1,int(len(unique)*train_frac))-1])
    train=(pd.to_datetime(d.next_bar_timestamp,utc=True)<=cutoff) if 'next_bar_timestamp' in d else (d.bar_timestamp<cutoff)
    test=d.bar_timestamp>cutoff
    if not train.any() or not test.any():raise ValueError('No usable chronological train/test split')
    return train,test,cutoff

def regularization_paths(frame,features,train,test,cs=(.001,.01,.1,1.,10.),max_rows=70000,seed=42):
    """Multinomial-style L1/L2 logistic classification and Ridge return regression.

    L1 and L2 denote regularization penalties, not ordinary least-squares L1/L2 losses.
    All imputation and scaling are fitted on training observations only.
    """
    d=frame.copy();d=d[d.direction.notna()].copy()
    tr=train.reindex(d.index,fill_value=False);te=test.reindex(d.index,fill_value=False)
    if tr.sum()>max_rows:
        ix=np.random.default_rng(seed).choice(d.index[tr].to_numpy(),max_rows,replace=False)
        tr=tr & d.index.isin(ix)
    x=d[features].apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan)
    y=d.direction.astype(int)
    records=[];coeff=[]
    for penalty in ('l1','l2'):
        for c in cs:
            pipe=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler(),
                LogisticRegression(penalty=penalty,C=c,solver='saga',max_iter=500,tol=.005,random_state=seed))
            pipe.fit(x.loc[tr],y.loc[tr]);pred=pipe.predict(x.loc[te]);p=pipe.predict_proba(x.loc[te]);classes=pipe[-1].classes_
            try:auc=roc_auc_score(y.loc[te],p,labels=classes,multi_class='ovr',average='macro')
            except ValueError:auc=np.nan
            records.append(dict(model=f'logistic_{penalty}',C=c,accuracy=accuracy_score(y.loc[te],pred),auc=auc,
                nonzero=int(np.count_nonzero(np.abs(pipe[-1].coef_)>1e-5)),train_rows=int(tr.sum()),test_rows=int(te.sum())))
            for k,cls in enumerate(classes):
                for name,value in zip(features,pipe[-1].coef_[k]):coeff.append(dict(model=penalty,C=c,cls=int(cls),feature=name,coefficient=float(value)))
    if 'next_return' in d:
        target=pd.to_numeric(d.next_return,errors='coerce');ok=target.notna()&np.isfinite(target)
        for alpha in (.1,1.,10.,100.):
            r=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler(),Ridge(alpha=alpha))
            r.fit(x.loc[tr&ok],target.loc[tr&ok]);prediction=r.predict(x.loc[te&ok]);actual=target.loc[te&ok]
            records.append(dict(model='ridge_return_l2',alpha=alpha,r2=r2_score(actual,prediction),mae=mean_absolute_error(actual,prediction),
                train_rows=int((tr&ok).sum()),test_rows=int((te&ok).sum())))
    return pd.DataFrame(records),pd.DataFrame(coeff)

def aligned_returns(frame,min_overlap=80):
    """Within-symbol adjusted-close returns aligned on exact timestamps; no forward fill."""
    d=frame[['symbol','bar_timestamp','adj_close']].copy()
    d['bar_timestamp']=pd.to_datetime(d.bar_timestamp,utc=True)
    d=d.sort_values(['symbol','bar_timestamp'])
    d['return']=d.groupby('symbol').adj_close.pct_change(fill_method=None)
    wide=d.pivot_table(index='bar_timestamp',columns='symbol',values='return',aggfunc='first')
    eligible=wide.columns[wide.notna().sum()>=min_overlap]
    return wide[eligible]

def within_sector_summary(wide,sectors,min_overlap=80):
    """Pairwise Pearson returns correlations, with explicit overlap counts."""
    sectors=sectors.drop_duplicates('symbol').set_index('symbol')['sector']
    groups={s:[t for t in tickers if t in wide.columns] for s,tickers in sectors.groupby(sectors).groups.items()}
    rows=[]
    for sector,tickers in groups.items():
        if len(tickers)<2:continue
        x=wide[tickers];corr=x.corr(min_periods=min_overlap);n=x.notna().astype(int).T.dot(x.notna().astype(int))
        mask=np.triu(np.ones(corr.shape,dtype=bool),1)
        for i,j in zip(*np.where(mask)):
            if pd.notna(corr.iat[i,j]):rows.append(dict(sector=sector,symbol_a=tickers[i],symbol_b=tickers[j],r=float(corr.iat[i,j]),overlap=int(n.iat[i,j])))
    return pd.DataFrame(rows)

def sector_return_series(wide,sectors,min_constituents=2):
    """Equal-weight same-timestamp sector return proxy (survivorship caveat)."""
    m=sectors.drop_duplicates('symbol').set_index('symbol')['sector']
    result={}
    for sec in sorted(m.dropna().unique()):
        symbols=[s for s in m.index[m==sec] if s in wide.columns]
        if len(symbols)<min_constituents:continue
        x=wide[symbols]
        result[sec]=x.mean(axis=1).where(x.notna().sum(axis=1)>=min_constituents)
    return pd.DataFrame(result,index=wide.index)
