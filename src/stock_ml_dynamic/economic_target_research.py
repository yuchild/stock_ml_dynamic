"""Notebook 37: leakage-controlled, economically aligned target comparison."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score, mean_absolute_error, mean_squared_error, r2_score, brier_score_loss
from .walk_forward import expanding_folds, fold_data
from .rf_deep_research import probability_matrix


def prepare_extras(train, test, columns):
    """Fit all outlier bounds and missing-value medians on training only."""
    a=train[columns].replace([np.inf,-np.inf],np.nan).apply(pd.to_numeric,errors='coerce')
    b=test[columns].replace([np.inf,-np.inf],np.nan).apply(pd.to_numeric,errors='coerce')
    lo=a.quantile(.005); hi=a.quantile(.995)
    a=a.clip(lower=lo,upper=hi,axis=1)
    b=b.clip(lower=lo,upper=hi,axis=1)
    imp=SimpleImputer(strategy='median',keep_empty_features=True)
    return imp.fit_transform(a).astype('float32'),imp.transform(b).astype('float32')


def choose_positions(frame, score, eligible, max_positions, cost):
    """Fixed threshold decided before evaluation; top scores among eligible names on each day."""
    x=frame[['bar_timestamp','symbol','next_return']].copy()
    x['score']=np.asarray(score,float)
    x['direction_label']=frame['direction'].to_numpy(dtype=int) if 'direction' in frame else -1
    x['eligible']=np.asarray(eligible,bool)
    x['bar_timestamp']=pd.to_datetime(x.bar_timestamp,utc=True)
    x=x.sort_values(['bar_timestamp','score','symbol'],ascending=[True,False,True])
    x['rank']=x.groupby('bar_timestamp')['eligible'].cumsum()
    x['selected']=x.eligible & (x['rank']<=max_positions)
    x['net_return']=x.next_return-cost
    x['portfolio_contribution']=np.where(x.selected,x.net_return/max_positions,0.)
    daily=x.groupby('bar_timestamp',sort=True).agg(portfolio_net=('portfolio_contribution','sum'),n_positions=('selected','sum'))
    daily['utilization']=daily.n_positions/max_positions
    return x,daily.reset_index()


def _auc_binary(y,score):
    try:return float(roc_auc_score(y,score))
    except ValueError:return np.nan


def run_target_comparison(frame, feature_cols, *, n_folds=3,cap=60000,rf_trees=160,cost_bps=10,max_positions=10,seed=42,output_dir=None):
    if n_folds!=3:raise ValueError('Use three predeclared outer folds')
    if cap<100 or max_positions<1:raise ValueError('Invalid capacity')
    cost=cost_bps/10000.
    predictive=[];portfolio=[];selection=[];daily_rows=[];audit=[]
    for fold in expanding_folds(frame,n_folds=n_folds):
        tr,te,xb,xv,_=fold_data(frame,fold,target_q=7)
        assert pd.to_datetime(tr.next_bar_timestamp,utc=True).max()<=fold.train_end
        assert pd.to_datetime(te.bar_timestamp,utc=True).min()>fold.train_end
        ax,az=prepare_extras(tr,te,feature_cols)
        X=np.column_stack([xb.astype('float32'),ax]); Z=np.column_stack([xv.astype('float32'),az])
        rng=np.random.default_rng(seed+fold.number)
        idx=np.sort(rng.choice(len(X),size=min(cap,len(X)),replace=False))
        Xtr=X[idx]
        ydir=tr.direction.to_numpy(dtype=int)[idx]
        ret_tr=tr.next_return.to_numpy(dtype=float)[idx]
        ret_te=te.next_return.to_numpy(dtype=float)
        ynet=(ret_tr>cost).astype(int)
        clf=RandomForestClassifier(n_estimators=rf_trees,max_depth=12,min_samples_leaf=20,max_features='sqrt',n_jobs=-1,random_state=seed+fold.number)
        clf.fit(Xtr,ydir)
        p=probability_matrix(clf,Z)
        # Regressor's target is winsorized using training-only percentiles, never test outcomes.
        low,high=np.quantile(ret_tr,[.005,.995])
        reg=HistGradientBoostingRegressor(max_iter=150,max_leaf_nodes=15,learning_rate=.05,l2_regularization=1,random_state=seed+fold.number)
        reg.fit(Xtr,np.clip(ret_tr,low,high))
        pred_ret=reg.predict(Z)
        eco=HistGradientBoostingClassifier(max_iter=150,max_leaf_nodes=15,learning_rate=.05,l2_regularization=1,random_state=seed+fold.number)
        eco.fit(Xtr,ynet)
        prob_net=eco.predict_proba(Z)[:,list(eco.classes_).index(1)]
        ytest=te.direction.to_numpy(dtype=int)
        net_test=(ret_te>cost).astype(int)
        try: multi_auc=roc_auc_score(ytest,p,labels=[0,1,2],multi_class='ovr',average='macro')
        except ValueError: multi_auc=np.nan
        predictive.extend([
            dict(fold=fold.number,model='direction_rf',accuracy=accuracy_score(ytest,p.argmax(1)),auc=multi_auc,mae=np.nan,rmse=np.nan,r2=np.nan,brier_net=np.nan),
            dict(fold=fold.number,model='return_regression',accuracy=np.nan,auc=np.nan,mae=mean_absolute_error(ret_te,pred_ret),rmse=np.sqrt(mean_squared_error(ret_te,pred_ret)),r2=r2_score(ret_te,pred_ret),brier_net=np.nan),
            dict(fold=fold.number,model='cost_aware_binary',accuracy=accuracy_score(net_test,prob_net>=.5),auc=_auc_binary(net_test,prob_net),mae=np.nan,rmse=np.nan,r2=np.nan,brier_net=brier_score_loss(net_test,prob_net))])
        policies={'direction_rf':(p[:,1],(p.argmax(1)==1)&(p[:,1]>=.40)),
                  'return_regression':(pred_ret,pred_ret>cost),
                  'cost_aware_binary':(prob_net,prob_net>=.5)}
        for name,(score,eligible) in policies.items():
            picks,daily=choose_positions(te,score,eligible,max_positions,cost)
            picks['fold']=fold.number;picks['model']=name
            daily['fold']=fold.number;daily['model']=name
            daily_rows.append(daily)
            chosen=picks[picks.selected]
            n=len(chosen)
            portfolio.append(dict(fold=fold.number,model=name,days=len(daily),trades=n,
                                  utilization=daily.utilization.mean(),mean_daily_net=daily.portfolio_net.mean(),
                                  total_compounded_net=np.prod(1+daily.portfolio_net)-1,
                                  max_drawdown=((1+daily.portfolio_net).cumprod()/(1+daily.portfolio_net).cumprod().cummax()-1).min(),
                                  mean_trade_net=chosen.net_return.mean() if n else np.nan,
                                  positive_net_rate=(chosen.net_return>0).mean() if n else np.nan))
            selection.append(dict(fold=fold.number,model=name,selected=n,
                                  mean_gross_return=chosen.next_return.mean() if n else np.nan,
                                  mean_net_return=chosen.net_return.mean() if n else np.nan,
                                  q7_up_precision=(chosen.direction_label==1).mean() if n else np.nan,
                                  net_positive_precision=(chosen.net_return>0).mean() if n else np.nan))
        audit.append(dict(fold=fold.number,train_rows=len(tr),validation_rows=len(te),
                          train_last_label=pd.to_datetime(tr.next_bar_timestamp,utc=True).max(),
                          validation_first_bar=pd.to_datetime(te.bar_timestamp,utc=True).min(),
                          train_cap=len(idx),validation_positive_net_base_rate=net_test.mean()))
    daily=pd.concat(daily_rows,ignore_index=True)
    pivot=daily.pivot_table(index=['fold','bar_timestamp'],columns='model',values='portfolio_net')
    pairs=[]
    for challenger in ['return_regression','cost_aware_binary']:
        for fold,g in pivot.groupby(level='fold'):
            delta=g[challenger]-g['direction_rf']
            pairs.append(dict(fold=fold,challenger=challenger,days=len(delta),mean_daily_delta=delta.mean(),
                              positive_days=(delta>0).mean(),std_daily_delta=delta.std()))
    out=dict(predictive=pd.DataFrame(predictive),portfolio=pd.DataFrame(portfolio),
             selection=pd.DataFrame(selection),daily=daily,paired_daily=pd.DataFrame(pairs),audit=pd.DataFrame(audit))
    if output_dir is not None:
        path=Path(output_dir);path.mkdir(parents=True,exist_ok=True)
        for key,df in out.items():df.to_csv(path/f'{key}.csv',index=False)
    return out


def summarize_research(result):
    p=result['portfolio'].groupby('model')[['mean_daily_net','mean_trade_net','utilization']].mean()
    winner=p.mean_daily_net.idxmax()
    return (f'Highest mean daily net return in this experiment: {winner}. '
            'This is a development-fold comparison, not evidence of deployable profitability. '
            'Inspect all fold-level results and execution assumptions before drawing conclusions.')
