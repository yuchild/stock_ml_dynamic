import numpy as np,pandas as pd
from stock_ml_dynamic.edge_robustness_41 import rank_with_exclusions,score_variants,tail_and_integrity

def test_refills_after_exclusion():
 d=pd.DataFrame(dict(bar_timestamp=['2026-01-01']*4,symbol=['A','B','C','D'],next_return=[.1,.2,.3,.4]))
 s=np.array([4.,3.,2.,1.])
 a=rank_with_exclusions(d,s,2,['A']);assert a.symbol.tolist()==['B','C']

def test_rank_preservation():
 d=pd.DataFrame(dict(fold=[1]*3,bar_timestamp=['2026-01-01']*3,symbol=['A','B','C'],next_return=[.1,.2,.3],score=[3.,2.,1.]))
 z=score_variants(d);assert z.sort_values('score',ascending=False).symbol.tolist()==z.sort_values('score_rank',ascending=False).symbol.tolist()

def test_integrity_flags():
 d=pd.DataFrame(dict(fold=[1,1],bar_timestamp=['2026-01-01']*2,symbol=['A','A'],next_return=[.5,-.3],score=[1.,2.]))
 t=d.copy();t['model']='ridge_all';t['k']=5
 q,_,_=tail_and_integrity(d,t);assert q.duplicate_symbol_dates.iloc[0]==1;assert q.abs_return_over_20pct.iloc[0]==2


def test_paired_inference_exposes_portfolio_size():
 from stock_ml_dynamic.edge_robustness_41 import paired_inference
 rows=[]
 for day in range(12):
  for model,ret in [('ridge_all',0.01),('exclude_CVNA',0.012)]:
   rows.append(dict(fold=1,model=model,k=5,bar_timestamp=f'2026-01-{day+1:02d}',daily_return=ret))
 result=paired_inference(pd.DataFrame(rows),reps=20)
 assert 'k' in result.columns
 assert result['k'].eq(5).all()
 assert len(result)==1

def test_run_portfolio_stress_accepts_raw_fitted_scores():
 from stock_ml_dynamic.edge_robustness_41 import run_portfolio_stress
 rows=[]
 for date in ('2026-01-01','2026-01-02'):
  for i,symbol in enumerate(('A','B','C','D')):
   rows.append(dict(fold=1,bar_timestamp=date,symbol=symbol,next_return=.01*i,score=float(i)))
 result=run_portfolio_stress(pd.DataFrame(rows),k_values=(2,))
 metrics=result['metrics']
 assert {'ridge_all','rank_pct','score_z'}.issubset(set(metrics.model))
 daily=result['daily'].pivot(index='bar_timestamp',columns='model',values='daily_return')
 assert np.allclose(daily['ridge_all'],daily['rank_pct'])
 assert np.allclose(daily['ridge_all'],daily['score_z'])
