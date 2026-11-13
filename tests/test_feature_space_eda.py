import numpy as np
import pandas as pd
from stock_ml_dynamic.feature_space_eda import feature_inventory,correlation_pairs,time_split,aligned_returns,within_sector_summary,sector_return_series

def test_inventory_excludes_targets():
    d=pd.DataFrame({'symbol':['A'],'direction':[1],'next_return':[.1],'volume':[10],'f':[.3],'category':['a']})
    assert feature_inventory(d)==(['f'],['category'])

def test_pairs():
    c=pd.DataFrame([[1.,.8],[.8,1.]],index=['a','b'],columns=['a','b'])
    assert np.isclose(correlation_pairs(c).iloc[0].pearson_r,.8)

def test_embargo():
    t=pd.date_range('2025-01-01',periods=11,tz='UTC')
    d=pd.DataFrame({'bar_timestamp':t[:-1],'next_bar_timestamp':t[1:]})
    tr,te,cut=time_split(d,.7)
    assert d.loc[tr,'next_bar_timestamp'].max()<=cut
    assert d.loc[te,'bar_timestamp'].min()>cut

def test_aligned_returns_not_price_levels():
    t=pd.date_range('2025-01-01',periods=5,tz='UTC')
    d=pd.DataFrame({'symbol':['A']*5+['B']*5,'bar_timestamp':list(t)*2,'adj_close':[100,110,121,133.1,146.41,50,55,60.5,66.55,73.205]})
    w=aligned_returns(d,min_overlap=3)
    assert w.shape[1]==2 and np.isclose(w['A'].iloc[1],.1)

def test_sector_pairs():
    t=pd.date_range('2025-01-01',periods=100,tz='UTC')
    w=pd.DataFrame({'A':np.arange(100)/1000,'B':np.arange(100)/1000,'C':-np.arange(100)/1000},index=t)
    s=pd.DataFrame({'symbol':['A','B','C'],'sector':['Tech','Tech','Other']})
    p=within_sector_summary(w,s,80)
    assert len(p)==1 and p.iloc[0].sector=='Tech' and np.isclose(p.iloc[0].r,1)
    assert 'Tech' in sector_return_series(w,s).columns


def test_pairs_duplicate_axis_names_and_nan():
    c = pd.DataFrame([[1., .8, np.nan], [.8, 1., -.4], [np.nan, -.4, 1.]],
                     index=pd.Index(['A','B','C'], name='symbol'),
                     columns=pd.Index(['A','B','C'], name='symbol'))
    pairs = correlation_pairs(c, 35)
    assert len(pairs) == 2
    assert pairs.iloc[0]['feature_a'] == 'A'
    assert pairs.iloc[0]['feature_b'] == 'B'
    assert np.isclose(pairs.iloc[0]['pearson_r'], .8)
