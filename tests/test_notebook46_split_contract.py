import json
from pathlib import Path
import pandas as pd
from stock_ml_dynamic.classifier_benchmark import eligible_universe
from stock_ml_dynamic.feature_ablation import development_holdout_split

def test_split_returns_boundary_not_closed_frame():
    dates = pd.date_range('2025-01-01', periods=120, freq='D', tz='UTC')
    frame = pd.DataFrame({'symbol':['AAA']*120,'interval_code':['1d']*120,'bar_timestamp':dates,'adj_z':[0.0]*120})
    eligible, summary = eligible_universe(frame)
    development, boundary = development_holdout_split(eligible,.20)
    assert isinstance(boundary,pd.Timestamp)
    assert len(development)==96
    assert (development.bar_timestamp<=boundary).all()

def test_notebook46_split_unpacked_correctly():
    root=Path(__file__).resolve().parents[1]
    nb=json.loads((root/'notebooks/46_provenance_and_leakage_validation.ipynb').read_text())
    code='\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code')
    assert 'eligible,eligibility_summary=eligible_universe(master)' in code
    assert 'development,holdout_boundary=development_holdout_split(eligible,.20)' in code
    assert 'len(closed)' not in code
