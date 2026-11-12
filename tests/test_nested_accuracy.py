import pandas as pd
import pytest
from stock_ml_dynamic.nested_accuracy import choose_candidate, feature_sets, _predict, run_nested_accuracy


def test_feature_schemes():
    sets=feature_sets(['a'],['volatility_5','drawdown_20'])
    assert sets['base']==['a']
    assert sets['volatility']==['a','volatility_5']
    assert sets['all']==['a','volatility_5','drawdown_20']


def test_inner_selection_accuracy_first():
    s=pd.DataFrame([{'candidate':'a','accuracy':.40,'balanced_accuracy':.39},{'candidate':'b','accuracy':.41,'balanced_accuracy':.30}])
    assert choose_candidate(s)[0]=='b'


def test_selection_tiebreak_balanced():
    s=pd.DataFrame([{'candidate':'a','accuracy':.40,'balanced_accuracy':.39},{'candidate':'b','accuracy':.40,'balanced_accuracy':.42}])
    assert choose_candidate(s)[0]=='b'


def test_requires_three_outer_folds():
    with pytest.raises(ValueError,match='three outer'):
        run_nested_accuracy(pd.DataFrame(),n_outer=2)
