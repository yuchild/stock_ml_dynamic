import pytest
from stock_ml_dynamic.db import qualified_table, column_env

def test_qualified_table(monkeypatch):
    monkeypatch.setenv("STOCK_TABLE", "raw.stock_prices")
    assert qualified_table() == ("raw", "stock_prices")

def test_invalid_table(monkeypatch):
    monkeypatch.setenv("STOCK_TABLE", "raw.stock_prices;DROP")
    with pytest.raises(ValueError):
        qualified_table()

def test_invalid_column(monkeypatch):
    monkeypatch.setenv("STOCK_SYMBOL_COLUMN", "symbol;DROP")
    with pytest.raises(ValueError):
        column_env("STOCK_SYMBOL_COLUMN", "symbol")
