from stock_ml_dynamic.db import PRICE_COLUMNS, qualified_table


def test_postgres_price_columns():
    assert PRICE_COLUMNS["close"] == "close_price"
    assert PRICE_COLUMNS["open"] == "open_price"
    assert PRICE_COLUMNS["volume"] == "volume"


def test_default_table(monkeypatch):
    monkeypatch.delenv("STOCK_TABLE", raising=False)
    assert qualified_table() == ("raw", "stock_prices")
