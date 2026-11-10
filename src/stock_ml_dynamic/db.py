"""Read-only PostgreSQL access for stock_ml_dynamic."""
import os
import re
from functools import lru_cache

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, inspect, text

load_dotenv()
IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
PRICE_COLUMNS = {"open": "open_price", "high": "high_price", "low": "low_price", "close": "close_price", "adj_close": "adj_close_price", "volume": "volume"}


def column_env(name, default):
    """Validated identifier setting retained for backward-compatible tests."""
    value = os.getenv(name, default)
    if not IDENTIFIER.fullmatch(value):
        raise ValueError(f"Invalid SQL identifier in {name}")
    return value


def qualified_table():
    parts = os.getenv("STOCK_TABLE", "raw.stock_prices").split(".")
    if len(parts) != 2 or not all(IDENTIFIER.fullmatch(p) for p in parts):
        raise ValueError("STOCK_TABLE must be schema.table with simple SQL identifiers")
    return tuple(parts)


@lru_cache(maxsize=1)
def engine():
    url = os.getenv("DATABASE_URL", "postgresql+psycopg:///stock_analytics_v0")
    if not url or "CHANGE_ME" in url:
        raise RuntimeError("Invalid DATABASE_URL")
    return create_engine(url, pool_pre_ping=True)


def table_columns(conn):
    schema, table = qualified_table()
    return {c["name"] for c in inspect(conn).get_columns(table, schema=schema)}


def verify_columns(conn):
    cols = table_columns(conn)
    needed = {"symbol", "interval_code", "bar_timestamp", "close_price"}
    missing = needed - cols
    if missing:
        raise ValueError(f"Missing stock-price columns: {sorted(missing)}")
    return cols


def coverage(conn):
    schema, table = qualified_table()
    verify_columns(conn)
    sql = text(f'''SELECT symbol, interval_code AS interval, COUNT(*) AS rows,
        MIN(bar_timestamp) AS earliest, MAX(bar_timestamp) AS latest
        FROM "{schema}"."{table}" GROUP BY symbol, interval_code
        ORDER BY symbol, interval_code''')
    return pd.read_sql(sql, conn)


def price_history(conn, symbol, interval, limit=1200):
    schema, table = qualified_table()
    cols = verify_columns(conn)
    projection = [f'"{source}" AS "{alias}"' for alias, source in PRICE_COLUMNS.items() if source in cols]
    sql = text(f'''SELECT bar_timestamp AS time, {", ".join(projection)}
        FROM "{schema}"."{table}"
        WHERE symbol=:symbol AND interval_code=:interval
        ORDER BY bar_timestamp DESC LIMIT :limit''')
    df = pd.read_sql(sql, conn, params={"symbol": symbol, "interval": interval, "limit": int(limit)})
    if df.empty:
        return df
    df = df.sort_values("time").reset_index(drop=True)
    for column in PRICE_COLUMNS:
        if column in df:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df
