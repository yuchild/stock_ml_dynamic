"""Per-ticker quality checks; gaps are descriptive, not calendar-aware."""
import pandas as pd


def audit_history(prices: pd.DataFrame) -> dict:
    if prices.empty:
        return {"bars": 0, "duplicate_timestamps": 0, "invalid_ohlc": 0, "negative_volume": 0, "largest_gap": None}
    df = prices.copy()
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    bad = (df[["open", "high", "low", "close"]].isna().any(axis=1)
           | (df[["open", "high", "low", "close"]] <= 0).any(axis=1)
           | (df.high < df[["open", "close", "low"]].max(axis=1))
           | (df.low > df[["open", "close", "high"]].min(axis=1)))
    gaps = pd.to_datetime(df.time, utc=True).sort_values().diff().dropna()
    return {"bars": len(df), "duplicate_timestamps": int(df.time.duplicated().sum()),
            "invalid_ohlc": int(bad.sum()), "negative_volume": int((df.volume < 0).sum()),
            "largest_gap": str(gaps.max()) if not gaps.empty else None}
