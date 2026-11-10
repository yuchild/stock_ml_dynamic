"""Causal, per-symbol feature engineering for chronological ML experiments."""
import numpy as np
import pandas as pd


def build_features(prices: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """Features use current/past bars; forward targets are for offline training only.

    Input is one ticker and one interval with ascending or unsorted time.
    Do not use target_* columns as model inputs. The most recent horizon rows
    have unknown labels and must be excluded from supervised training.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    required = {"time", "open", "high", "low", "close", "volume"}
    missing = required - set(prices.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if prices.empty:
        return prices.copy()
    df = prices.copy().sort_values("time").reset_index(drop=True)
    if df.time.duplicated().any():
        raise ValueError("Duplicate timestamps: resolve data quality issues first")
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
    close = df.close
    ret = close.pct_change(fill_method=None)
    df["return_1"] = ret
    df["return_5"] = close.pct_change(5, fill_method=None)
    df["return_20"] = close.pct_change(20, fill_method=None)
    for window in (10, 20, 50):
        avg = close.rolling(window, min_periods=window).mean()
        df[f"sma_{window}_ratio"] = close / avg - 1
    df["volatility_20"] = ret.rolling(20, min_periods=20).std()
    df["volume_ratio_20"] = df.volume / df.volume.rolling(20, min_periods=20).mean()
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))
    df.loc[(loss == 0) & (gain > 0), "rsi_14"] = 100
    df.loc[(loss == 0) & (gain == 0), "rsi_14"] = 50
    ema12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
    df["macd_ratio"] = (ema12 - ema26) / close
    df["range_ratio"] = (df.high - df.low) / close
    df["body_ratio"] = (df.close - df.open) / df.open
    future_close = close.shift(-horizon)
    df["target_forward_return"] = future_close / close - 1
    df["target_up"] = pd.Series(pd.NA, index=df.index, dtype="Int64")
    known = df["target_forward_return"].notna()
    df.loc[known, "target_up"] = (df.loc[known, "target_forward_return"] > 0).astype(int).to_numpy()
    return df.replace([np.inf, -np.inf], np.nan)


FEATURE_COLUMNS = ["return_1", "return_5", "return_20", "sma_10_ratio", "sma_20_ratio", "sma_50_ratio", "volatility_20", "volume_ratio_20", "rsi_14", "macd_ratio", "range_ratio", "body_ratio"]
