"""Preview causal features and training labels without fitting a model."""
import streamlit as st
from stock_ml_dynamic.db import coverage, engine, price_history
from stock_ml_dynamic.features import FEATURE_COLUMNS, build_features

st.set_page_config(page_title="Feature Explorer | Stock ML Dynamic", layout="wide")
st.title("Feature Explorer")
st.caption("Read-only, single-symbol features. Forward targets are labels, not model inputs.")
@st.cache_data(ttl=180)
def load_coverage():
    with engine().connect() as conn:
        return coverage(conn)
@st.cache_data(ttl=180)
def load_history(symbol, interval, limit):
    with engine().connect() as conn:
        return price_history(conn, symbol, interval, limit)
try:
    data = load_coverage()
    symbols = sorted(data.symbol.unique())
    if not symbols:
        st.warning("No stored symbols")
        st.stop()
    col1,col2,col3,col4 = st.columns(4)
    symbol = col1.selectbox("Ticker", symbols, index=symbols.index("AAPL") if "AAPL" in symbols else 0)
    intervals = sorted(data.loc[data.symbol==symbol, "interval"].unique())
    interval = col2.selectbox("Interval", intervals, index=intervals.index("1d") if "1d" in intervals else 0)
    horizon = col3.number_input("Prediction horizon (bars)", min_value=1, max_value=50, value=5)
    limit = col4.selectbox("Recent bars", [250, 500, 1000, 2000, 3000], index=2)
    prices = load_history(symbol, interval, limit)
    if len(prices) < 60:
        st.warning("At least 60 bars recommended for indicators")
    features = build_features(prices, int(horizon))
    st.metric("Fully populated feature rows", int(features[FEATURE_COLUMNS].notna().all(axis=1).sum()))
    st.line_chart(features.set_index("time")[["close", "sma_20_ratio"]], y=None)
    st.dataframe(features.tail(30).iloc[::-1], hide_index=True, use_container_width=True)
    st.download_button("Download feature preview CSV", features.to_csv(index=False), f"{symbol}_{interval}_features.csv", "text/csv")
    st.info("Targets on the latest horizon bars are intentionally null. Training must use time-based splits and purge overlapping target horizons.")
except Exception as exc:
    st.error(f"Feature preview failed: {exc}")
