"""Read-only database-wide coverage and selected-symbol audit."""
import pandas as pd
import streamlit as st
from stock_ml_dynamic.db import coverage, engine, price_history
from stock_ml_dynamic.health import audit_history

st.set_page_config(page_title="Data Health | Stock ML Dynamic", layout="wide")
st.title("Data Health")
st.caption("Coverage does not imply candle completeness. Gap lengths include market closures.")
@st.cache_data(ttl=180)
def get_coverage():
    with engine().connect() as conn:
        return coverage(conn)
@st.cache_data(ttl=180)
def get_history(symbol, interval, limit):
    with engine().connect() as conn:
        return price_history(conn, symbol, interval, limit)
try:
    data = get_coverage()
except Exception as exc:
    st.error(f"Database connection failed: {exc}")
    st.stop()
if data.empty:
    st.warning("No stored prices")
    st.stop()
interval_order = ["1m", "5m", "15m", "1h", "1d", "1wk", "1mo"]
summary = data.groupby("interval", as_index=False).agg(symbols=("symbol", "nunique"), bars=("rows", "sum"), earliest=("earliest", "min"), latest=("latest", "max"))
summary["order"] = summary.interval.map({v:i for i,v in enumerate(interval_order)})
summary = summary.sort_values("order").drop(columns="order")
a,b,c = st.columns(3)
a.metric("Distinct symbols", data.symbol.nunique())
b.metric("Stored bars", f"{int(data.rows.sum()):,}")
c.metric("Populated combinations", f"{len(data):,}")
st.dataframe(summary, use_container_width=True, hide_index=True)
known = set(zip(data.symbol, data.interval))
missing = pd.DataFrame([{"symbol": s, "interval": i} for s in sorted(data.symbol.unique()) for i in interval_order if (s,i) not in known], columns=["symbol","interval"])
st.write(f"Missing intervals among stored symbols: **{len(missing):,}**")
if not missing.empty:
    st.download_button("Download missing combinations", missing.to_csv(index=False), "missing_intervals.csv", "text/csv")
st.divider()
st.subheader("Inspect one ticker / interval")
col1,col2,col3 = st.columns(3)
symbol = col1.selectbox("Ticker", sorted(data.symbol.unique()), index=sorted(data.symbol.unique()).index("AAPL") if "AAPL" in data.symbol.unique() else 0)
intervals = sorted(data.loc[data.symbol==symbol, "interval"].unique(), key=lambda x: interval_order.index(x) if x in interval_order else 99)
interval = col2.selectbox("Interval", intervals, index=intervals.index("1d") if "1d" in intervals else 0)
limit = col3.selectbox("Most recent bars to audit", [500, 2000, 10000], index=1)
try:
    history = get_history(symbol, interval, limit)
    report = audit_history(history)
    st.json(report)
    st.caption("Largest gap is a raw timestamp difference, not proof of missing market-session bars. Audit is limited to selected recent bars.")
except Exception as exc:
    st.error(f"Audit failed: {exc}")
