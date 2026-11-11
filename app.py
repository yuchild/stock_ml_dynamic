"""Stock ML Dynamic — read-only market coverage and price exploration."""
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from stock_ml_dynamic.db import coverage, engine, price_history

st.set_page_config(page_title="Stock ML Dynamic", page_icon="📈", layout="wide")
st.title("Stock ML Dynamic")
st.caption("PostgreSQL market data · Configurable ML experiments · Saved model comparisons")

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
except Exception as exc:
    st.error(f"Database unavailable: {exc}")
    st.stop()
if data.empty:
    st.warning("No stock-price data available")
    st.stop()

order = ["1m", "5m", "15m", "1h", "1d", "1wk", "1mo"]
symbols = sorted(data.symbol.dropna().unique().tolist())
with st.sidebar:
    st.header("Market explorer")
    symbol = st.selectbox("Ticker", symbols, index=symbols.index("AAPL") if "AAPL" in symbols else 0)
    intervals = sorted(data.loc[data.symbol == symbol, "interval"].unique(), key=lambda v: order.index(v) if v in order else 99)
    interval = st.selectbox("Interval", intervals, index=intervals.index("1d") if "1d" in intervals else 0)
    limit = st.select_slider("Bars to display", options=[100, 250, 500, 1000, 2000, 3000], value=500)
    show_sma = st.checkbox("Show 20/50-day (bar) moving averages", value=True)

selected = data.loc[(data.symbol == symbol) & (data.interval == interval)].iloc[0]
metric1, metric2, metric3, metric4 = st.columns(4)
metric1.metric("Symbols in database", f"{len(symbols):,}")
metric2.metric("Bars for selection", f"{int(selected['rows']):,}")
metric3.metric("Earliest stored", str(selected.earliest)[:10])
metric4.metric("Latest stored", str(selected.latest)[:19])

try:
    history = load_history(symbol, interval, limit)
except Exception as exc:
    st.error(f"Unable to load price history: {exc}")
    st.stop()

if not history.empty:
    has_volume = "volume" in history.columns
    fig = make_subplots(rows=2 if has_volume else 1, cols=1, shared_xaxes=True,
                        row_heights=[0.78, 0.22] if has_volume else [1], vertical_spacing=0.035)
    fig.add_trace(go.Candlestick(x=history.time, open=history.open, high=history.high,
                                 low=history.low, close=history.close, name="OHLC"), row=1, col=1)
    if show_sma:
        for window in [20, 50]:
            if len(history) >= window:
                fig.add_trace(go.Scatter(x=history.time, y=history.close.rolling(window).mean(),
                                         name=f"SMA {window}", mode="lines"), row=1, col=1)
    if has_volume:
        fig.add_trace(go.Bar(x=history.time, y=history.volume, name="Volume", marker_color="#64748b"), row=2, col=1)
    fig.update_layout(title=f"{symbol} · {interval}", height=650, xaxis_rangeslider_visible=False,
                      margin=dict(l=15, r=15, t=55, b=20), legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("Recent candles"):
        st.dataframe(history.tail(30).iloc[::-1], use_container_width=True, hide_index=True)

with st.expander("Database coverage and missing intervals"):
    summary = data.groupby("interval", as_index=False).agg(symbols=("symbol", "nunique"), bars=("rows", "sum"),
                                                           earliest=("earliest", "min"), latest=("latest", "max"))
    st.dataframe(summary, use_container_width=True, hide_index=True)
    st.caption("Missing means no stored bars for that ticker/interval; provider history limits may apply.")
    known = set(zip(data.symbol, data.interval))
    missing = pd.DataFrame([{"symbol": sym, "interval": iv} for sym in symbols for iv in order if (sym, iv) not in known])
    st.write(f"Missing known-symbol/interval pairs: {len(missing):,}")
    if not missing.empty:
        st.download_button("Download missing pairs CSV", missing.to_csv(index=False), "missing_stock_intervals.csv", "text/csv")
        st.dataframe(missing.head(300), use_container_width=True, hide_index=True)

st.info("Read-only market data. Train and revise models in notebooks; view saved evaluations in Experiment Results. Model rankings are exploratory, not validated trading signals.")

st.subheader("Saved model results")
from stock_ml_dynamic.experiment_training import list_experiments, comparison_rows
saved = list_experiments()
rows = comparison_rows(saved, symbol=symbol, interval=interval)
if rows:
    results = pd.DataFrame(rows)
    configurations = sorted(results.spec_hash.unique())
    chosen_config = st.selectbox("Experiment configuration", configurations)
    matching = results.loc[results.spec_hash == chosen_config].copy()
    if "roc_auc" in matching:
        matching = matching.sort_values("roc_auc", ascending=False, na_position="last")
        valid = matching.loc[matching.roc_auc.notna()]
        if not valid.empty:
            best = valid.iloc[0]
            st.metric("Highest observed holdout ROC-AUC", f"{best.roc_auc:.3f}")
            st.caption(f"Candidate: {best.model} · run {best.run_id}. This is exploratory, not a validated best model.")
    elif "mae" in matching:
        matching = matching.sort_values("mae")
        st.caption("Regression experiments: lower MAE is better. ROC-AUC does not apply.")
    st.dataframe(matching, use_container_width=True, hide_index=True)
    st.caption("Only compare identical target definitions and test periods. A final untouched evaluation is required before deployment.")
else:
    st.caption("No saved model experiments for this ticker and interval yet. Run a training notebook to create one.")
