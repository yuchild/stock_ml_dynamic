"""Configurable, reproducible ML experiments."""
import pandas as pd
import streamlit as st
from stock_ml_dynamic.db import coverage, engine, price_history
from stock_ml_dynamic.experiment import AVAILABLE_FEATURES, ExperimentSpec
from stock_ml_dynamic.experiment_training import train_experiment, save_experiment, list_experiments, comparison_rows

st.set_page_config(page_title='Model Training | Stock ML Dynamic', layout='wide')
st.title('Model Training — Experiment Workbench')
st.caption('Edit feature selection and target definitions before every training run. Saved runs remain immutable.')

@st.cache_data(ttl=180)
def get_coverage():
    with engine().connect() as conn:
        return coverage(conn)

@st.cache_data(ttl=180)
def get_prices(symbol, interval, bars):
    with engine().connect() as conn:
        return price_history(conn, symbol, interval, bars)

try:
    data = get_coverage()
    symbols = sorted(data.symbol.unique())
    if not symbols:
        st.warning('No stocks found')
        st.stop()
    a, b, c, d = st.columns(4)
    symbol = a.selectbox('Ticker', symbols, index=symbols.index('AAPL') if 'AAPL' in symbols else 0)
    intervals = sorted(data.loc[data.symbol == symbol, 'interval'].unique())
    interval = b.selectbox('Interval', intervals, index=intervals.index('1d') if '1d' in intervals else 0)
    bars = c.selectbox('Most recent bars', [500, 1000, 2000, 3000], index=2)
    horizon = d.number_input('Target horizon (bars)', 1, 60, 5)
    st.subheader('1 · Features')
    selected = st.multiselect('Model input features', list(AVAILABLE_FEATURES), default=list(AVAILABLE_FEATURES))
    with st.expander('Indicator parameters', expanded=False):
        p1, p2, p3 = st.columns(3)
        sma_short = p1.number_input('Short SMA window', 2, 500, 10)
        sma_medium = p2.number_input('Medium SMA window', 2, 500, 20)
        sma_long = p3.number_input('Long SMA window', 2, 500, 50)
        p4, p5, p6 = st.columns(3)
        volatility_window = p4.number_input('Volatility window', 2, 500, 20)
        volume_window = p5.number_input('Volume window', 2, 500, 20)
        rsi_window = p6.number_input('RSI window', 2, 500, 14)
        st.caption('Feature column names retain their legacy labels even when windows change; the exact windows are saved in the experiment configuration.')
    st.subheader('2 · Target definition')
    target_type = st.radio('Prediction target', ['direction', 'threshold', 'return'], horizontal=True,
                           format_func=lambda x: {'direction': 'Price direction', 'threshold': 'Return above threshold', 'return': 'Forward return (regression)'}[x])
    threshold = st.number_input('Positive-return threshold (%)', -99.0, 999.0, 1.0, 0.25, disabled=target_type != 'threshold')
    test_fraction = st.slider('Chronological test fraction', .1, .4, .2, .05)
    spec = ExperimentSpec(features=tuple(selected), horizon=int(horizon), target_type=target_type,
                          threshold=threshold/100 if target_type == 'threshold' else 0., test_fraction=float(test_fraction),
                          sma_short=int(sma_short), sma_medium=int(sma_medium), sma_long=int(sma_long),
                          volatility_window=int(volatility_window), volume_window=int(volume_window), rsi_window=int(rsi_window))
    st.caption(f'Configuration ID: `{spec.fingerprint()}` · Feature count: {len(selected)}')
    st.download_button('Download experiment configuration', pd.Series(spec.as_dict()).to_json(indent=2),
                       file_name=f'experiment_{spec.fingerprint()}.json', mime='application/json')
    if st.button('Train and save experiment', type='primary'):
        with st.spinner('Training models...'):
            models, meta = train_experiment(get_prices(symbol, interval, bars), spec, symbol, interval)
            folder = save_experiment(models, meta)
        st.success(f'Saved run {meta["run_id"]} to {folder}')
        st.dataframe(pd.DataFrame(meta['metrics']).T, use_container_width=True)
        st.warning(meta['warning'])
except Exception as exc:
    st.error(f'Training page failed: {exc}')
