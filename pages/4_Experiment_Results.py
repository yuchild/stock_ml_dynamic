"""Saved experiments are compared without executing model artifacts."""
import pandas as pd
import streamlit as st
from stock_ml_dynamic.experiment_training import list_experiments, comparison_rows

st.set_page_config(page_title='Experiment Results | Stock ML Dynamic', layout='wide')
st.title('Experiment Results')
st.caption('Compare saved experiments. ROC-AUC ranking applies only to binary classification; a single holdout is not sufficient for production model selection.')
records = list_experiments()
if not records:
    st.info('No saved experiments yet. Train one on the Model Training page.')
    st.stop()
symbols = sorted({r['symbol'] for r in records})
symbol = st.selectbox('Ticker', symbols)
intervals = sorted({r['interval'] for r in records if r['symbol'] == symbol})
interval = st.selectbox('Interval', intervals)
rows = comparison_rows(records, symbol=symbol, interval=interval)
if not rows:
    st.stop()
frame = pd.DataFrame(rows)
st.write(f'Saved model evaluations: {len(frame)}')
st.caption('Different target definitions or test periods are not directly comparable. Filter to a matching configuration before ranking.')
configs = sorted(frame.spec_hash.unique())
config = st.selectbox('Compare within configuration ID', configs)
filtered = frame.loc[frame.spec_hash == config].copy()
if 'roc_auc' in filtered:
    filtered = filtered.sort_values('roc_auc', ascending=False, na_position='last')
    valid = filtered.loc[filtered.roc_auc.notna()]
    if not valid.empty:
        winner = valid.iloc[0]
        st.metric('Highest observed holdout ROC-AUC', f'{winner.roc_auc:.3f}')
        st.write(f'Candidate: **{winner.model}** · run `{winner.run_id}`')
        st.warning('This is the highest *observed* score, not a verified best model. Repeated selection on the same holdout causes selection bias; use walk-forward validation and a final untouched test set.')
else:
    filtered = filtered.sort_values('mae')
    st.caption('Regression experiments ranked by lowest MAE; ROC-AUC does not apply.')
st.dataframe(filtered, use_container_width=True, hide_index=True)
with st.expander('Saved configuration and evaluation metadata'):
    run_ids = filtered.run_id.unique().tolist()
    run = st.selectbox('Run', run_ids)
    st.json(next(r for r in records if r['run_id'] == run))
