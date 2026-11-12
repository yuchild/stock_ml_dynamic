"""Leakage-aware OHLCV feature table and train-fitted target/candle transformers."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

NUMERIC_FEATURES = ['top_pct','body_pct','bottom_pct','gap','adj_z','slope','concavity']
CATEGORICAL_FEATURES = ['candle_type','month_of_year','day_of_month','day_of_week','hour_of_day']
CANDLE_FEATURES = ['top_pct','body_pct','bottom_pct','gap']
LABEL_MAPPING = {0:'no_change', 2:'down', 1:'up'}


def build_candle_table(prices: pd.DataFrame, z_window: int = 30, timestamp_timezone: str = 'America/New_York') -> pd.DataFrame:
    """Build features using only current and earlier bars. No training or fitted transformers.

    Expected columns: bar_timestamp, open_price, high_price, low_price,
    close_price, adj_close_price, volume. One symbol/interval per call.
    """
    if z_window < 2:
        raise ValueError('z_window must be >= 2')
    required = ['bar_timestamp','open_price','high_price','low_price','close_price','adj_close_price']
    missing = set(required)-set(prices.columns)
    if missing:
        raise ValueError(f'Missing columns: {sorted(missing)}')
    df = prices.copy()
    df['bar_timestamp'] = pd.to_datetime(df['bar_timestamp'], utc=True)
    df = df.sort_values('bar_timestamp').reset_index(drop=True)
    if df['bar_timestamp'].duplicated().any():
        raise ValueError('Duplicate bar timestamps; filter by symbol and interval first')
    for col in required[1:]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    o,c,h,l = (df[x] for x in ('open_price','close_price','high_price','low_price'))
    full = h-l
    valid = full > 0
    df['top_pct'] = ((h-pd.concat([o,c],axis=1).max(axis=1))/full).where(valid, 0.0)
    df['body_pct'] = ((o-c).abs()/full).where(valid, 0.0)
    df['bottom_pct'] = ((pd.concat([o,c],axis=1).min(axis=1)-l)/full).where(valid, 0.0)
    prev_range = (h-l).shift(1)
    df['gap'] = ((o-c.shift(1))/prev_range).where(prev_range.ne(0) & prev_range.notna(), 0.0)
    adj = df['adj_close_price']
    trailing = adj.shift(1).rolling(z_window, min_periods=z_window)
    mu, std = trailing.mean(), trailing.std(ddof=1)
    df['adj_z'] = ((adj-mu)/std).where(std.ne(0), 0.0)
    df.loc[mu.isna(), 'adj_z'] = np.nan
    # Quadratic interpolant through equally spaced (t-2,t-1,t):
    # first derivative at t = (3*y_t - 4*y_{t-1} + y_{t-2})/2
    # second derivative = y_t - 2*y_{t-1} + y_{t-2}
    df['slope'] = (3*adj - 4*adj.shift(1) + adj.shift(2))/2
    df['concavity'] = adj - 2*adj.shift(1) + adj.shift(2)
    ts = df['bar_timestamp'].dt.tz_convert(timestamp_timezone)
    df['month_of_year'] = ts.dt.month.astype('Int64')
    df['day_of_month'] = ts.dt.day.astype('Int64')
    df['day_of_week'] = ts.dt.dayofweek.astype('Int64')
    df['hour_of_day'] = ts.dt.hour.astype('Int64')
    df['adj_close'] = adj
    df['adj_close_pctc'] = adj.pct_change(fill_method=None)
    df['next_return'] = df['adj_close_pctc'].shift(-1)
    df['next_adj_close'] = adj.shift(-1)
    df['direction'] = pd.Series(pd.NA, index=df.index, dtype='Int64')
    df['candle_type'] = pd.Series(pd.NA, index=df.index, dtype='Int64')
    return df


@dataclass
class DirectionQuantiles:
    """Train-only quantile target. Legacy q=3 retained for saved-model compatibility.

    q=7 maps ascending septiles to [2,2,0,0,1,1,1].
    Duplicate edges fail explicitly rather than silently changing class semantics.
    """
    q: int = 3
    edges_: np.ndarray | None = None
    labels_: list | None = None

    def fit(self, training_next_returns: pd.Series):
        values = pd.to_numeric(training_next_returns, errors='coerce').replace([np.inf,-np.inf],np.nan).dropna()
        if values.empty:
            raise ValueError('No finite training returns')
        if self.q not in (3,7):
            raise ValueError('Supported quantile targets: q=3 or q=7')
        _, edges = pd.qcut(values, q=self.q, duplicates='drop', retbins=True)
        n = len(edges)-1
        if n != self.q:
            raise ValueError(f'q={self.q} produced only {n} unique bins; cannot preserve the requested target mapping')
        self.edges_ = np.asarray(edges, dtype=float)
        self.labels_ = [2,0,1] if self.q == 3 else [2,2,0,0,1,1,1]
        return self

    def transform(self, next_returns: pd.Series) -> pd.Series:
        if self.edges_ is None:
            raise ValueError('Fit quantiles first')
        edges = self.edges_.copy()
        edges[0], edges[-1] = -np.inf, np.inf
        # pd.cut labels must be unique for ordered categoricals. Use bin IDs then map.
        ids = pd.cut(pd.to_numeric(next_returns, errors='coerce'), bins=edges,
                     labels=False, include_lowest=True)
        mapping = dict(enumerate(self.labels_))
        return pd.Series(ids, index=next_returns.index).map(mapping).astype('Int64')


@dataclass
class CandleClusters:
    """Training-only elbow analysis; labels are arbitrary cluster IDs."""
    scaler_: StandardScaler | None = None
    model_: KMeans | None = None
    inertia_: pd.DataFrame | None = None
    chosen_k_: int | None = None

    def fit(self, training_features: pd.DataFrame, k_values=range(2, 9), selected_k=None):
        clean = training_features[CANDLE_FEATURES].replace([np.inf,-np.inf],np.nan).dropna()
        if len(clean) < 4:
            raise ValueError('At least four valid training candles are required')
        self.scaler_ = StandardScaler().fit(clean)
        x = self.scaler_.transform(clean)
        ks = sorted(set(int(k) for k in k_values if 2 <= int(k) < len(x)))
        if not ks:
            raise ValueError('No feasible k values')
        scores = []
        models = {}
        for k in ks:
            model = KMeans(n_clusters=k, random_state=42, n_init=10).fit(x)
            models[k] = model
            scores.append({'k':k, 'inertia':model.inertia_})
        self.inertia_ = pd.DataFrame(scores)
        # Maximum distance from line between first and last elbow points.
        # Heuristic only; inspect plot and override selected_k if appropriate.
        points = self.inertia_[['k','inertia']].to_numpy(dtype=float)
        if len(points) <= 2:
            auto_k = ks[0]
        else:
            normalized = (points-points.min(axis=0))/np.maximum(points.max(axis=0)-points.min(axis=0),1e-12)
            a,b = normalized[0],normalized[-1]
            delta = normalized-a
            distances = np.abs((b[0]-a[0])*delta[:,1] - (b[1]-a[1])*delta[:,0])/max(np.linalg.norm(b-a),1e-12)
            auto_k = ks[int(np.argmax(distances))]
        self.chosen_k_ = auto_k if selected_k is None else int(selected_k)
        if self.chosen_k_ not in models:
            raise ValueError(f'selected_k must be in {ks}')
        self.model_ = models[self.chosen_k_]
        return self

    def transform(self, features: pd.DataFrame) -> pd.Series:
        if self.model_ is None:
            raise ValueError('Fit candle clusters first')
        clean = features[CANDLE_FEATURES].replace([np.inf,-np.inf],np.nan)
        valid = clean.notna().all(axis=1)
        result = pd.Series(pd.NA, index=features.index, dtype='Int64')
        if valid.any():
            result.loc[valid] = self.model_.predict(self.scaler_.transform(clean.loc[valid])).astype(int)
        return result


def apply_training_transformers(frame: pd.DataFrame, training_end_position: int, k_values=range(2,9), selected_k=None):
    """Training boundary is exclusive. Returns full feature table and fitted transformers.

    Exclude the final training row from quantile fitting because its next-return label
    belongs to the next (validation) period. The last row always has unknown target.
    """
    if not 40 <= training_end_position < len(frame):
        raise ValueError('Training boundary must be >=40 and less than frame length')
    result = frame.copy()
    train = result.iloc[:training_end_position]
    quantiles = DirectionQuantiles().fit(train['next_return'].iloc[:-1])
    clusters = CandleClusters().fit(train, k_values=k_values, selected_k=selected_k)
    result['direction'] = quantiles.transform(result['next_return'])
    result['candle_type'] = clusters.transform(result)
    return result, quantiles, clusters
