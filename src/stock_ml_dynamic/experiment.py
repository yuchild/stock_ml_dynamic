"""Versioned experiment specifications; no database writes."""
from dataclasses import dataclass, asdict
import hashlib
import json

AVAILABLE_FEATURES = (
    'return_1', 'return_5', 'return_20', 'sma_10_ratio', 'sma_20_ratio',
    'sma_50_ratio', 'volatility_20', 'volume_ratio_20', 'rsi_14',
    'macd_ratio', 'range_ratio', 'body_ratio',
)

@dataclass(frozen=True)
class ExperimentSpec:
    features: tuple[str, ...] = AVAILABLE_FEATURES
    horizon: int = 5
    target_type: str = 'direction'  # direction, threshold, return
    threshold: float = 0.0  # fractional return, e.g. 0.01 for +1%
    test_fraction: float = 0.2
    min_labeled_rows: int = 200
    random_state: int = 42
    sma_short: int = 10
    sma_medium: int = 20
    sma_long: int = 50
    volatility_window: int = 20
    volume_window: int = 20
    rsi_window: int = 14

    def validate(self):
        if not self.features or len(set(self.features)) != len(self.features) or set(self.features) - set(AVAILABLE_FEATURES):
            raise ValueError('Select one or more distinct supported features')
        if self.target_type not in ('direction', 'threshold', 'return'):
            raise ValueError('Invalid target type')
        if self.horizon < 1 or not 0 < self.test_fraction < 0.5:
            raise ValueError('Invalid horizon or test fraction')
        if not -1 < self.threshold < 10:
            raise ValueError('Threshold must be a fractional return between -1 and 10')
        for key in ('sma_short','sma_medium','sma_long','volatility_window','volume_window','rsi_window'):
            if getattr(self, key) < 2 or getattr(self, key) > 500:
                raise ValueError(f'{key} must be between 2 and 500')
        return self

    def as_dict(self):
        result = asdict(self)
        result['features'] = list(self.features)
        return result

    def fingerprint(self):
        return hashlib.sha256(json.dumps(self.as_dict(), sort_keys=True).encode()).hexdigest()[:12]
