"""Leakage-aware, single-symbol chronological classification baselines."""
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .features import FEATURE_COLUMNS, build_features

@dataclass(frozen=True)
class TrainingConfig:
    horizon: int = 5
    test_fraction: float = 0.2
    min_labeled_rows: int = 200
    random_state: int = 42


def prepare_split(prices: pd.DataFrame, config: TrainingConfig):
    if not 0 < config.test_fraction < 0.5:
        raise ValueError("test_fraction must be between 0 and 0.5")
    if config.horizon < 1:
        raise ValueError("horizon must be positive")
    frame = build_features(prices, horizon=config.horizon)
    # Split by chronological bar position BEFORE filtering NaNs, then purge
    # train rows whose forward label touches the test interval.
    n = len(frame)
    split = int(n * (1 - config.test_fraction))
    train_end = split - config.horizon
    if train_end <= 0:
        raise ValueError("Not enough bars for purged chronological split")
    valid_features = frame[FEATURE_COLUMNS].notna().all(axis=1)
    valid_labels = frame.target_up.notna()
    train = frame.iloc[:train_end].loc[valid_features.iloc[:train_end] & valid_labels.iloc[:train_end]].copy()
    test = frame.iloc[split:].loc[valid_features.iloc[split:] & valid_labels.iloc[split:]].copy()
    if len(train) + len(test) < config.min_labeled_rows or len(train) < 50 or len(test) < 20:
        raise ValueError(f"Insufficient labeled feature rows: train={len(train)}, test={len(test)}")
    if train.target_up.nunique() < 2:
        raise ValueError("Training set contains only one class")
    return train, test, {"split_bar_index": split, "purged_bars": config.horizon, "raw_bars": n}


def train_classifier(prices: pd.DataFrame, config: TrainingConfig = TrainingConfig()):
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss, log_loss, roc_auc_score
    train, test, split_info = prepare_split(prices, config)
    x_train, x_test = train[FEATURE_COLUMNS], test[FEATURE_COLUMNS]
    y_train, y_test = train.target_up.astype(int), test.target_up.astype(int)
    models = {
        "majority_baseline": DummyClassifier(strategy="most_frequent"),
        "hist_gradient_boosting": HistGradientBoostingClassifier(max_iter=150, max_leaf_nodes=15, learning_rate=0.05, l2_regularization=1.0, random_state=config.random_state),
    }
    evaluations = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        predicted = model.predict(x_test)
        metrics = {"accuracy": float(accuracy_score(y_test, predicted)), "balanced_accuracy": float(balanced_accuracy_score(y_test, predicted))}
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(x_test)
            positive = prob[:, list(model.classes_).index(1)] if 1 in model.classes_ else np.zeros(len(y_test))
            metrics["brier_score"] = float(brier_score_loss(y_test, positive))
            if y_test.nunique() == 2:
                metrics["roc_auc"] = float(roc_auc_score(y_test, positive))
        evaluations[name] = metrics
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": asdict(config), "features": FEATURE_COLUMNS,
        "train_rows": len(train), "test_rows": len(test),
        "train_start": str(train.time.iloc[0]), "train_end": str(train.time.iloc[-1]),
        "test_start": str(test.time.iloc[0]), "test_end": str(test.time.iloc[-1]),
        "split": split_info, "metrics": evaluations,
        "caveat": "Single holdout evaluation; not a backtest or evidence of profitable trading.",
    }
    return models, metadata


def save_run(models, metadata, output_dir):
    import joblib
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=False)
    for name, model in models.items():
        joblib.dump(model, folder / f"{name}.joblib")
    (folder / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return folder
