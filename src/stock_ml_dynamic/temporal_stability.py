"""Notebook 20: fold-paired temporal robustness diagnostics, development data only.

Fold-level uncertainty is descriptive: expanding-window folds overlap in training
and must not be treated as independent replicates. No holdout/ledger access.
"""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd

REQUIRED = {"fold", "model", "roc_auc", "accuracy", "balanced_accuracy", "log_loss", "ece"}
METRICS = ("roc_auc", "accuracy", "balanced_accuracy", "log_loss", "ece")
LOWER_BETTER = {"log_loss", "ece"}


def validate_scores(scores: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED - set(scores.columns)
    if missing:
        raise ValueError(f"Missing fold metric columns: {sorted(missing)}")
    data = scores.copy()
    if data.empty:
        raise ValueError("No fold metrics available")
    if data[["fold", "model"]].isna().any().any():
        raise ValueError("Missing fold or model")
    if data.duplicated(["fold", "model"]).any():
        raise ValueError("Duplicate (fold, model) metric rows")
    for metric in METRICS:
        data[metric] = pd.to_numeric(data[metric], errors="coerce")
        if not np.isfinite(data[metric].to_numpy()).all():
            raise ValueError(f"Nonfinite metric: {metric}")
    if not data["roc_auc"].between(0, 1).all() or not data["accuracy"].between(0, 1).all():
        raise ValueError("Invalid classification metric range")
    return data


def paired_fold_deltas(scores: pd.DataFrame, candidate: str, reference: str = "random_forest") -> pd.DataFrame:
    """Positive improvement is better for every metric; pairs require matching folds."""
    data = validate_scores(scores)
    a = data.loc[data.model == candidate].set_index("fold")
    b = data.loc[data.model == reference].set_index("fold")
    if a.empty or b.empty:
        raise ValueError("Both candidate and reference models must exist")
    if set(a.index) != set(b.index):
        raise ValueError("Candidate/reference fold sets differ; cannot compare paired scores")
    out = pd.DataFrame({"fold": sorted(a.index)})
    for metric in METRICS:
        delta = a.loc[out.fold, metric].to_numpy() - b.loc[out.fold, metric].to_numpy()
        out[f"improvement_{metric}"] = -delta if metric in LOWER_BETTER else delta
    return out


def summarize_pairs(scores: pd.DataFrame, reference: str = "random_forest") -> pd.DataFrame:
    data = validate_scores(scores)
    rows = []
    for candidate in sorted(set(data.model) - {reference}):
        paired = paired_fold_deltas(data, candidate, reference)
        n = len(paired)
        row = {"candidate": candidate, "reference": reference, "n_folds": n}
        for metric in METRICS:
            values = paired[f"improvement_{metric}"].to_numpy()
            row[f"mean_improvement_{metric}"] = float(values.mean())
            row[f"min_improvement_{metric}"] = float(values.min())
            row[f"positive_folds_{metric}"] = int((values > 0).sum())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("mean_improvement_roc_auc", ascending=False).reset_index(drop=True)


def fold_winners(scores: pd.DataFrame, metric: str = "roc_auc") -> pd.DataFrame:
    data = validate_scores(scores)
    if metric not in METRICS:
        raise ValueError(f"Unsupported metric: {metric}")
    idx = data.groupby("fold")[metric].idxmin() if metric in LOWER_BETTER else data.groupby("fold")[metric].idxmax()
    return data.loc[idx, ["fold", "model", metric]].sort_values("fold").reset_index(drop=True)


def write_diagnostics(scores: pd.DataFrame, output_dir: str | Path, reference: str = "random_forest"):
    data = validate_scores(scores)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paired = summarize_pairs(data, reference)
    winners = fold_winners(data)
    paired.to_csv(out / "paired_improvements.csv", index=False)
    winners.to_csv(out / "fold_winners.csv", index=False)
    data.to_csv(out / "fold_metrics_copy.csv", index=False)
    manifest = {"purpose": "DEVELOPMENT_ONLY_FOLD_DIAGNOSTICS", "reference": reference,
                "n_folds": int(data.fold.nunique()), "models": sorted(data.model.unique().tolist()),
                "folds_are_independent": False,
                "warning": "Fold comparisons are descriptive; overlapping expanding training windows invalidate naive independent-fold confidence intervals.",
                "final_holdout": "NOT_ACCESSED", "prospective_ledger": "NOT_ACCESSED"}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return paired, winners, manifest
