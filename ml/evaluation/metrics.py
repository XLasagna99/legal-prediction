"""Evaluation utilities.

The two functions that matter most here are `temporal_split` (so you never
leak the future into training) and `group_metrics` (so you catch bias that
aggregate accuracy hides).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class ClassificationReport:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float | None
    n: int
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "roc_auc": self.roc_auc,
            "n": self.n,
            **self.extra,
        }


def temporal_split(
    df: pd.DataFrame,
    date_column: str,
    test_fraction: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split chronologically: oldest rows train, newest rows test.

    This is the honest way to evaluate a "predict before the case happens"
    model. A random split lets information from the future leak into training
    and will inflate every metric you report.
    """
    ordered = df.sort_values(date_column).reset_index(drop=True)
    cutoff = int(len(ordered) * (1 - test_fraction))
    return ordered.iloc[:cutoff].copy(), ordered.iloc[cutoff:].copy()


def evaluate(y_true, y_pred, y_score=None) -> ClassificationReport:
    """Standard binary-classification metrics."""
    roc = None
    if y_score is not None and len(np.unique(y_true)) > 1:
        roc = float(roc_auc_score(y_true, y_score))
    return ClassificationReport(
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        roc_auc=roc,
        n=int(len(y_true)),
    )


def group_metrics(
    df: pd.DataFrame,
    group_column: str,
    y_true_column: str,
    y_pred_column: str,
) -> pd.DataFrame:
    """Per-group performance — your first line of defense against bias.

    Pass a column like protected attribute, court, or representation status.
    Large gaps in error rates across groups are a red flag worth investigating
    before anyone relies on the model.
    """
    rows = []
    for value, sub in df.groupby(group_column):
        rep = evaluate(sub[y_true_column], sub[y_pred_column])
        rows.append({group_column: value, **rep.to_dict()})
    return pd.DataFrame(rows).sort_values("n", ascending=False)
