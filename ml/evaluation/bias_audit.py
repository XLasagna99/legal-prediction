"""Bias audit: per-group performance for the latest trained baseline.

ROADMAP.md Phase 2 and docs/MODEL_IMPROVEMENTS.md Phase 4.3 both flag this
as an open item. Loads the model named in the registry's `latest.json`,
reproduces the exact temporal test split `train_baseline.py` used, and runs
`group_metrics()` by `court` -- the only metadata column worth grouping by
right now, since `case_type`/`represented` are constant "unknown" for every
row (docs/MODEL.md §7).

Usage: `python -m evaluation.bias_audit` from the `ml/` directory.
"""
from __future__ import annotations

import json

import joblib
import pandas as pd

from evaluation.metrics import group_metrics, temporal_split
from training.train_baseline import DATA_PATH, DATE_COLUMN, LABEL_COLUMN, REGISTRY


def run_audit(group_column: str = "court") -> pd.DataFrame:
    latest = json.loads((REGISTRY / "latest.json").read_text())
    pipeline = joblib.load(REGISTRY / latest["model_path"])

    df = pd.read_csv(DATA_PATH, parse_dates=[DATE_COLUMN])
    df = df.dropna(subset=["text", LABEL_COLUMN, DATE_COLUMN])
    df[group_column] = df[group_column].fillna("unknown").astype(str)

    _, test_df = temporal_split(df, DATE_COLUMN, test_fraction=0.2)
    test_df = test_df.copy()
    test_df["pred"] = pipeline.predict(test_df)

    return group_metrics(test_df, group_column, LABEL_COLUMN, "pred")


def main() -> None:
    report = run_audit("court")
    pd.set_option("display.width", 120)
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
