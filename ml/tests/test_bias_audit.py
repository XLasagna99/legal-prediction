"""Tests for ml/evaluation/bias_audit.py.

ROADMAP.md / docs/MODEL_IMPROVEMENTS.md Phase 4.3 flag the bias audit as
unimplemented -- this covers `run_audit()`'s wiring (registry lookup,
temporal split reuse, grouped prediction) with a tiny fitted pipeline
rather than depending on the real dataset/artifact being present.
"""
import json

import pandas as pd
import joblib
import pytest
from sklearn.dummy import DummyClassifier

from evaluation import bias_audit


@pytest.fixture
def fake_registry(tmp_path, monkeypatch):
    df = pd.DataFrame(
        {
            "text": [f"case {i}" for i in range(10)],
            "outcome": ["dismissed", "allowed_full"] * 5,
            "filed_date": pd.date_range("2020-01-01", periods=10),
            "court": ["SGHC"] * 6 + ["SGCA"] * 4,
        }
    )
    data_path = tmp_path / "cases.csv"
    df.to_csv(data_path, index=False)

    pipeline = DummyClassifier(strategy="most_frequent")
    pipeline.fit(df[["text"]], df["outcome"])
    model_path = tmp_path / "baseline_test.joblib"
    joblib.dump(pipeline, model_path)

    registry = tmp_path
    (registry / "latest.json").write_text(json.dumps({"model_path": model_path.name}))

    monkeypatch.setattr(bias_audit, "DATA_PATH", data_path)
    monkeypatch.setattr(bias_audit, "REGISTRY", registry)
    return df


def test_run_audit_groups_by_court(fake_registry):
    result = bias_audit.run_audit("court")
    assert set(result["court"]) <= {"SGHC", "SGCA"}
    assert result["n"].sum() == 2  # 20% test fraction of 10 rows


def test_run_audit_unknown_group_column_filled(fake_registry, tmp_path):
    df = pd.read_csv(bias_audit.DATA_PATH)
    df["court"] = None
    df.to_csv(bias_audit.DATA_PATH, index=False)
    result = bias_audit.run_audit("court")
    assert list(result["court"]) == ["unknown"]
