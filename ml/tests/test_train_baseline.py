"""Tests for ml/training/train_baseline.py.

`main()` used to slice `predict_proba(test_df)[:, 1]` and threshold at 0.5
-- correct only for binary labels. When `outcome` went from 2 classes to 5
(2026-08-22), this broke silently unless caught by a test that actually
exercises `main()` end-to-end with multi-class data. That's what
test_main_end_to_end_multiclass below does, via monkeypatched DATA_PATH/
REGISTRY so it never touches the real data/registry directories.
"""
import json

import pandas as pd
import pytest

import training.train_baseline as train_baseline
from training.train_baseline import build_pipeline, effective_number_class_weight


def _synthetic_cases_csv(n_per_class: int = 7) -> pd.DataFrame:
    """A small, deterministic multi-class dataset spread over time, with
    enough shared vocabulary per class to survive TfidfVectorizer's
    min_df=3, and both classes represented in the eventual train/test split.
    """
    classes = {
        "dismissed": "the application statutory demand bankruptcy order failed insufficient",
        "allowed_full": "the appeal creditors judicial management restructuring granted",
        "withdrawn": "the parties settlement deed consent withdrawn flat hdb",
    }
    rows = []
    day = 0
    for cls, vocab in classes.items():
        for i in range(n_per_class):
            rows.append(
                {
                    "text": f"{vocab} case number {i} additional filler words here",
                    "filed_date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=day),
                    "outcome": cls,
                    "court": "SGHC" if i % 2 == 0 else "SGCA",
                    "case_type": "unknown",
                    "represented": "unknown",
                }
            )
            day += 3  # interleave classes chronologically across all rows
    df = pd.DataFrame(rows).sort_values("filed_date").reset_index(drop=True)
    return df


# --- build_pipeline ---


def test_build_pipeline_has_expected_steps():
    pipe = build_pipeline(["court"])
    assert list(pipe.named_steps.keys()) == ["features", "clf"]


def test_build_pipeline_fit_predict_multiclass():
    df = _synthetic_cases_csv()
    pipe = build_pipeline(["court", "case_type", "represented"])
    pipe.fit(df, df["outcome"])

    proba = pipe.predict_proba(df)
    preds = pipe.predict(df)

    assert proba.shape == (len(df), 3)  # one column per class, not 2
    assert set(preds).issubset({"dismissed", "allowed_full", "withdrawn"})
    assert list(pipe.classes_) == sorted(["dismissed", "allowed_full", "withdrawn"])


def test_build_pipeline_fit_predict_binary_still_works():
    # Backward-compatibility check: the pipeline must still work correctly
    # if `outcome` only ever has 2 classes (e.g. a future non-Singapore source).
    df = _synthetic_cases_csv()
    df = df[df["outcome"] != "withdrawn"].reset_index(drop=True)
    pipe = build_pipeline(["court"])
    pipe.fit(df, df["outcome"])

    proba = pipe.predict_proba(df)
    preds = pipe.predict(df)

    assert proba.shape == (len(df), 2)
    assert set(preds).issubset({"dismissed", "allowed_full"})


# --- effective_number_class_weight (docs/MODEL_IMPROVEMENTS.md Phase 0.3) ---


def test_effective_number_class_weight_rarer_class_gets_larger_weight():
    y = ["a"] * 90 + ["b"] * 10
    weights = effective_number_class_weight(y)
    assert weights["b"] > weights["a"]


def test_effective_number_class_weight_equal_classes_get_equal_weight():
    y = ["a"] * 10 + ["b"] * 10
    weights = effective_number_class_weight(y)
    assert weights["a"] == pytest.approx(weights["b"])


def test_effective_number_class_weight_sample_weighted_average_is_one():
    # Normalized so sum(weight[c] * count[c]) == n_total (sklearn's
    # "balanced" convention) -- NOT a simple unweighted average of the
    # per-class weight values, which was a real bug (see the function's
    # docstring and docs/MODEL_RESULTS.md Run 4 vs Run 5): with unequal
    # class sizes those two normalizations differ substantially, and the
    # simple-average version silently shrank the whole loss scale.
    y = ["a"] * 90 + ["b"] * 10 + ["c"] * 5
    weights = effective_number_class_weight(y)
    counts = pd.Series(y).value_counts()
    sample_weighted_avg = sum(weights[c] * counts[c] for c in counts.index) / len(y)
    assert sample_weighted_avg == pytest.approx(1.0)


def test_effective_number_class_weight_close_to_balanced_at_small_n():
    # At beta=0.999 and n in the tens-to-hundreds range (this project's
    # scale), effective-number weighting barely differs from "balanced" --
    # the compression it's designed for only kicks in at much larger n.
    # It should still be *slightly* less extreme for the rare class, but
    # nowhere near as different as an earlier (buggy) version of this
    # function produced.
    y = ["a"] * 95 + ["b"] * 5
    weights = effective_number_class_weight(y)
    balanced_b_weight = len(y) / (2 * 5)  # sklearn's "balanced" formula
    assert weights["b"] < balanced_b_weight
    assert weights["b"] == pytest.approx(balanced_b_weight, rel=0.1)


def test_build_pipeline_accepts_custom_class_weight_dict():
    df = _synthetic_cases_csv()
    weights = effective_number_class_weight(df["outcome"])
    pipe = build_pipeline(["court"], class_weight=weights)
    pipe.fit(df, df["outcome"])
    assert pipe.named_steps["clf"].class_weight == weights
    # still produces valid predictions, not just accepts the param
    assert set(pipe.predict(df)).issubset(set(df["outcome"].unique()))


# --- main(): full pipeline, isolated from real data/registry via monkeypatch ---


def test_main_end_to_end_multiclass(tmp_path, monkeypatch):
    data_path = tmp_path / "cases.csv"
    registry = tmp_path / "registry"

    df = _synthetic_cases_csv()
    df.to_csv(data_path, index=False)

    monkeypatch.setattr(train_baseline, "DATA_PATH", data_path)
    monkeypatch.setattr(train_baseline, "REGISTRY", registry)

    train_baseline.main()

    latest = json.loads((registry / "latest.json").read_text())
    metrics = latest["metrics"]

    # These would previously have KeyError'd or silently computed nonsense
    # under the old binary-only [:, 1] slicing once `outcome` had >2 classes.
    assert metrics["n"] == latest["n_test"]
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0
    assert (registry / latest["model_path"]).exists()

    # Phase 0.3: effective-number class weights recorded on the model card.
    assert latest["class_weight_scheme"].startswith("effective_number")
    assert set(latest["class_weights"].keys()) == {"dismissed", "allowed_full", "withdrawn"}


def test_main_raises_clearly_when_no_data(tmp_path, monkeypatch):
    monkeypatch.setattr(train_baseline, "DATA_PATH", tmp_path / "missing.csv")
    monkeypatch.setattr(train_baseline, "REGISTRY", tmp_path / "registry")
    with pytest.raises(SystemExit):
        train_baseline.main()
