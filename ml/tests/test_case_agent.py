"""Tests for ml/agent/case_agent.py, the interactive outcome-prediction agent.

Trains a tiny synthetic model into a temp registry (same pattern as
test_train_baseline.py's test_main_end_to_end_multiclass) and monkeypatches
inference.predict's registry path so these tests never touch the real
trained model or its cache.
"""
import pandas as pd
import pytest

import inference.predict as predict_module
from agent.case_agent import DISPOSITION_MEANINGS, answer_question, build_report
from training import train_baseline


def _synthetic_cases_csv(n_per_class: int = 7) -> pd.DataFrame:
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
            day += 3
    return pd.DataFrame(rows).sort_values("filed_date").reset_index(drop=True)


@pytest.fixture
def report(tmp_path, monkeypatch):
    data_path = tmp_path / "cases.csv"
    registry = tmp_path / "registry"
    _synthetic_cases_csv().to_csv(data_path, index=False)

    monkeypatch.setattr(train_baseline, "DATA_PATH", data_path)
    monkeypatch.setattr(train_baseline, "REGISTRY", registry)
    train_baseline.main()

    monkeypatch.setattr(predict_module, "REGISTRY", registry)
    predict_module._load.cache_clear()

    case = {
        "text": "the appeal creditors judicial management restructuring granted new case",
        "court": "SGHC",
        "case_type": "unknown",
        "represented": "unknown",
    }
    return build_report(case)


def test_build_report_has_ranked_probabilities_summing_to_one(report):
    total = sum(p["probability"] for p in report["ranked_probabilities"])
    assert total == pytest.approx(1.0, abs=1e-6)
    assert report["ranked_probabilities"][0]["probability"] >= report["ranked_probabilities"][1]["probability"]


def test_build_report_label_is_a_known_disposition(report):
    assert report["label"] in DISPOSITION_MEANINGS


def test_build_report_english_summary_mentions_top_label_and_disclaimer(report):
    summary = report["english_summary"]
    assert report["label"].upper() in summary
    assert "not legal advice" in summary.lower()


def test_build_report_model_tier_defaults_to_local_ensemble(report):
    # `report`'s fixture doesn't set USE_LLM_ENSEMBLE, so this exercises the
    # real default (no LLM call, no API cost) rather than an opted-in tier.
    assert report["model_tier"] == "local ensemble"
    assert "[Model tier used: local ensemble]" in report["english_summary"]


def test_answer_question_outcome_intent(report):
    answer = answer_question(report, "what is the likely outcome?")
    assert report["label"].upper() in answer


def test_answer_question_confidence_intent(report):
    answer = answer_question(report, "how confident are you?")
    assert report["confidence_band"] in answer


def test_answer_question_accuracy_intent_uses_real_model_metrics(report):
    answer = answer_question(report, "how accurate is this model?")
    acc = report["model_metrics"]["accuracy"]
    assert f"{acc:.0%}" in answer


def test_answer_question_meaning_intent(report):
    answer = answer_question(report, "what does dismissed mean?")
    assert "DISMISSED" in answer
    assert DISPOSITION_MEANINGS["dismissed"] in answer


def test_answer_question_other_outcomes_phrasing_hits_alternatives_not_outcome(report):
    # Regression test: "what other outcomes are possible?" contains the
    # keyword "outcome" too, and used to be swallowed by the generic
    # "outcome" intent instead of "alternatives" -- found by manually
    # running the agent end-to-end (docs/MODEL_RESULTS.md-style spot check).
    answer = answer_question(report, "what other outcomes are possible?")
    assert "Other outcomes considered" in answer


def test_answer_question_unmatched_returns_help_text(report):
    answer = answer_question(report, "what's the weather like today?")
    assert "I can answer questions about" in answer


def test_answer_question_never_fabricates_beyond_model_output(report):
    # The costs intent must not invent a figure when none was extracted --
    # the synthetic case text has no costs language at all.
    answer = answer_question(report, "what costs might be awarded?")
    assert "no costs figure" in answer.lower() or "don't have" in answer.lower()
