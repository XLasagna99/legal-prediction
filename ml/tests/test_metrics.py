"""Tests for ml/evaluation/metrics.py.

evaluate() was rewritten on 2026-08-22 to support multi-class labels
(macro-averaged precision/recall/f1, multi-class roc_auc with a defensive
fallback) alongside the original binary case. These tests cover both paths
plus the rare-class/missing-class edge case that motivated the try/except
around roc_auc_score.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score, precision_score, recall_score

from evaluation.metrics import evaluate, group_metrics, temporal_split


# --- temporal_split ---


def test_temporal_split_orders_chronologically():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2024-06-01", "2020-01-01", "2023-01-01", "2022-01-01", "2021-01-01"]
            ),
            "value": [5, 1, 4, 3, 2],
        }
    )
    train, test = temporal_split(df, "date", test_fraction=0.2)
    assert train["date"].max() < test["date"].min()
    assert list(train["value"]) == [1, 2, 3, 4]
    assert list(test["value"]) == [5]


def test_temporal_split_respects_test_fraction():
    df = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=100), "value": range(100)})
    train, test = temporal_split(df, "date", test_fraction=0.2)
    assert len(train) == 80
    assert len(test) == 20


def test_temporal_split_does_not_mutate_input():
    df = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=10), "value": range(10)})
    original_order = list(df["value"])
    temporal_split(df, "date")
    assert list(df["value"]) == original_order


# --- evaluate: binary case ---


def test_evaluate_binary_perfect_predictions():
    y_true = [0, 1, 0, 1, 1]
    y_pred = [0, 1, 0, 1, 1]
    y_score = [0.1, 0.9, 0.2, 0.8, 0.95]
    report = evaluate(y_true, y_pred, y_score)
    assert report.accuracy == 1.0
    assert report.roc_auc == 1.0
    assert report.n == 5


def test_evaluate_binary_no_score_leaves_roc_auc_none():
    report = evaluate([0, 1, 0], [0, 1, 1])
    assert report.roc_auc is None


def test_evaluate_single_class_in_y_true_leaves_roc_auc_none():
    # roc_auc_score is undefined with only one class present.
    report = evaluate([1, 1, 1], [1, 1, 0], [0.9, 0.8, 0.6])
    assert report.roc_auc is None


# --- evaluate: multi-class case ---


def test_evaluate_multiclass_macro_metrics_match_sklearn():
    y_true = ["a", "a", "b", "b", "c", "c"]
    y_pred = ["a", "b", "b", "b", "c", "a"]
    report = evaluate(y_true, y_pred)
    assert report.precision == pytest.approx(
        precision_score(y_true, y_pred, average="macro", zero_division=0)
    )
    assert report.recall == pytest.approx(recall_score(y_true, y_pred, average="macro", zero_division=0))
    assert report.f1 == pytest.approx(f1_score(y_true, y_pred, average="macro", zero_division=0))


def test_evaluate_multiclass_roc_auc_with_probability_matrix():
    classes = np.array(["a", "b", "c"])
    y_true = ["a", "b", "c", "a", "b", "c"]
    y_pred = ["a", "b", "c", "a", "b", "c"]
    # confident, correct probability matrix aligned to `classes` order
    y_score = np.array(
        [
            [0.9, 0.05, 0.05],
            [0.05, 0.9, 0.05],
            [0.05, 0.05, 0.9],
            [0.9, 0.05, 0.05],
            [0.05, 0.9, 0.05],
            [0.05, 0.05, 0.9],
        ]
    )
    report = evaluate(y_true, y_pred, y_score, classes=classes)
    assert report.roc_auc is not None
    assert report.roc_auc == pytest.approx(1.0)


def test_evaluate_multiclass_missing_class_in_test_set_does_not_raise():
    # Regression: a class present in `classes` (i.e. seen during training)
    # but absent from y_true (small/imbalanced test set) used to be able to
    # raise inside roc_auc_score. evaluate() must swallow that and return
    # roc_auc=None instead of crashing the whole training run.
    classes = np.array(["a", "b", "c", "rare"])
    y_true = ["a", "b", "c", "a", "b", "c"]  # "rare" never appears
    y_pred = ["a", "b", "c", "a", "b", "a"]
    y_score = np.array(
        [
            [0.7, 0.1, 0.1, 0.1],
            [0.1, 0.7, 0.1, 0.1],
            [0.1, 0.1, 0.7, 0.1],
            [0.7, 0.1, 0.1, 0.1],
            [0.1, 0.7, 0.1, 0.1],
            [0.4, 0.3, 0.2, 0.1],
        ]
    )
    report = evaluate(y_true, y_pred, y_score, classes=classes)
    assert report.roc_auc is None
    assert report.accuracy is not None  # other metrics still computed


# --- group_metrics ---


def test_group_metrics_one_row_per_group():
    df = pd.DataFrame(
        {
            "court": ["SGHC", "SGHC", "SGCA", "SGCA", "SGCA"],
            "y_true": [0, 1, 0, 1, 1],
            "y_pred": [0, 1, 0, 1, 0],
        }
    )
    result = group_metrics(df, "court", "y_true", "y_pred")
    assert set(result["court"]) == {"SGHC", "SGCA"}
    assert result[result["court"] == "SGCA"]["n"].iloc[0] == 3
    assert result[result["court"] == "SGHC"]["n"].iloc[0] == 2
