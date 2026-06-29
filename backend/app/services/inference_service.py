"""Serving-side inference.

Deliberately decoupled from the `ml` training package: the backend depends only
on the serialized artifact + joblib, not on training code. This mirrors the
logic in ml/inference/predict.py — keep them in sync, or refactor the shared
load/predict into a small installable package once the project grows.
"""
from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

import joblib
import pandas as pd

from app.core.config import settings


class ModelNotTrainedError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def _load() -> tuple[Any, dict]:
    latest = settings.model_dir / "latest.json"
    if not latest.exists():
        raise ModelNotTrainedError(
            f"No trained model in {settings.model_dir}. "
            "Train one with `python -m training.train_baseline` in ml/."
        )
    meta = json.loads(latest.read_text())
    model = joblib.load(settings.model_dir / meta["model_path"])
    return model, meta


def is_ready() -> bool:
    try:
        _load()
        return True
    except ModelNotTrainedError:
        return False


def get_model_info() -> dict:
    _, meta = _load()
    return {
        "version": meta["version"],
        "trained_at": meta["trained_at"],
        "metrics": meta["metrics"],
    }


def run_prediction(case: dict) -> dict:
    model, meta = _load()

    row = {"text": case.get("text", "")}
    for col in meta.get("metadata_columns", []):
        row[col] = str(case.get(col) or "unknown")
    frame = pd.DataFrame([row])

    proba = float(model.predict_proba(frame)[0, 1])
    label = int(proba >= 0.5)
    confidence = abs(proba - 0.5) * 2

    return {
        "label": label,
        "probability": proba,
        "confidence": confidence,
        "model_version": meta["version"],
        "disclaimer": (
            "Statistical estimate from patterns in past cases. Not legal advice "
            "and not a guarantee of any outcome."
        ),
    }
