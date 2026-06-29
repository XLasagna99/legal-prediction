"""Load the latest model artifact and make predictions.

This module is imported by the backend. It deliberately has no training
dependencies and does no I/O beyond loading the artifact once.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "registry"


class ModelNotTrainedError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def _load() -> tuple[Any, dict]:
    latest = REGISTRY / "latest.json"
    if not latest.exists():
        raise ModelNotTrainedError(
            "No trained model found. Run `python -m training.train_baseline`."
        )
    meta = json.loads(latest.read_text())
    model = joblib.load(REGISTRY / meta["model_path"])
    return model, meta


def model_info() -> dict:
    _, meta = _load()
    return {
        "version": meta["version"],
        "trained_at": meta["trained_at"],
        "metrics": meta["metrics"],
    }


def predict(case: dict) -> dict:
    """Predict an outcome probability for a single case.

    `case` should contain at least a `text` field plus whatever metadata
    columns the model was trained on. Missing metadata is filled with
    "unknown".

    Returns label, probability, and a short uncertainty note. The note is part
    of the contract: never surface a bare verdict without it.
    """
    model, meta = _load()

    row = {"text": case.get("text", "")}
    for col in meta.get("metadata_columns", []):
        row[col] = str(case.get(col, "unknown") or "unknown")
    frame = pd.DataFrame([row])

    proba = float(model.predict_proba(frame)[0, 1])
    label = int(proba >= 0.5)
    confidence = abs(proba - 0.5) * 2  # 0 at the coin-flip line, 1 at the extremes

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
