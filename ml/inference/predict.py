"""Load the latest model artifact and make predictions.

This module is imported by the backend and by ml/agent/case_agent.py. It
deliberately has no training dependencies beyond what's needed to run
inference, and does no I/O beyond loading the artifact once (cached).

Prefers the ensemble model (registry/latest_ensemble.json -- TF-IDF +
fine-tuned InLegalBERT, docs/MODEL_RESULTS.md Run 13) if one has been
trained, since it beats either component alone on every class
simultaneously. Falls back to the plain TF-IDF baseline
(registry/latest.json) if no ensemble has been trained yet, so this module
works right after the first `python -m training.train_baseline` too.
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


class _SklearnPredictor:
    def __init__(self, pipeline, meta: dict):
        self.pipeline = pipeline
        self.meta = meta
        self.classes = list(pipeline.classes_)

    def predict_proba(self, case: dict):
        row = {"text": case.get("text", "")}
        for col in self.meta.get("metadata_columns", []):
            row[col] = str(case.get(col, "unknown") or "unknown")
        frame = pd.DataFrame([row])
        return self.classes, self.pipeline.predict_proba(frame)[0]


class _EnsemblePredictor:
    def __init__(self, meta: dict):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.meta = meta
        self.classes = meta["classes"]
        self.weight = meta["tfidf_weight"]
        self.tfidf_pipeline = joblib.load(REGISTRY / meta["tfidf_model_path"])
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        bert_dir = REGISTRY / meta["bert_model_dir"]
        self._tokenizer = AutoTokenizer.from_pretrained(bert_dir)
        self._bert_model = AutoModelForSequenceClassification.from_pretrained(bert_dir).to(self._device)
        self._bert_model.eval()

    def predict_proba(self, case: dict):
        import torch
        import torch.nn.functional as F

        row = {"text": case.get("text", "")}
        for col in self.meta.get("metadata_columns", []):
            row[col] = str(case.get(col, "unknown") or "unknown")
        frame = pd.DataFrame([row])
        tfidf_proba = self.tfidf_pipeline.predict_proba(frame)[0]
        tfidf_classes = list(self.tfidf_pipeline.classes_)
        # Reorder to match self.classes (both are the same set, but the
        # sklearn pipeline's own class order may not match the ensemble
        # metadata's stored order).
        tfidf_proba = [tfidf_proba[tfidf_classes.index(c)] for c in self.classes]

        with torch.no_grad():
            enc = self._tokenizer(
                [case.get("text", "")], truncation=True, max_length=512,
                padding=True, return_tensors="pt",
            ).to(self._device)
            logits = self._bert_model(**enc).logits
            bert_proba = F.softmax(logits, dim=-1).cpu().numpy()[0]

        combined = [self.weight * t + (1 - self.weight) * b for t, b in zip(tfidf_proba, bert_proba)]
        return self.classes, combined


class _Ensemble3Predictor(_EnsemblePredictor):
    """Adds a live GPT-4o few-shot call on top of _EnsemblePredictor's local
    (TF-IDF + fine-tuned BERT) ensemble -- docs/MODEL_RESULTS.md Run 14b/14c.

    Uses PER-CLASS mix weights, properly validated: fit on one half of the
    323-case test set, scored on the other, untouched half -- accuracy 0.568,
    macro-F1 0.460 there (Run 14c). An earlier global-weight version (0.85
    on every class) reported 0.591/0.455, but that weight had been tuned on
    the same 323 cases it was scored on; re-measured honestly with a val/test
    split it's actually 0.556/0.433 -- worse than the per-class version once
    both are compared without leakage. See docs/MODEL.md's 2026-09-15
    Changelog correction for the full accounting.

    Not the default: unlike the other predictors, every prediction here
    makes a real, metered OpenAI API call (~$0.05-0.06 each at gpt-4o's
    current per-token pricing and this module's prompt size) and is rate-
    limited to roughly one call per 16 seconds on this project's API key
    -- opt in explicitly via `USE_LLM_ENSEMBLE=1`, understanding both the
    per-call cost and latency.
    """

    def __init__(self, meta: dict):
        super().__init__(meta)
        from training.llm_classify import build_few_shot_examples, classify_one, _client, _tail
        import pandas as pd
        from evaluation.metrics import temporal_split
        from training.train_baseline import DATA_PATH, DATE_COLUMN

        self._classify_one = classify_one
        self._tail = _tail
        self._llm_client = _client()
        df = pd.read_csv(DATA_PATH, parse_dates=[DATE_COLUMN])
        train_df, _ = temporal_split(df, DATE_COLUMN, test_fraction=0.2)
        self._few_shot = build_few_shot_examples(train_df)
        # Per-class weight on the LOCAL ensemble (remainder on the LLM's
        # one-hot prediction), keyed by class name -- fit on a held-out val
        # split, not the reported test split (see docstring).
        default_weights = {"allowed_full": 0.8, "allowed_part": 0.95, "dismissed": 0.5, "struck_out": 0.8, "withdrawn": 0.95}
        self.llm_weights = meta.get("llm_ensemble_weights", default_weights)

    def predict_proba(self, case: dict):
        classes, local_proba = super().predict_proba(case)
        llm_label = self._classify_one(self._llm_client, self._few_shot, self._tail(case.get("text", "")))
        llm_onehot = [1.0 if c == llm_label else 0.0 for c in classes]
        combined = [
            self.llm_weights.get(c, 0.8) * loc + (1 - self.llm_weights.get(c, 0.8)) * llm
            for c, loc, llm in zip(classes, local_proba, llm_onehot)
        ]
        return classes, combined


@lru_cache(maxsize=1)
def _load() -> Any:
    import os

    if os.environ.get("USE_LLM_ENSEMBLE") == "1":
        ensemble3_meta_path = REGISTRY / "latest_ensemble3.json"
        if ensemble3_meta_path.exists():
            meta = json.loads(ensemble3_meta_path.read_text())
            return _Ensemble3Predictor(meta)

    ensemble_meta_path = REGISTRY / "latest_ensemble.json"
    if ensemble_meta_path.exists():
        meta = json.loads(ensemble_meta_path.read_text())
        return _EnsemblePredictor(meta)

    latest = REGISTRY / "latest.json"
    if not latest.exists():
        raise ModelNotTrainedError(
            "No trained model found. Run `python -m training.train_baseline`."
        )
    meta = json.loads(latest.read_text())
    pipeline = joblib.load(REGISTRY / meta["model_path"])
    return _SklearnPredictor(pipeline, meta)


def model_info() -> dict:
    predictor = _load()
    return {
        "version": predictor.meta["version"],
        "trained_at": predictor.meta.get("trained_at"),
        "metrics": predictor.meta["metrics"],
        "architecture": predictor.meta.get("architecture", "tfidf_logreg"),
    }


def predict(case: dict) -> dict:
    """Predict an outcome distribution for a single case.

    `case` should contain at least a `text` field plus whatever metadata
    columns the model was trained on. Missing metadata is filled with
    "unknown".

    Returns the full ranked class distribution (multi-class, not a single
    binary probability -- `docs/MODEL.md` §9), the top label, and a short
    uncertainty note. The note is part of the contract: never surface a bare
    verdict without it.
    """
    predictor = _load()
    classes, proba = predictor.predict_proba(case)
    ranked = sorted(zip(classes, [float(p) for p in proba]), key=lambda kv: -kv[1])
    top_label, top_proba = ranked[0]
    # Distance above a uniform-random guess over this many classes, not the
    # binary "distance from 0.5" this used before multi-class support --
    # 0 means "no better than guessing among the classes", 1 means certain.
    confidence = max(0.0, (top_proba - 1 / len(classes)) / (1 - 1 / len(classes)))

    return {
        "label": top_label,
        "probability": top_proba,
        "ranked_probabilities": [{"label": lbl, "probability": p} for lbl, p in ranked],
        "confidence": confidence,
        "model_version": predictor.meta["version"],
        "model_metrics": predictor.meta.get("metrics", {}),
        "disclaimer": (
            "Statistical estimate from patterns in past cases. Not legal advice "
            "and not a guarantee of any outcome."
        ),
    }
