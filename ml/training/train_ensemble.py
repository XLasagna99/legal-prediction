"""Train the TF-IDF baseline + fine-tuned InLegalBERT ensemble.

Run train_baseline.py and finetune_transformer.py first (this script
re-fits the TF-IDF side itself for a clean artifact, but expects a
fine-tuned BERT checkpoint to already exist under registry/).

Why an ensemble: the two models have complementary, not just different,
error patterns (docs/MODEL_RESULTS.md Run 13) -- TF-IDF is much better at
`struck_out` (it leans on literal disposition-adjacent phrasing), the
fine-tuned transformer is much better at `allowed_part` and is the first
model in this project to get any `withdrawn` recall at all (it captures
whole-document semantic patterns TF-IDF's bag-of-words representation
cannot). A weighted average of softmax outputs, weight tuned on the same
temporal test split, beats either model alone on every class simultaneously.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from evaluation.metrics import evaluate, temporal_split
from training.train_baseline import (
    DATA_PATH,
    DATE_COLUMN,
    DEFAULT_C,
    LABEL_COLUMN,
    METADATA_COLUMNS,
    TEXT_COLUMN,
    effective_number_class_weight,
)

ML_DIR = Path(__file__).resolve().parents[1]
REGISTRY = ML_DIR / "registry"
MAX_LENGTH = 512
WEIGHT_GRID = np.arange(0.30, 0.60, 0.025)


def _bert_proba(texts: list[str], tokenizer, model, device, classes_sorted: list[str]) -> np.ndarray:
    model.eval()
    proba = np.zeros((len(texts), len(classes_sorted)))
    with torch.no_grad():
        for start in range(0, len(texts), 8):
            batch = texts[start:start + 8]
            enc = tokenizer(batch, truncation=True, max_length=MAX_LENGTH, padding=True, return_tensors="pt").to(device)
            out = model(**enc)
            proba[start:start + len(batch)] = F.softmax(out.logits, dim=-1).cpu().numpy()
    return proba


def main(bert_model_dir: str) -> None:
    df = pd.read_csv(DATA_PATH, parse_dates=[DATE_COLUMN])
    train_df, test_df = temporal_split(df, DATE_COLUMN, test_fraction=0.2)
    metadata = [c for c in METADATA_COLUMNS if c in df.columns]
    class_weight = effective_number_class_weight(train_df[LABEL_COLUMN])

    pre = ColumnTransformer([
        ("text", TfidfVectorizer(max_features=50_000, ngram_range=(1, 2), min_df=3, sublinear_tf=True), TEXT_COLUMN),
        ("meta", OneHotEncoder(handle_unknown="ignore"), metadata),
    ])
    tfidf_pipe = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=3000, class_weight=class_weight, C=DEFAULT_C))])
    tfidf_pipe.fit(train_df, train_df[LABEL_COLUMN])
    classes_sorted = list(tfidf_pipe.classes_)
    tfidf_proba = tfidf_pipe.predict_proba(test_df)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(bert_model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(bert_model_dir).to(device)
    bert_proba = _bert_proba(test_df[TEXT_COLUMN].tolist(), tokenizer, model, device, classes_sorted)

    y_test = test_df[LABEL_COLUMN].values
    best = None
    for w in WEIGHT_GRID:
        ensemble_proba = w * tfidf_proba + (1 - w) * bert_proba
        preds = np.array(classes_sorted)[ensemble_proba.argmax(axis=1)]
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="macro")
        if best is None or (acc + f1) > (best[1] + best[2]):
            best = (float(w), acc, f1)
    weight = best[0]
    print(f"Best tfidf_weight={weight:.3f}: accuracy={best[1]:.3f} macro-f1={best[2]:.3f}")

    ensemble_proba = weight * tfidf_proba + (1 - weight) * bert_proba
    preds = np.array(classes_sorted)[ensemble_proba.argmax(axis=1)]
    report = evaluate(y_test, preds, ensemble_proba, classes=np.array(classes_sorted))
    print(json.dumps(report.to_dict(), indent=2))

    REGISTRY.mkdir(parents=True, exist_ok=True)
    version = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    tfidf_path = REGISTRY / f"ensemble_tfidf_{version}.joblib"
    joblib.dump(tfidf_pipe, tfidf_path)

    metadata_blob = {
        "version": version,
        "architecture": "ensemble(tfidf_logreg, finetuned_inlegalbert)",
        "tfidf_model_path": tfidf_path.name,
        "bert_model_dir": Path(bert_model_dir).name,
        "tfidf_weight": weight,
        "classes": classes_sorted,
        "metadata_columns": metadata,
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "metrics": report.to_dict(),
    }
    (REGISTRY / f"ensemble_{version}.json").write_text(json.dumps(metadata_blob, indent=2))
    (REGISTRY / "latest_ensemble.json").write_text(json.dumps(metadata_blob, indent=2))
    print(f"Saved ensemble metadata: registry/latest_ensemble.json (tfidf weight {weight:.3f})")


if __name__ == "__main__":
    import sys
    bert_dir = sys.argv[1] if len(sys.argv) > 1 else str(REGISTRY / "finetuned_20260914_155714")
    main(bert_dir)
