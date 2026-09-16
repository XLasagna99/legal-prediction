"""Phase 2 (docs/MODEL_IMPROVEMENTS.md): linear-probe fine-tuning over a
frozen legal-domain transformer, as an alternative to the TF-IDF baseline
in train_baseline.py.

Why linear-probe (freeze the backbone, train only a classification head)
instead of full fine-tuning: docs/MODEL_IMPROVEMENTS.md Phase 2.3 flags
full fine-tuning of a large transformer on a few hundred examples as
higher-variance / more overfitting-prone than TF-IDF+LogReg at this
dataset's scale (docs/MODEL.md problem #3/#4 already found the baseline
overfitting to judge/party names in a much lower-capacity model). A frozen
backbone + linear head is the lower-risk first test of whether transformer
representations beat bag-of-words at all before spending compute on
anything heavier.

Backbone: InLegalBERT (law-ai/InLegalBERT) -- Legal-BERT further pretrained
on 5.4M Indian case documents, the nearest large Commonwealth/common-law
corpus to Singapore's system (docs/MODEL_IMPROVEMENTS.md Phase 2.1). A
512-token window is standard BERT context; `truncation_side="left"` keeps
the LAST 512 tokens of each case's (leakage-stripped) text, since Singapore
judgments state their disposition and surrounding reasoning near the end,
not the start (same rationale as extract_outcome_heuristic's tail window).

Expected input: same data/processed/cases.csv as train_baseline.py.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from transformers import AutoModel, AutoTokenizer

from evaluation.metrics import evaluate, temporal_split
from training.train_baseline import (
    DATA_PATH,
    DATE_COLUMN,
    LABEL_COLUMN,
    METADATA_COLUMNS,
    TEXT_COLUMN,
    effective_number_class_weight,
)

ML_DIR = Path(__file__).resolve().parents[1]
REGISTRY = ML_DIR / "registry"

BACKBONE_NAME = "law-ai/InLegalBERT"
MAX_LENGTH = 512
BATCH_SIZE = 8
POOLING = "mean"
MAX_CHUNKS = 24  # ~12,000 tokens/doc cap, to bound runtime on very long judgments


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor, pooling: str) -> torch.Tensor:
    if pooling == "cls":
        return last_hidden_state[:, 0, :]
    mask = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def embed_texts(texts: list[str], tokenizer, model, device: torch.device, pooling: str = "mean") -> np.ndarray:
    """Frozen-backbone embeddings, batched, no gradients.

    `pooling="mean"` (attention-mask-weighted mean over all tokens) is the
    default -- the raw [CLS] token ("pooling="cls"") was tried first and
    measured *worse* than the TF-IDF baseline (docs/MODEL_RESULTS.md Run 7),
    consistent with the known finding that an un-fine-tuned BERT's [CLS]
    vector isn't a reliable sentence representation on its own; mean pooling
    over the last hidden layer is the standard fix.
    """
    model.eval()
    embeddings = []
    with torch.no_grad():
        for start in range(0, len(texts), BATCH_SIZE):
            batch = texts[start:start + BATCH_SIZE]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            ).to(device)
            out = model(**encoded)
            vec = _pool(out.last_hidden_state, encoded["attention_mask"], pooling)
            embeddings.append(vec.cpu().numpy())
    return np.concatenate(embeddings, axis=0)


def embed_texts_chunked(
    texts: list[str], tokenizer, model, device: torch.device, pooling: str = "mean"
) -> np.ndarray:
    """Like embed_texts, but covers the WHOLE document instead of only the
    last 512 tokens: split into non-overlapping 512-token chunks, embed each
    chunk, and average the chunk embeddings into one document vector.

    Motivated by docs/MODEL_IMPROVEMENTS.md Phase 2.2 -- per-class median
    judgment length here is ~47,600-103,000 characters (roughly 10,000-
    20,000+ tokens), so a single 512-token window (embed_texts) sees under
    5% of a typical document. This is the cheap approximation of Phase 2.2's
    Longformer recommendation: it doesn't give the model cross-chunk
    attention the way a real long-context transformer would, but it does
    let every part of the document contribute to the final embedding,
    which a single truncated window cannot.
    """
    model.eval()
    doc_embeddings = []
    with torch.no_grad():
        for text in texts:
            # No truncation here -- we want every token, split across chunks.
            token_ids = tokenizer(text, truncation=False, add_special_tokens=False)["input_ids"]
            if not token_ids:
                doc_embeddings.append(np.zeros(model.config.hidden_size, dtype=np.float32))
                continue
            chunk_size = MAX_LENGTH - 2  # room for [CLS]/[SEP]
            chunk_vecs = []
            starts = list(range(0, len(token_ids), chunk_size))[:MAX_CHUNKS]
            for start in starts:
                chunk = token_ids[start:start + chunk_size]
                encoded = tokenizer.prepare_for_model(
                    chunk, return_tensors="pt", truncation=True, max_length=MAX_LENGTH
                ).to(device)
                encoded = {k: v.unsqueeze(0) if v.dim() == 1 else v for k, v in encoded.items()}
                out = model(**encoded)
                mask = encoded["attention_mask"]
                vec = _pool(out.last_hidden_state, mask, pooling)
                chunk_vecs.append(vec.cpu().numpy()[0])
            doc_embeddings.append(np.mean(chunk_vecs, axis=0))
    return np.array(doc_embeddings, dtype=np.float32)


def main() -> None:
    if not DATA_PATH.exists():
        raise SystemExit(f"No dataset at {DATA_PATH}. See docs/DATA.md for how to build one.")

    df = pd.read_csv(DATA_PATH, parse_dates=[DATE_COLUMN])
    df = df.dropna(subset=[TEXT_COLUMN, LABEL_COLUMN, DATE_COLUMN])

    train_df, test_df = temporal_split(df, DATE_COLUMN, test_fraction=0.2)

    device = _device()
    print(f"Loading {BACKBONE_NAME} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(BACKBONE_NAME)
    tokenizer.truncation_side = "left"  # keep the LAST 512 tokens, not the first
    model = AutoModel.from_pretrained(BACKBONE_NAME).to(device)
    for p in model.parameters():
        p.requires_grad = False  # frozen backbone -- linear probe only

    print(f"Embedding {len(train_df)} train + {len(test_df)} test cases (chunked, whole-document)...")
    train_emb = embed_texts_chunked(train_df[TEXT_COLUMN].tolist(), tokenizer, model, device, pooling=POOLING)
    test_emb = embed_texts_chunked(test_df[TEXT_COLUMN].tolist(), tokenizer, model, device, pooling=POOLING)

    class_weight = effective_number_class_weight(train_df[LABEL_COLUMN])
    clf = LogisticRegression(max_iter=2000, class_weight=class_weight)
    clf.fit(train_emb, train_df[LABEL_COLUMN])

    proba = clf.predict_proba(test_emb)
    preds = clf.predict(test_emb)
    report = evaluate(test_df[LABEL_COLUMN], preds, proba, classes=clf.classes_)

    REGISTRY.mkdir(parents=True, exist_ok=True)
    version = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    model_path = REGISTRY / f"transformer_{version}.joblib"
    joblib.dump(clf, model_path)

    metadata_blob = {
        "version": version,
        "model_path": model_path.name,
        "architecture": "frozen-linear-probe-chunked",
        "backbone": BACKBONE_NAME,
        "pooling": POOLING,
        "max_length": MAX_LENGTH,
        "max_chunks": MAX_CHUNKS,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "train_date_range": [str(train_df[DATE_COLUMN].min()), str(train_df[DATE_COLUMN].max())],
        "test_date_range": [str(test_df[DATE_COLUMN].min()), str(test_df[DATE_COLUMN].max())],
        "class_weight_scheme": "effective_number(beta=0.999)",
        "class_weights": {str(k): round(v, 4) for k, v in class_weight.items()},
        "metrics": report.to_dict(),
    }
    (REGISTRY / "latest_transformer.json").write_text(json.dumps(metadata_blob, indent=2))
    (REGISTRY / f"transformer_{version}.json").write_text(json.dumps(metadata_blob, indent=2))

    print(json.dumps(metadata_blob["metrics"], indent=2))


if __name__ == "__main__":
    main()
