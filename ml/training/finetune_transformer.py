"""Phase 2 escalation: full (not frozen) fine-tuning of InLegalBERT.

train_transformer.py's linear-probe design (frozen backbone) was chosen at
422 training examples specifically to avoid overfitting
(docs/MODEL_IMPROVEMENTS.md Phase 2.3). After the Phase 1.1 targeted
re-scrape grew the training set to ~1,300 examples (docs/MODEL_RESULTS.md
Run 12), that overfitting risk is lower, so this tries the next escalation:
unfreezing the backbone and fine-tuning it jointly with the classification
head, end to end, with a class-weighted loss for the same imbalance reasons
as train_baseline.py's effective_number_class_weight().

Same last-512-tokens-of-the-document choice as train_transformer.py (the
disposition and surrounding reasoning sit near the end of a Singapore
judgment), same InLegalBERT backbone.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from evaluation.metrics import evaluate, temporal_split
from training.train_baseline import DATA_PATH, DATE_COLUMN, LABEL_COLUMN, TEXT_COLUMN, effective_number_class_weight

ML_DIR = Path(__file__).resolve().parents[1]
REGISTRY = ML_DIR / "registry"

BACKBONE_NAME = "law-ai/InLegalBERT"
MAX_LENGTH = 512
BATCH_SIZE = 8
EPOCHS = 4
LR = 2e-5


class CaseDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, label_to_idx):
        self.encodings = tokenizer(
            texts, truncation=True, max_length=MAX_LENGTH, padding="max_length"
        )
        self.labels = [label_to_idx[label] for label in labels]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    df = pd.read_csv(DATA_PATH, parse_dates=[DATE_COLUMN])
    train_df, test_df = temporal_split(df, DATE_COLUMN, test_fraction=0.2)
    print(f"train={len(train_df)} test={len(test_df)}")

    classes = sorted(train_df[LABEL_COLUMN].unique())
    label_to_idx = {c: i for i, c in enumerate(classes)}
    idx_to_label = {i: c for c, i in label_to_idx.items()}

    class_weight_dict = effective_number_class_weight(train_df[LABEL_COLUMN])
    weight_tensor = torch.tensor(
        [class_weight_dict[idx_to_label[i]] for i in range(len(classes))], dtype=torch.float
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(BACKBONE_NAME)
    tokenizer.truncation_side = "left"  # keep the LAST 512 tokens
    model = AutoModelForSequenceClassification.from_pretrained(
        BACKBONE_NAME, num_labels=len(classes)
    ).to(device)

    train_ds = CaseDataset(train_df[TEXT_COLUMN].tolist(), train_df[LABEL_COLUMN].tolist(), tokenizer, label_to_idx)
    test_ds = CaseDataset(test_df[TEXT_COLUMN].tolist(), test_df[LABEL_COLUMN].tolist(), tokenizer, label_to_idx)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    optimizer = AdamW(model.parameters(), lr=LR)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")
            out = model(**batch)
            loss = F.cross_entropy(out.logits, labels, weight=weight_tensor)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        print(f"epoch {epoch+1}/{EPOCHS}: avg loss {total_loss/len(train_loader):.4f}")

    model.eval()
    all_proba = []
    all_preds = []
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            out = model(**batch)
            proba = F.softmax(out.logits, dim=-1).cpu().numpy()
            all_proba.append(proba)
            all_preds.extend(proba.argmax(axis=1).tolist())

    proba_matrix = np.concatenate(all_proba, axis=0)
    preds = [idx_to_label[i] for i in all_preds]
    y_test = test_df[LABEL_COLUMN].values

    report = evaluate(y_test, np.array(preds), proba_matrix, classes=np.array([idx_to_label[i] for i in range(len(classes))]))
    print(json.dumps(report.to_dict(), indent=2))

    from sklearn.metrics import classification_report as sk_report
    print(sk_report(y_test, preds))

    REGISTRY.mkdir(parents=True, exist_ok=True)
    version = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    model_dir = REGISTRY / f"finetuned_{version}"
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    metadata_blob = {
        "version": version,
        "architecture": "full-finetune",
        "backbone": BACKBONE_NAME,
        "max_length": MAX_LENGTH,
        "epochs": EPOCHS,
        "lr": LR,
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "metrics": report.to_dict(),
    }
    (REGISTRY / f"finetuned_{version}.json").write_text(json.dumps(metadata_blob, indent=2))
    (REGISTRY / "latest_finetuned.json").write_text(json.dumps(metadata_blob, indent=2))
    print(f"Saved to {model_dir}")


if __name__ == "__main__":
    main()
