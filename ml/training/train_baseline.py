"""Train a baseline legal-judgment-prediction model.

Baseline = TF-IDF over pre-decision text + a few metadata columns, fed to
logistic regression. It is intentionally simple and interpretable. Treat its
score as your honest performance floor: if a transformer can't beat this,
something is wrong with your data or your framing. If this baseline scores
suspiciously high (e.g. >0.95 AUC), suspect leakage first.

Expected input: data/processed/cases.csv with at least these columns:
    text          - pre-decision text (facts/complaint/brief), NO opinion text
    filed_date    - date used for the temporal split (ISO format)
    outcome       - categorical label (2 or more classes; sklearn's
                    LogisticRegression and evaluate() both handle either)
Optional metadata columns are listed in METADATA_COLUMNS below.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from evaluation.metrics import evaluate, temporal_split

ML_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = ML_DIR.parent
DATA_PATH = REPO_ROOT / "data" / "processed" / "cases.csv"
REGISTRY = ML_DIR / "registry"

TEXT_COLUMN = "text"
DATE_COLUMN = "filed_date"
LABEL_COLUMN = "outcome"
METADATA_COLUMNS = ["court", "case_type", "represented"]  # adjust to your data

# Class-Balanced Loss (Cui et al., CVPR 2019, https://arxiv.org/abs/1901.05555):
# weight per class inversely proportional to the "effective number" of
# samples, (1 - beta^n) / (1 - beta), rather than raw inverse frequency.
# Near-duplicate examples in a class contribute diminishing marginal
# information, so a big class isn't punished as harshly as pure 1/n would,
# and a tiny class (e.g. allowed_part, ~4% of this dataset) isn't
# over-weighted to the point of instability the way sklearn's "balanced"
# (pure inverse frequency) can be. See docs/MODEL_IMPROVEMENTS.md Phase 0.3.
EFFECTIVE_NUMBER_BETA = 0.999


def effective_number_class_weight(y, beta: float = EFFECTIVE_NUMBER_BETA) -> dict:
    """Per-class weight dict for `LogisticRegression(class_weight=...)`,
    normalized so the *sample-weighted* average weight is 1 -- i.e.
    sum(weight[c] * count[c] for c) == len(y) -- matching sklearn's
    "balanced" convention (`sklearn.utils.class_weight.compute_class_weight`),
    so the two schemes are comparable in overall loss magnitude and differ
    only in how they treat rare vs. common classes, not in total scale.

    Normalizing to a simple per-class average of 1 instead (i.e. dividing by
    `len(counts)` rather than by the sample-weighted sum) was tried first and
    is a real bug to avoid: with 5 very unequally-sized classes, the rare
    classes' large raw weights dominate a simple average, so that scheme
    silently shrinks every weight by several-fold relative to "balanced" --
    which, against a fixed `C`, acts like added L2 regularization pressure
    and drags accuracy down across *all* classes, not just the rare ones.
    That confounded an earlier evaluation of this function -- see
    docs/MODEL_RESULTS.md Run 4 vs Run 5.
    """
    counts = pd.Series(y).value_counts()
    effective_num = 1.0 - np.power(beta, counts)
    raw_weight = (1.0 - beta) / effective_num
    n_total = counts.sum()
    scale = n_total / (raw_weight * counts).sum()
    normalized = raw_weight * scale
    return normalized.to_dict()


# Regularization strength for the final LogisticRegression. sklearn's default
# (C=1.0) was left untuned until 2026-09-14 (docs/MODEL_IMPROVEMENTS.md Phase
# 0.5); grid-searching C against the growing dataset consistently favored a
# much weaker penalty (larger C) than the default -- see docs/MODEL_RESULTS.md
# Run 12 for the sweep this value is chosen from. Re-sweep if the dataset
# composition changes materially (it's sensitive to n and class balance).
DEFAULT_C = 10.0


def build_pipeline(metadata_columns: list[str], class_weight="balanced", C: float = DEFAULT_C) -> Pipeline:
    features = ColumnTransformer(
        transformers=[
            (
                "text",
                TfidfVectorizer(
                    max_features=50_000,
                    ngram_range=(1, 2),
                    min_df=3,
                    sublinear_tf=True,
                ),
                TEXT_COLUMN,
            ),
            (
                "meta",
                OneHotEncoder(handle_unknown="ignore"),
                metadata_columns,
            ),
        ],
        remainder="drop",
    )
    return Pipeline(
        steps=[
            ("features", features),
            ("clf", LogisticRegression(max_iter=2000, class_weight=class_weight, C=C)),
        ]
    )


def main() -> None:
    if not DATA_PATH.exists():
        raise SystemExit(
            f"No dataset at {DATA_PATH}. See docs/DATA.md for how to build one."
        )

    df = pd.read_csv(DATA_PATH, parse_dates=[DATE_COLUMN])
    df = df.dropna(subset=[TEXT_COLUMN, LABEL_COLUMN, DATE_COLUMN])

    metadata = [c for c in METADATA_COLUMNS if c in df.columns]
    for col in metadata:
        df[col] = df[col].fillna("unknown").astype(str)

    train_df, test_df = temporal_split(df, DATE_COLUMN, test_fraction=0.2)
    class_weight = effective_number_class_weight(train_df[LABEL_COLUMN])
    pipeline = build_pipeline(metadata, class_weight=class_weight)
    pipeline.fit(train_df, train_df[LABEL_COLUMN])

    # predict_proba's full matrix + predict() (argmax) rather than a manual
    # [:, 1] slice + 0.5 threshold -- the latter only makes sense for binary
    # labels, and `outcome` may now have more than two categories.
    proba = pipeline.predict_proba(test_df)
    preds = pipeline.predict(test_df)
    report = evaluate(test_df[LABEL_COLUMN], preds, proba, classes=pipeline.classes_)

    REGISTRY.mkdir(parents=True, exist_ok=True)
    version = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    model_path = REGISTRY / f"baseline_{version}.joblib"
    joblib.dump(pipeline, model_path)

    metadata_blob = {
        "version": version,
        "model_path": model_path.name,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "train_date_range": [
            str(train_df[DATE_COLUMN].min()),
            str(train_df[DATE_COLUMN].max()),
        ],
        "test_date_range": [
            str(test_df[DATE_COLUMN].min()),
            str(test_df[DATE_COLUMN].max()),
        ],
        "metadata_columns": metadata,
        "class_weight_scheme": f"effective_number(beta={EFFECTIVE_NUMBER_BETA})",
        "class_weights": {str(k): round(v, 4) for k, v in class_weight.items()},
        "C": DEFAULT_C,
        "metrics": report.to_dict(),
    }
    (REGISTRY / "latest.json").write_text(json.dumps(metadata_blob, indent=2))
    (REGISTRY / f"baseline_{version}.json").write_text(json.dumps(metadata_blob, indent=2))

    print(json.dumps(metadata_blob["metrics"], indent=2))
    if (report.roc_auc or 0) > 0.95:
        print("\n[!] AUC > 0.95 on a baseline is a classic leakage smell. "
              "Check that no post-decision text slipped into `text`.")


if __name__ == "__main__":
    main()
