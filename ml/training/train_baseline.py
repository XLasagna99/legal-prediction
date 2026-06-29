"""Train a baseline legal-judgment-prediction model.

Baseline = TF-IDF over pre-decision text + a few metadata columns, fed to
logistic regression. It is intentionally simple and interpretable. Treat its
score as your honest performance floor: if a transformer can't beat this,
something is wrong with your data or your framing. If this baseline scores
suspiciously high (e.g. >0.95 AUC), suspect leakage first.

Expected input: data/processed/cases.csv with at least these columns:
    text          - pre-decision text (facts/complaint/brief), NO opinion text
    filed_date    - date used for the temporal split (ISO format)
    outcome       - binary label (0/1)
Optional metadata columns are listed in METADATA_COLUMNS below.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
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


def build_pipeline(metadata_columns: list[str]) -> Pipeline:
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
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
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
    pipeline = build_pipeline(metadata)
    pipeline.fit(train_df, train_df[LABEL_COLUMN])

    proba = pipeline.predict_proba(test_df)[:, 1]
    preds = (proba >= 0.5).astype(int)
    report = evaluate(test_df[LABEL_COLUMN], preds, proba)

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
