"""Build data/processed/cases.csv from the raw Singapore Courts ingestion file.

Reads data/raw/singapore_bankruptcy_cases.jsonl (see
ingestion/singapore_courts.py), maps each record through
`clean_singapore_record()`, drops rows where the outcome heuristic couldn't
find a disposition, and writes the result to data/processed/cases.csv in the
schema `training/train_baseline.py` expects.

Run from the `ml/` directory: `python -m preprocessing.build_cases_csv`
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from preprocessing.clean import clean_singapore_record

ML_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = ML_DIR.parent
RAW_PATH = REPO_ROOT / "data" / "raw" / "singapore_bankruptcy_cases.jsonl"
OUT_PATH = REPO_ROOT / "data" / "processed" / "cases.csv"


def main() -> None:
    if not RAW_PATH.exists():
        raise SystemExit(f"No raw data at {RAW_PATH}. Run ingestion/singapore_courts.py first.")

    raw_records = []
    with RAW_PATH.open(encoding="utf-8") as f:
        for line in f:
            raw_records.append(json.loads(line))

    df = pd.DataFrame(clean_singapore_record(r) for r in raw_records)

    n_total = len(df)
    n_no_outcome = int(df["outcome"].isna().sum())
    n_no_text = int((df["text"].str.len() == 0).sum())

    df = df.dropna(subset=["text", "filed_date", "outcome"])
    df = df[df["text"].str.len() > 0]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    print(f"Read {n_total} raw record(s)")
    print(f"Dropped {n_no_outcome} with no outcome match from the heuristic ({n_no_outcome / n_total:.1%})")
    print(f"Dropped {n_no_text} with empty text")
    print(f"Wrote {len(df)} labeled case(s) to {OUT_PATH}")
    print("\nOutcome balance (dismissed / allowed_full / allowed_part / struck_out / withdrawn):")
    print(df["outcome"].value_counts(normalize=True))
    print("\nCourt breakdown:")
    print(df["court"].value_counts())
    print("\ncosts_amount_bucket balance (Level 1 structured field, docs/OUTCOME_PREDICTION.md):")
    print(df["costs_amount_bucket"].value_counts(normalize=True))


if __name__ == "__main__":
    main()
