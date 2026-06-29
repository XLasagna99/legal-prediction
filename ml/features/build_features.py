"""Feature engineering helpers.

For the baseline, vectorization lives inside the sklearn Pipeline
(see training/train_baseline.py), so this module is where you add anything the
pipeline does not cover: derived metadata, text-length signals, party features,
embeddings, etc. Keep it leakage-safe — only pre-decision information.
"""
from __future__ import annotations

import pandas as pd


def add_text_length(df: pd.DataFrame, text_column: str = "text") -> pd.DataFrame:
    df = df.copy()
    df["text_char_len"] = df[text_column].str.len().fillna(0).astype(int)
    df["text_word_len"] = df[text_column].str.split().map(len)
    return df


def assemble(df: pd.DataFrame) -> pd.DataFrame:
    """Single entry point: apply derived features in order."""
    return add_text_length(df)
