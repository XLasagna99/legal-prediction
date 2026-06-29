"""Clean raw cases and, above all, strip post-decision leakage.

Turn raw ingested records into a tidy table with columns:
    text, filed_date, outcome, [metadata...]

The functions here are scaffolding. The leakage filter is the part you must
take seriously: phrases that announce the result ("affirmed", "we reverse",
"the motion is denied") are giveaways. If they survive into `text`, your model
is reading the answer instead of predicting it.
"""
from __future__ import annotations

import re

# Extend this list for your jurisdiction and case type.
LEAKAGE_PATTERNS = [
    r"\b(affirm(ed|s)?|revers(ed|es)?|remand(ed|s)?)\b",
    r"\bthe (motion|appeal|petition) is (granted|denied|dismissed)\b",
    r"\b(we|the court) (hold|find|conclude|rule)s?\b",
    r"\bit is (so )?ordered\b",
]
_LEAKAGE_RE = re.compile("|".join(LEAKAGE_PATTERNS), flags=re.IGNORECASE)


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def strip_leakage(text: str) -> str:
    """Remove sentences that announce the outcome.

    Crude but illustrative: split into sentences, drop any that match a
    leakage pattern. In practice, prefer using genuinely pre-decision documents
    over trying to scrub an opinion after the fact.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text or "")
    kept = [s for s in sentences if not _LEAKAGE_RE.search(s)]
    return normalize_whitespace(" ".join(kept))


def clean_record(raw: dict) -> dict:
    """Map one raw record to the processed schema. Adapt field names to source."""
    return {
        "text": strip_leakage(normalize_whitespace(raw.get("text", ""))),
        "filed_date": raw.get("filed_date"),
        "outcome": raw.get("outcome"),
        "court": raw.get("court", "unknown"),
        "case_type": raw.get("case_type", "unknown"),
        "represented": raw.get("represented", "unknown"),
    }
