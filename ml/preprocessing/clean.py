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


# --- Singapore Courts (elitigation.sg) source, see ingestion/singapore_courts.py ---
#
# Unlike CourtListener, elitigation.sg has no structured "outcome" or
# "filed_date" field -- the raw record is just the full judgment text plus
# a citation and decision date. `filed_date` below is really the decision
# date (the only date the source gives us), and `outcome` has to be guessed
# from the judgment's own concluding language via `extract_outcome_heuristic`.
# Singapore judgments also don't phrase dispositions the way US opinions do
# ("I dismiss the application" rather than "the motion is denied"), so the
# base LEAKAGE_PATTERNS above miss most of them -- these patterns catch that.

SG_LEAKAGE_PATTERNS = [
    r"\bi (hereby |therefore |accordingly )?(have )?(dismiss(ed)?|allow(ed)?|grant(ed)?|refus(ed|e))\b",
    r"\b(dismissed|allowed|struck out|refused) with costs\b",
    r"\b(the )?(application|appeal|claim|action|originating (application|claim)) (is|are|was|were) (dismissed|allowed|granted|refused|struck out)\b",
    # Added for the struck_out/withdrawn outcome classes (see
    # extract_outcome_heuristic below) -- these two categories didn't exist
    # when the patterns above were written, so nothing stripped their own
    # disposition wording. Confirmed via inspecting a trained model's
    # coefficients that "struck out"/"withdrawn" were leaking directly in as
    # top features for their own classes.
    r"\b(struck out|strike out|striking out)\b",
    r"\bwithdrawn\b",
    r"\ballow(?:ed)? (?:the \w+(?: \w+)? )?to withdraw\b",
    r"\bpermitted to withdraw\b",
]
_SG_LEAKAGE_RE = re.compile("|".join(SG_LEAKAGE_PATTERNS), flags=re.IGNORECASE)

def strip_sg_leakage(text: str) -> str:
    """Like `strip_leakage`, but for Singapore-style disposition phrasing
    ("I dismiss the application") that the base patterns don't catch.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text or "")
    kept = [s for s in sentences if not _SG_LEAKAGE_RE.search(s)]
    return normalize_whitespace(" ".join(kept))


# Five disposition categories, each with its own pattern. `withdrawn` and
# `struck_out` used to be silently folded into `dismissed`/`allowed` by the
# old binary heuristic -- e.g. "I allowed the claimants to withdraw SUM 733"
# came out as outcome=1 ("allowed"), which reads as a win when a withdrawal
# is really neither side prevailing on the merits. Splitting them out fixes
# that.
_WITHDRAWN_MONEY_CONTEXT_RE = re.compile(
    # A bare "\bwithdrawn\b" also matches financial withdrawals ("the sum
    # withdrawn from the account") -- surfaced when widening `window` in
    # extract_outcome_heuristic pulled in more of these false positives than
    # the original 3,000-char tail did (see docs/MODEL.md Changelog). Case
    # numbers (e.g. "OA 498", "SUM 733") are the usual subject of a genuine
    # case-withdrawal sentence, not a fixed noun list, so exclude by nearby
    # financial-context words instead of requiring a specific subject.
    r"(?:sum|amount|monies|money|fund|cash|balance|deposit|account|cpf|\$\d)s?\W+(?:\w+\W+){0,4}withdrawn"
    r"|withdrawn\W+(?:\w+\W+){0,6}(?:from|account|cpf|balance|\$\d)",
    flags=re.IGNORECASE,
)
_WITHDRAWN_RE = re.compile(
    r"\bwithdrawn\b"
    r"|\ballow(?:ed)?\s+(?:the\s+\w+(?:\s+\w+)?\s+)?to\s+withdraw\b"
    r"|\bpermitted\s+to\s+withdraw\b",
    flags=re.IGNORECASE,
)


def _find_withdrawn_matches(tail: str):
    """Matches of _WITHDRAWN_RE, excluding ones in obvious financial-withdrawal context."""
    return [
        m for m in _WITHDRAWN_RE.finditer(tail)
        if not _WITHDRAWN_MONEY_CONTEXT_RE.search(tail[max(0, m.start() - 60):m.end() + 60])
    ]
_ALLOWED_PART_RE = re.compile(
    # Allows an intervening object phrase ("allow the application in part"),
    # same style as _WITHDRAWN_RE -- found missing via a unit test that used
    # exactly this phrasing and got miscategorized as allowed_full instead.
    r"\b(?:allow(?:ed)?|grant(?:ed)?)\s+(?:the\s+\w+(?:\s+\w+)?\s+)?in\s+part\b"
    r"|\bpartial(?:ly)?\s+(?:allow(?:ed)?|grant(?:ed)?)\b",
    flags=re.IGNORECASE,
)
_STRUCK_OUT_RE = re.compile(r"\bstruck\s+out\b|\bstrike\s+out\b", flags=re.IGNORECASE)
_DISMISSED_RE = re.compile(
    r"\bi\s+(?:have\s+|hereby\s+|therefore\s+|accordingly\s+)*(?:dismiss(?:ed)?|refus(?:ed|e))\b"
    r"|\b(?:the\s+)?(?:application|appeal|claim|action|originating\s+(?:application|claim))\s+"
    r"(?:is|are|was|were)\s+(?:dismissed|refused)\b",
    flags=re.IGNORECASE,
)
_ALLOWED_FULL_RE = re.compile(
    r"\bi\s+(?:have\s+|hereby\s+|therefore\s+|accordingly\s+)*(?:allow(?:ed)?|grant(?:ed)?)\b"
    r"|\b(?:the\s+)?(?:application|appeal|claim|action|originating\s+(?:application|claim))\s+"
    r"(?:is|are|was|were)\s+(?:allowed|granted)\b",
    flags=re.IGNORECASE,
)

# Checked in this order: for each category, find its LAST match in the
# window (closest to the real disposition). Whichever category's last match
# is furthest along in the text wins overall; a category earlier in this
# list breaks ties (two categories' last matches at essentially the same
# position) in favor of the more specific one, e.g. "struck_out" over
# "dismissed" for "...correct to strike out the statement of claim...".
_CATEGORY_PATTERNS = [
    ("withdrawn", _WITHDRAWN_RE),
    ("struck_out", _STRUCK_OUT_RE),
    ("allowed_part", _ALLOWED_PART_RE),
    ("dismissed", _DISMISSED_RE),
    ("allowed_full", _ALLOWED_FULL_RE),
]


def extract_outcome_heuristic(text: str, window: int = 6000) -> str | None:
    """Guess a disposition category from the final `window` characters of a
    Singapore judgment, where the disposition is announced: one of
    "dismissed", "allowed_full", "allowed_part", "struck_out", "withdrawn",
    or None if no disposition phrase was found in the window.

    `window` was widened from 3,000 to 6,000 (docs/MODEL_IMPROVEMENTS.md
    Phase 1.2) after measuring the tradeoff directly: 3,000 misses real
    dispositions that fall slightly further back (coverage 48.4% -> 53.0%
    of raw cases at 6,000), while windows much wider than 6,000 (tested up
    to the full document) pull in a fast-rising rate of false positives --
    "struck out"/"withdrawn" mentioned in case history, a losing argument,
    or (for "withdrawn" specifically) a bank/CPF withdrawal rather than the
    court's own final disposition. 6,000 was the largest window where a
    manual spot-check of every newly-rescued case still showed genuine
    disposition language near a "Conclusion" heading.

    This is a rule-based heuristic, not ground truth -- it will mislabel or
    miss cases with unusual phrasing (multi-issue rulings, hypothetical
    "had X not happened, I would have..." asides, interlocutory applications
    with no dismiss/allow wording at all). Treat a None as "couldn't tell",
    not "no outcome" -- drop it rather than guessing. Spot-check a sample
    before trusting this at scale.
    """
    tail = text[-window:] if text else ""
    best_category = None
    best_pos = -1
    for category, pattern in _CATEGORY_PATTERNS:
        matches = _find_withdrawn_matches(tail) if category == "withdrawn" else list(pattern.finditer(tail))
        if not matches:
            continue
        pos = matches[-1].start()
        if pos > best_pos:
            best_pos = pos
            best_category = category
    return best_category


# --- Level 1 structured field: costs_amount_bucket (docs/OUTCOME_PREDICTION.md) ---
#
# Singapore judgments state a costs order in fairly formulaic phrasing near
# the disposition ("costs fixed at $16,000", "awarded costs ... in the sum
# of $25,000 all-in", "costs at $10,000 ... to be paid by X to Y"). This
# extracts the dollar figure nearest such phrasing and bins it, rather than
# treating it as a raw-number regression target -- see OUTCOME_PREDICTION.md
# Level 2 for why binning, not regression, is the realistic choice at this
# dataset's scale.

_COSTS_AMOUNT_RE = re.compile(
    r"costs?\b[^.]{0,120}?\$\s?([\d][\d,]*(?:\.\d+)?)",
    flags=re.IGNORECASE,
)

_COSTS_BUCKETS = [
    (10_000, "under_10k"),
    (50_000, "10k_to_50k"),
]


def _bucket_costs_amount(amount: float) -> str:
    for threshold, label in _COSTS_BUCKETS:
        if amount < threshold:
            return label
    return "over_50k"


def extract_costs_amount_bucket(text: str, window: int = 6000) -> str:
    """Guess a binned costs order from the final `window` characters of a
    Singapore judgment: "none" (no costs figure found), "under_10k",
    "10k_to_50k", or "over_50k".

    Rule-based, not ground truth -- picks the LAST dollar figure that
    appears near the word "costs" in the window (closest to the actual
    order, past any earlier mention of costs in submissions/argument).
    Misses costs stated as a formula ("costs on the indemnity basis, to be
    taxed if not agreed") rather than a fixed figure -- those correctly
    return "none" since no bindable number was stated in the judgment.
    """
    tail = text[-window:] if text else ""
    matches = list(_COSTS_AMOUNT_RE.finditer(tail))
    if not matches:
        return "none"
    amount_str = matches[-1].group(1).replace(",", "")
    try:
        amount = float(amount_str)
    except ValueError:
        return "none"
    return _bucket_costs_amount(amount)


def clean_singapore_record(raw: dict) -> dict:
    """Map one raw elitigation.sg record to the processed schema."""
    text = normalize_whitespace(raw.get("text", ""))
    outcome = extract_outcome_heuristic(text)
    costs_amount_bucket = extract_costs_amount_bucket(text)
    stripped = strip_sg_leakage(strip_leakage(text))
    citation = raw.get("citation") or ""
    if "SGCA" in citation:
        court = "SGCA"
    elif "SGHCR" in citation:
        court = "SGHCR"
    elif "SGHC" in citation:
        court = "SGHC"
    else:
        court = "unknown"
    return {
        "text": stripped,
        "filed_date": raw.get("decision_date"),
        "outcome": outcome,
        "costs_amount_bucket": costs_amount_bucket,
        "court": court,
        "case_type": "unknown",
        "represented": "unknown",
    }
