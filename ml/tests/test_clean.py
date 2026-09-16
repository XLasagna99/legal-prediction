"""Tests for ml/preprocessing/clean.py.

Several of these are direct regression tests for the struck_out/withdrawn
leakage bug found and fixed on 2026-08-28 (docs/MODEL.md, docs/
MODEL_IMPROVEMENTS.md Phase 0.1): the outcome-heuristic's own disposition
words were surviving into `text` because SG_LEAKAGE_PATTERNS didn't cover
them. If that regresses, test_strip_sg_leakage_removes_struck_out and
test_strip_sg_leakage_removes_withdrawn should fail immediately.
"""
from preprocessing.clean import (
    clean_singapore_record,
    extract_costs_amount_bucket,
    extract_outcome_heuristic,
    normalize_whitespace,
    strip_leakage,
    strip_sg_leakage,
)


# --- normalize_whitespace ---


def test_normalize_whitespace_collapses_and_strips():
    assert normalize_whitespace("  a  b\n\tc  ") == "a b c"


def test_normalize_whitespace_handles_none():
    assert normalize_whitespace(None) == ""


# --- strip_leakage (US-style patterns) ---


def test_strip_leakage_removes_affirmed_sentence():
    text = "The facts are undisputed. The court affirmed the decision below. Costs follow the event."
    result = strip_leakage(text)
    assert "affirmed" not in result.lower()
    assert "facts are undisputed" in result
    assert "costs follow the event" in result.lower()


def test_strip_leakage_keeps_clean_text_untouched():
    text = "The plaintiff filed a complaint alleging breach of contract."
    assert strip_leakage(text) == text


# --- strip_sg_leakage: the regression tests for the 2026-08-28 fix ---


def test_strip_sg_leakage_removes_struck_out():
    text = "The background facts are set out above. The claim was struck out in its entirety. Nothing further arises."
    result = strip_sg_leakage(text)
    assert "struck out" not in result.lower()
    assert "background facts" in result.lower()
    assert "nothing further arises" in result.lower()


def test_strip_sg_leakage_removes_strike_out_and_striking_out_variants():
    for phrase in ["the court will strike out the pleading", "striking out is warranted here"]:
        result = strip_sg_leakage(f"Some context is set out here. {phrase}. More context follows.")
        assert phrase not in result.lower()
        assert "some context is set out here" in result.lower()
        assert "more context follows" in result.lower()


def test_strip_sg_leakage_removes_withdrawn():
    text = "The parties negotiated for some time. The application was withdrawn by consent. Costs are reserved."
    result = strip_sg_leakage(text)
    assert "withdrawn" not in result.lower()
    assert "negotiated for some time" in result.lower()


def test_strip_sg_leakage_removes_allowed_to_withdraw():
    text = "Background is set out at [10]. I allowed the claimants to withdraw SUM 733. However, had it not been withdrawn, I would have dismissed it."
    result = strip_sg_leakage(text)
    assert "withdraw" not in result.lower()
    assert "background is set out" in result.lower()


def test_strip_sg_leakage_does_not_remove_struck_off():
    # "struck off" (companies register) is a distinct legal concept from
    # "struck out" (pleadings/actions) and should NOT be treated as leakage.
    text = "The company was struck off the register in 2019."
    assert strip_sg_leakage(text) == text


def test_strip_sg_leakage_removes_dismiss_and_allow():
    text = "I have considered the submissions. I therefore dismiss the application. No order as to costs."
    result = strip_sg_leakage(text)
    assert "dismiss" not in result.lower()
    assert "considered the submissions" in result.lower()


# --- extract_outcome_heuristic ---


def test_extract_outcome_dismissed():
    text = "Some analysis. " * 50 + "For these reasons, I therefore dismiss the application."
    assert extract_outcome_heuristic(text) == "dismissed"


def test_extract_outcome_allowed_full():
    text = "Some analysis. " * 50 + "For these reasons, I allow the appeal."
    assert extract_outcome_heuristic(text) == "allowed_full"


def test_extract_outcome_allowed_part():
    text = "Some analysis. " * 50 + "I allow the application in part."
    assert extract_outcome_heuristic(text) == "allowed_part"


def test_extract_outcome_struck_out():
    text = "Some analysis. " * 50 + "Accordingly, the statement of claim is struck out."
    assert extract_outcome_heuristic(text) == "struck_out"


def test_extract_outcome_withdrawn():
    text = "Some analysis. " * 50 + "I allowed the claimant to withdraw the application."
    assert extract_outcome_heuristic(text) == "withdrawn"


def test_extract_outcome_withdrawn_excludes_financial_withdrawal():
    # A bare "withdrawn" also matches a financial withdrawal ("the sum
    # withdrawn from the account"), not just a case being withdrawn -- this
    # surfaced as a real bug when the extraction window was widened past
    # 3,000 chars (docs/MODEL.md 2026-09-14 Changelog). No other disposition
    # language is present, so the correct result is None, not "withdrawn".
    text = (
        "Some analysis. " * 50
        + "The monies were withdrawn from the account by the first defendant "
        + "without authorisation. I will hear the parties on costs at a later date."
    )
    assert extract_outcome_heuristic(text) is None


# --- extract_costs_amount_bucket (Level 1 structured field, docs/OUTCOME_PREDICTION.md) ---


def test_extract_costs_amount_bucket_none_when_no_costs_figure():
    text = "Some analysis. " * 50 + "I dismiss the application."
    assert extract_costs_amount_bucket(text) == "none"


def test_extract_costs_amount_bucket_under_10k():
    text = "Some analysis. " * 50 + "I awarded the plaintiff costs of $6,000 (all-in)."
    assert extract_costs_amount_bucket(text) == "under_10k"


def test_extract_costs_amount_bucket_10k_to_50k():
    text = "Some analysis. " * 50 + "Costs fixed at $20,000 inclusive of disbursements."
    assert extract_costs_amount_bucket(text) == "10k_to_50k"


def test_extract_costs_amount_bucket_over_50k():
    text = "Some analysis. " * 50 + "Costs of and incidental to the appeal fixed at $65,000."
    assert extract_costs_amount_bucket(text) == "over_50k"


def test_extract_costs_amount_bucket_uses_last_match_in_window():
    text = "Some analysis. " * 50 + "Costs of the earlier motion were $2,000. Costs of this application fixed at $30,000."
    assert extract_costs_amount_bucket(text) == "10k_to_50k"


def test_extract_outcome_none_when_no_disposition_language():
    text = "Some analysis. " * 50 + "I will hear the parties on costs at a later date."
    assert extract_outcome_heuristic(text) is None


def test_extract_outcome_empty_text_returns_none():
    assert extract_outcome_heuristic("") is None
    assert extract_outcome_heuristic(None) is None


def test_extract_outcome_last_match_wins():
    # Regression for the real 2026_SGHC_161 case found during spot-checking:
    # an earlier "struck out" mention (about a sub-finding) should not win
    # over a later, more authoritative "dismissed" disposition.
    text = (
        "The AR was correct to strike out the statement of claim in its entirety "
        "and to dismiss the action, for the reasons given. "
        "I have therefore dismissed the appeal with costs."
    )
    assert extract_outcome_heuristic(text) == "dismissed"


def test_extract_outcome_only_searches_within_window():
    # A disposition phrase outside the window should not be found.
    text = "I dismiss the application." + (" filler word" * 2000)
    assert extract_outcome_heuristic(text, window=100) is None


# --- clean_singapore_record ---


def _raw_sg_record(citation="[2026] SGHC 156", text_suffix="I therefore dismiss the application."):
    return {
        "case_id": "2026_SGHC_156",
        "case_name": "ABE ISAAC (PTE) LTD v ATTORNEY-GENERAL",
        "citation": citation,
        "decision_date": "2026-07-31",
        "case_number": "HC/OA 711/2025",
        "catchwords": ["[Administrative Law]"],
        "judgment_url": "https://www.elitigation.sg/gdviewer/s/2026_SGHC_156",
        "text": "Some background. " * 20 + text_suffix,
        "coram": "Chua Lee Ming J",
    }


def test_clean_singapore_record_maps_filed_date_and_outcome():
    result = clean_singapore_record(_raw_sg_record())
    assert result["filed_date"] == "2026-07-31"
    assert result["outcome"] == "dismissed"
    assert result["case_type"] == "unknown"
    assert result["represented"] == "unknown"


def test_clean_singapore_record_strips_disposition_language_from_text():
    result = clean_singapore_record(_raw_sg_record())
    assert "dismiss" not in result["text"].lower()


def test_clean_singapore_record_court_from_citation():
    assert clean_singapore_record(_raw_sg_record(citation="[2026] SGCA 32"))["court"] == "SGCA"
    assert clean_singapore_record(_raw_sg_record(citation="[2026] SGHCR 23"))["court"] == "SGHCR"
    assert clean_singapore_record(_raw_sg_record(citation="[2026] SGHC 156"))["court"] == "SGHC"
    assert clean_singapore_record(_raw_sg_record(citation="[2026] SGFC 105"))["court"] == "unknown"


def test_clean_singapore_record_outcome_none_when_unmatched():
    raw = _raw_sg_record(text_suffix="I will hear the parties on costs.")
    result = clean_singapore_record(raw)
    assert result["outcome"] is None
