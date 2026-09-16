"""Tests for the pure HTML-parsing helpers in ml/ingestion/singapore_courts.py.

These use a static fixture modeled on the real page structure (captured
2026-07 while reverse-engineering elitigation.sg's search results page,
see docs/MODEL.md §2) so they run offline and can't hit the live site.
`search_cases()`/`fetch_judgment()` themselves make real HTTP requests and
are intentionally not unit tested here -- that would need network mocking,
and their pure-parsing logic (`_parse_result_cards`, `_extract_date`) is
what's actually worth guarding against a refactor. Note these tests can't
catch the live site changing its HTML structure -- only a real fetch would.
"""
from bs4 import BeautifulSoup

from ingestion.singapore_courts import _extract_date, _parse_result_cards

SEARCH_RESULT_FIXTURE = """
<div class="row">
<div class="card col-12">
    <div class="card-body gd p-2">
        <div class="row align-items-center justify-content-between">
            <div class="col-xs-12 col-sm-12 col-md-12 col-lg-8 col-xl-8">
                <div class="align-items-center">
                    <div class="gd-catchword-container">
                        <a class="gd-cw" data-searchterm='CatchWords:"Insolvency Law"' href="#">
                            [Insolvency Law]
                        </a>
                    </div>
                    <div class="gd-card-body">
                        <a target="_blank" data-toggle="tooltip" href='/gdviewer/s/2026_SGHC_156'
                           class="h5 gd-heardertext">
                            ABE ISAAC (PTE) LTD v ATTORNEY-GENERAL
                        </a>
                        <br />
                        <a href="#" class="citation-num-link" data-searchparam='"[2026] SGHC 156"'>
                            <span class="gd-addinfo-text">[2026] SGHC 156 |</span>
                        </a>
                        <a class="decision-date-link" data-searchparam='DecisionDate:"2026-07-31"' href="#">
                            <span class="gd-addinfo-text"> Decision Date: 31 Jul 2026 | </span>
                        </a>
                        <a class="case-num-link" data-searchparam='"HC/OA 711/2025 ( HC/SUM 619/2026 ) "' href="#">
                            <span class="gd-addinfo-text"> HC/OA 711/2025 ( HC/SUM 619/2026 )  </span>
                        </a>
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>
</div>
"""


def test_parse_result_cards_extracts_case_id():
    soup = BeautifulSoup(SEARCH_RESULT_FIXTURE, "html.parser")
    cards = _parse_result_cards(soup)
    assert len(cards) == 1
    assert cards[0]["case_id"] == "2026_SGHC_156"


def test_parse_result_cards_extracts_case_name():
    soup = BeautifulSoup(SEARCH_RESULT_FIXTURE, "html.parser")
    cards = _parse_result_cards(soup)
    assert cards[0]["case_name"] == "ABE ISAAC (PTE) LTD v ATTORNEY-GENERAL"


def test_parse_result_cards_extracts_citation_without_trailing_pipe():
    soup = BeautifulSoup(SEARCH_RESULT_FIXTURE, "html.parser")
    cards = _parse_result_cards(soup)
    assert cards[0]["citation"] == "[2026] SGHC 156"


def test_parse_result_cards_extracts_decision_date():
    soup = BeautifulSoup(SEARCH_RESULT_FIXTURE, "html.parser")
    cards = _parse_result_cards(soup)
    assert cards[0]["decision_date"] == "2026-07-31"


def test_parse_result_cards_extracts_case_number():
    soup = BeautifulSoup(SEARCH_RESULT_FIXTURE, "html.parser")
    cards = _parse_result_cards(soup)
    assert "HC/OA 711/2025" in cards[0]["case_number"]


def test_parse_result_cards_extracts_catchwords():
    soup = BeautifulSoup(SEARCH_RESULT_FIXTURE, "html.parser")
    cards = _parse_result_cards(soup)
    assert cards[0]["catchwords"] == ['[Insolvency Law]']


def test_parse_result_cards_builds_full_judgment_url():
    soup = BeautifulSoup(SEARCH_RESULT_FIXTURE, "html.parser")
    cards = _parse_result_cards(soup)
    assert cards[0]["judgment_url"] == "https://www.elitigation.sg/gdviewer/s/2026_SGHC_156"


def test_parse_result_cards_returns_empty_list_for_no_matches():
    soup = BeautifulSoup("<div>no results</div>", "html.parser")
    assert _parse_result_cards(soup) == []


def test_extract_date_returns_none_for_missing_link():
    assert _extract_date(None) is None


def test_extract_date_returns_none_when_pattern_absent():
    soup = BeautifulSoup('<a data-searchparam="something else"></a>', "html.parser")
    assert _extract_date(soup.find("a")) is None
