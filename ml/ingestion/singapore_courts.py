"""Pull raw bankruptcy/insolvency judgments from the Singapore Courts' public
judgments portal (elitigation.sg).

CourtListener (see `courtlistener.py`) has no Singapore coverage at all --
its `courts` list is US federal/state/tribal/territory/military plus three
UK courts. This module targets elitigation.sg instead, which is Singapore's
official portal for Supreme Court judgments and publishes full judgment text
free, with no login or token required.

There is no documented public API. Everything here is HTML scraping of a
server-rendered results page and judgment page, reverse-engineered by
inspecting requests against https://www.elitigation.sg/gd/:
  - Search: GET https://www.elitigation.sg/gdviewer/Home/Index
    key params: SearchPhrase (supports AND/OR/NOT/"phrase"/CatchWords:"..."),
    Filter (court; only "SUPCT" -- Supreme Court -- was found during
    exploration), YearOfDecision, CurrentPage (10 results/page).
  - Judgment text: GET https://www.elitigation.sg/gdviewer/s/{case_id}
    full text lives in <div id="divJudgement">.

No robots.txt was found (404) and no scraping-specific terms of use were
located at the time this was written -- re-check before running large or
repeated pulls, and keep request volume polite (see `_SLEEP_SECONDS`).

CRITICAL: like `courtlistener.py`, the full judgment text here is written
*after* the decision and states the outcome directly (e.g. "the application
is dismissed"). Do NOT use it as a model feature as-is -- route it through
the same leakage-stripping step in `ml/preprocessing/clean.py`, or use only
genuinely pre-decision documents as features.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

SEARCH_URL = "https://www.elitigation.sg/gdviewer/Home/Index"
JUDGMENT_URL = "https://www.elitigation.sg/gdviewer/s/{case_id}"

ML_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = ML_DIR.parent / "data" / "raw"

_HEADERS = {"User-Agent": "legal-prediction-research-bot/0.1 (educational dataset build)"}
_SLEEP_SECONDS = 1.0


def _search_page(query: str, court: str, page: int) -> BeautifulSoup:
    params = {
        "SearchPhrase": query,
        "Filter": court,
        "YearOfDecision": "All",
        "SortBy": "DateOfDecision",
        "CurrentPage": page,
        "SortAscending": "False",
        "PageSize": 0,
        "Verbose": "False",
        "SearchQueryTime": 0,
        "SearchTotalHits": 0,
        "SearchMode": "True",
        "SpanMultiplePages": "False",
    }
    resp = requests.get(SEARCH_URL, headers=_HEADERS, params=params, timeout=30)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser")


def _extract_date(date_link) -> str | None:
    if date_link is None:
        return None
    m = re.search(r'DecisionDate:"([\d-]+)"', date_link.get("data-searchparam", ""))
    return m.group(1) if m else None


def _parse_result_cards(soup: BeautifulSoup) -> list[dict]:
    cards = []
    for link in soup.select("a.gd-heardertext"):
        href = link.get("href", "")
        match = re.search(r"/gdviewer/s/(.+)$", href)
        if not match:
            continue
        card_body = link.find_parent("div", class_="gd-card-body")
        if card_body is None:
            continue
        citation = card_body.select_one("a.citation-num-link")
        date_link = card_body.select_one("a.decision-date-link")
        case_num = card_body.select_one("a.case-num-link")
        catchword_scope = card_body.parent or card_body
        catchwords = [cw.get_text(strip=True) for cw in catchword_scope.select("a.gd-cw")]
        cards.append(
            {
                "case_id": match.group(1),
                "case_name": link.get_text(strip=True),
                "citation": citation.get_text(strip=True).rstrip("|").strip() if citation else None,
                "decision_date": _extract_date(date_link),
                "case_number": case_num.get_text(strip=True) if case_num else None,
                "catchwords": catchwords,
                "judgment_url": f"https://www.elitigation.sg{href}",
            }
        )
    return cards


def search_cases(query: str, court: str = "SUPCT", max_results: int = 1000) -> list[dict]:
    """Search judgments by keyword. Returns lightweight result-card dicts
    (case name, citation, date, catchwords, case id) -- no full judgment
    text yet. Call `fetch_judgment()` per case id for that.

    `query` supports the portal's search syntax: AND/OR/NOT, "exact phrase",
    and `CatchWords:"..."` to filter by a specific subject tag (e.g.
    `CatchWords:"Insolvency Law"`).
    """
    results: list[dict] = []
    page = 1
    while len(results) < max_results:
        soup = _search_page(query, court, page)
        cards = _parse_result_cards(soup)
        if not cards:
            break
        results.extend(cards)
        page += 1
        time.sleep(_SLEEP_SECONDS)
    return results[:max_results]


def fetch_judgment(case_id: str) -> dict:
    """Fetch one judgment's full text by its case id (e.g. "2026_SGHC_156",
    as returned in `search_cases()` results).
    """
    url = JUDGMENT_URL.format(case_id=case_id)
    resp = requests.get(url, headers=_HEADERS, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    body = soup.select_one("#divJudgement")
    coram = soup.select_one(".HN-Coram")
    return {
        "case_id": case_id,
        "text": body.get_text(" ", strip=True) if body else "",
        "coram": coram.get_text(" ", strip=True) if coram else None,
    }


def fetch_bankruptcy_dataset(max_results: int = 1000, court: str = "SUPCT") -> list[dict]:
    """Pull up to `max_results` bankruptcy/insolvency judgments: search, then
    fetch full text for each hit. Writes incrementally, one JSON object per
    line, to `data/raw/singapore_bankruptcy_cases.jsonl` so a crash partway
    through doesn't lose completed records.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DIR / "singapore_bankruptcy_cases.jsonl"

    cards = search_cases('"bankruptcy" OR "insolvency"', court=court, max_results=max_results)
    print(f"Found {len(cards)} candidate case(s); fetching full text...")

    records: list[dict] = []
    with out_path.open("w", encoding="utf-8") as f:
        for i, card in enumerate(cards, start=1):
            judgment = fetch_judgment(card["case_id"])
            record = {**card, **judgment}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            records.append(record)
            if i % 25 == 0 or i == len(cards):
                print(f"  {i}/{len(cards)} fetched")
            time.sleep(_SLEEP_SECONDS)

    print(f"Wrote {len(records)} record(s) to {out_path}")
    return records


if __name__ == "__main__":
    fetch_bankruptcy_dataset(max_results=1000)
