"""Pull raw cases from CourtListener (Free Law Project).

This is a starting point, not a finished ingester. You need a (free) API token
from CourtListener and you must respect their rate limits and terms of use.

CRITICAL: the opinion `plain_text` returned by the API is written *after* the
decision and reveals the outcome. Do NOT use it as a model feature. Use it only
to derive labels, and pull pre-decision text (complaints, briefs, dockets)
separately. Keeping label-source text out of feature-source text is the single
most important thing this layer does.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import requests

API_BASE = "https://www.courtlistener.com/api/rest/v4"
ML_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = ML_DIR.parent / "data" / "raw"


def _headers() -> dict:
    token = os.environ.get("COURTLISTENER_API_TOKEN")
    if not token:
        raise SystemExit("Set COURTLISTENER_API_TOKEN in your environment.")
    return {"Authorization": f"Token {token}"}


def fetch_opinions(court: str, page_size: int = 50, max_pages: int = 1) -> list[dict]:
    """Fetch opinion records for a court. Returns a list of raw JSON dicts.

    Extend the params (date filters, docket joins) to match your task. Add
    polite delays and handle pagination + retries before running at scale.
    """
    results: list[dict] = []
    url = f"{API_BASE}/opinions/"
    # Opinion has no `court` field of its own — court lives on the docket,
    # reached through the opinion's cluster.
    params = {"cluster__docket__court": court, "page_size": page_size}
    for _ in range(max_pages):
        resp = requests.get(url, headers=_headers(), params=params, timeout=30)
        resp.raise_for_status()
        payload = resp.json()
        results.extend(payload.get("results", []))
        url = payload.get("next")
        params = {}  # `next` already encodes them
        if not url:
            break
        time.sleep(1.0)  # be a good API citizen
    return results


if __name__ == "__main__":
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print("Configure court IDs and date ranges before running a real pull.")
    print("See https://www.courtlistener.com/help/api/rest/ for the schema.")
