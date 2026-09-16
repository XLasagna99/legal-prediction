"""Shared disposition-label metadata.

Single source of truth for the 5 outcome categories' human-readable
meaning -- previously duplicated near-verbatim in ml/agent/case_agent.py
and ml/training/llm_classify.py.
"""
from __future__ import annotations

DISPOSITION_MEANINGS = {
    "dismissed": "the application/appeal/claim is refused on its merits",
    "allowed_full": "the application/appeal/claim is allowed or granted in full",
    "allowed_part": "the application/appeal/claim is allowed or granted in part only",
    "struck_out": "the application/pleading is struck out -- a procedural disposal, "
    "not a ruling on the merits",
    "withdrawn": "the moving party withdrew before any ruling on the merits -- "
    "neither side wins or loses on the substance",
}
