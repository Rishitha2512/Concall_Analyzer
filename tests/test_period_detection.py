"""
test_period_detection.py
------------------------
Tests period detection, deduplication, stub flagging, and chronological sorting.
"""
from __future__ import annotations

from pathlib import Path
import pytest

from data_loader import (
    clean_transcript_text,
    detect_fiscal_period,
    detect_call_date,
    load_company_data,
    inspect_file_integrity,
)


def test_tcs_deduplication():
    """Verify that TCS Q5.pdf is identified as duplicate of Q4.pdf."""
    res = load_company_data("TCS", use_cache=True)
    dups = res["duplicates"]
    assert len(dups) >= 1, "Expected TCS duplicate file detection"
    assert any("Q5" in d["filename"] and "Q4" in d["duplicate_of"] for d in dups)


def test_infosys_chronological_order():
    """Verify that Infosys Q4 FY25 precedes Q1 FY26 chronologically."""
    res = load_company_data("Infosys", use_cache=True)
    quarters = res["loaded_quarters"]
    assert "Q4 FY25" in quarters[0] or "Q4 FY25" in quarters[1]
    # Q4 FY25 must appear before Q1 FY26
    idx_q4 = next(i for i, q in enumerate(quarters) if "Q4 FY25" in q)
    idx_q1 = next(i for i, q in enumerate(quarters) if "Q1 FY26" in q)
    assert idx_q4 < idx_q1, f"Q4 FY25 ({idx_q4}) must precede Q1 FY26 ({idx_q1})"


def test_infoedge_q5_is_q1_fy25():
    """Verify that infoedge Q5.pdf is identified as Q1 FY25 and placed first."""
    res = load_company_data("infoedge", use_cache=True)
    quarters = res["loaded_quarters"]
    assert quarters[0] == "Q1 FY25", f"Expected Q1 FY25 to be first quarter, got {quarters[0]}"


def test_ltimindtree_stub_flagging():
    """Verify that LTIMindtree Q1.pdf and Q3.pdf are flagged as stubs."""
    res = load_company_data("LTIMindtree", use_cache=True)
    flagged = res["flagged_files"]
    assert len(flagged) >= 2, "Expected 2 flagged files for LTIMindtree"
    flagged_names = [f["filename"] for f in flagged]
    assert "Q1.pdf" in flagged_names
    assert "Q3.pdf" in flagged_names


def test_ligature_normalization():
    """Verify ligature cleaning converts ?scal to fiscal and preserves words."""
    raw = "third quarter of \ufb01scal year FY2026 \ufb01nancial results"
    cleaned = clean_transcript_text(raw)
    assert "fiscal" in cleaned
    assert "financial" in cleaned
