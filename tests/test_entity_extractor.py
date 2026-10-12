"""
test_entity_extractor.py
------------------------
Tests named entity extraction and Key Numbers Scorecard generation.
"""
from __future__ import annotations

import pandas as pd
import pytest

from entity_extractor import (
    build_qoq_change_table,
    build_quarter_scorecard,
    extract_all_entities,
)


def test_entity_extraction_patterns():
    sample = (
        "In Q1 FY26, we achieved revenue growth of 3.5% and operating margins of 24.2%. "
        "Total contract value was $9.4 billion, signed with Microsoft and AWS across North America."
    )
    df = extract_all_entities(sample, quarter="Q1 FY26", company="TCS")
    assert not df.empty
    types = df["Type"].unique().tolist()
    assert "Percentage" in types
    assert "Money" in types
    assert "Fiscal Period" in types
    assert "Organization / Ecosystem" in types or "Geography" in types


def test_quarter_scorecard_metrics():
    transcript = {
        "company": "TCS",
        "quarter": "Q1 FY26",
        "sentences": [
            {"text": "Our operating margin stood at 24.5%, reflecting disciplined execution.", "speaker": "CEO", "section": "Prepared Remarks"},
            {"text": "Constant currency revenue growth was 3.1% year-on-year.", "speaker": "CFO", "section": "Prepared Remarks"},
            {"text": "Total contract value for the quarter closed at $4.4 billion.", "speaker": "CEO", "section": "Prepared Remarks"},
            {"text": "Voluntary attrition reduced further to 12.3% on an annualized basis.", "speaker": "CHRO", "section": "Prepared Remarks"},
        ],
    }
    sc = build_quarter_scorecard(transcript)
    metrics = sc["metrics"]
    assert metrics["Operating Margin"]["value"] == "24.5%"
    assert metrics["Revenue Growth"]["value"] == "3.1%"
    assert metrics["TCV / Order Book"]["value"] == "$4.4 billion"
    assert metrics["Attrition"]["value"] == "12.3%"


def test_qoq_change_table_calculation():
    sc_q1 = {
        "quarter": "Q1 FY26",
        "company": "TCS",
        "metrics": {
            "Operating Margin": {"value": "24.0%", "raw_num": 24.0, "sentence": "Margin was 24.0%", "speaker": "CFO"},
        },
    }
    sc_q2 = {
        "quarter": "Q2 FY26",
        "company": "TCS",
        "metrics": {
            "Operating Margin": {"value": "24.5%", "raw_num": 24.5, "sentence": "Margin expanded to 24.5%", "speaker": "CFO"},
        },
    }
    df_qoq = build_qoq_change_table([sc_q1, sc_q2])
    assert not df_qoq.empty
    q2_margin = df_qoq[(df_qoq["Quarter"] == "Q2 FY26") & (df_qoq["Metric"] == "Operating Margin")]
    assert not q2_margin.empty
    assert "+0.5%" in q2_margin.iloc[0]["QoQ Shift"]
