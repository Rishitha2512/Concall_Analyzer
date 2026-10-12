"""
test_qa_engine.py
-----------------
Tests question answering with citations, quarter filtering, and comparison queries.
"""
from __future__ import annotations

import pytest

from qa_engine import RetrievalQAEngine


def test_retrieval_and_citations():
    engine = RetrievalQAEngine()
    transcripts = [
        {
            "company": "TCS",
            "quarter": "Q1 FY26",
            "sentences": [
                {
                    "text": "Our Q1 operating margins stood at 24.5%, driven by productivity improvements.",
                    "speaker": "K Krithivasan",
                    "speaker_role": "Management",
                    "section": "Prepared Remarks",
                },
                {
                    "text": "We signed $9.4 billion in total contract value across North America.",
                    "speaker": "Samir Seksaria",
                    "speaker_role": "Management",
                    "section": "Prepared Remarks",
                },
            ],
        }
    ]
    engine.build_index(transcripts)

    res = engine.answer_query("What was the operating margin in Q1 FY26?")
    assert "24.5%" in res["answer"]
    assert len(res["citations"]) > 0
    cit = res["citations"][0]
    assert cit["speaker"] == "K Krithivasan"
    assert cit["quarter"] == "Q1 FY26"
    assert cit["section"] == "Prepared Remarks"


def test_comparison_intent_parsing():
    engine = RetrievalQAEngine()
    filters = engine._parse_query_filters("Compare TCS and Infosys on operating margin outlook")
    assert filters["is_comparison"] is True
    assert "TCS" in filters["compare_companies"]
    assert "Infosys" in filters["compare_companies"]
