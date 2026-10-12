"""
test_topic_model.py
-------------------
Tests supervised statement classification, stance detection, and reliability reporting.
"""
from __future__ import annotations

import pytest

from topic_model import (
    TOPIC_CATEGORIES,
    get_model_evaluation,
    get_topic_engine,
)


def test_topic_engine_initialization():
    engine = get_topic_engine()
    assert engine.model is not None
    assert len(engine.evaluation_report) > 0


def test_statement_classification():
    engine = get_topic_engine()

    # Margin statement
    s_margin = "Operating margins expanded by eighty basis points due to operational efficiencies."
    res_m = engine.classify_statement(s_margin)
    assert res_m["topic"] == "Margins & Costs"
    assert res_m["confidence"] > 0.1

    # AI statement
    s_ai = "We now have over four hundred active generative AI enterprise projects in production."
    res_ai = engine.classify_statement(s_ai)
    assert res_ai["topic"] == "AI & Digital"

    # Guidance statement
    s_guidance = "We are updating our constant currency revenue growth guidance to 3.5 to 5.0 percent."
    res_g = engine.classify_statement(s_guidance)
    assert res_g["topic"] == "Guidance & Outlook"


def test_stance_classification():
    engine = get_topic_engine()

    s_cautious = "We remain cautious regarding discretionary budget cuts and persistent macroeconomic headwinds."
    res_c = engine.classify_statement(s_cautious)
    assert res_c["stance"] == "Cautious"
    assert res_c["risk_severity"] in ("Medium", "High")

    s_confident = "We delivered robust, industry-leading performance and accelerated deal momentum."
    res_conf = engine.classify_statement(s_confident)
    assert res_conf["stance"] == "Confident"


def test_evaluation_metrics_structure():
    ev = get_model_evaluation()
    assert "test_accuracy" in ev
    assert ev["test_accuracy"] > 0.4
    assert len(ev["confusion_matrix"]) == len(TOPIC_CATEGORIES)
    assert len(ev["confusion_matrix"][0]) == len(TOPIC_CATEGORIES)
