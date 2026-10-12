"""
test_peer_analysis.py
---------------------
Tests cross-company co-occurrence mining, network graph generation, and plain insights.
"""
from __future__ import annotations

import pytest

from peer_analysis import (
    build_peer_linkage_network,
    extract_statement_items,
    get_cooccurrence_engine,
)


def test_concept_extraction():
    sent = "We are deploying generative ai tools for cloud migration across our enterprise clients."
    items = extract_statement_items(sent)
    assert "GenAI & Agents" in items
    assert "Cloud & Data Estate" in items


def test_cooccurrence_rules():
    engine = get_cooccurrence_engine()
    rules = engine.get_rules()
    assert not rules.empty, "Expected co-occurrence rules across company calls"
    assert "Relationship Affinity" in rules.columns
    assert "Support" in rules.columns
    assert "Concept A" in rules.columns


def test_plain_language_insights():
    engine = get_cooccurrence_engine()
    insights = engine.get_plain_insights()
    assert len(insights) > 0
    assert any("When" in ins and "%" in ins for ins in insights)


def test_network_graph_creation():
    engine = get_cooccurrence_engine()
    rules = engine.get_rules()
    fig = build_peer_linkage_network(rules)
    assert fig is not None
    assert len(fig.data) >= 2  # Edge trace and Node trace
