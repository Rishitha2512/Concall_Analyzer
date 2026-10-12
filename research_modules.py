"""
research_modules.py
--------------------
Compatibility bridge exposing modern topic classification, multilingual
profiling, entity extraction, question answering, and co-occurrence mining.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from entity_extractor import extract_all_entities
from multilingual import analyze_script_profile
from peer_analysis import get_cooccurrence_engine
from qa_engine import get_qa_engine
from topic_model import TOPIC_CATEGORIES, compute_topic_distribution, get_topic_engine


def theme_matrix(transcripts: List[Dict[str, Any]]) -> pd.DataFrame:
    """Calculates topic distribution matrix across quarters for heatmap and radar."""
    df = compute_topic_distribution(transcripts)
    if df.empty:
        return pd.DataFrame()
    pivot = df.pivot(index="Quarter", columns="Topic", values="Share (%)").fillna(0.0)
    return pivot


def classify_statements(text: str, max_rows: int = 40) -> pd.DataFrame:
    """Classifies statements using the supervised seed-trained classifier."""
    engine = get_topic_engine()
    from data_loader import clean_transcript_text
    import re
    cleaned = clean_transcript_text(text)
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", cleaned) if len(s.split()) >= 7]
    rows = []
    for s in sents[:max_rows]:
        meta = engine.classify_statement(s)
        rows.append({
            "Statement": s,
            "Category": meta["topic"],
            "Model Confidence": meta["confidence"],
            "Stance": meta["stance"],
            "Risk Severity": meta["risk_severity"],
        })
    return pd.DataFrame(rows)


def multilingual_profile(text: str) -> Dict[str, Any]:
    prof = analyze_script_profile(text)
    return {
        "detected_script": prof["primary_script"],
        "script_counts": prof["script_counts"],
        "tokens": prof["word_count"],
        "unique_tokens": prof["unique_words"],
        "sentences": max(1, prof["word_count"] // 15),
    }


def extract_entities(text: str) -> pd.DataFrame:
    return extract_all_entities(text)


def chatbot_answer(text: str, question: str, top_k: int = 5) -> Dict[str, Any]:
    engine = get_qa_engine([{"company": "Current Company", "quarter": "Analyzed Quarter", "sentences": [{"text": s} for s in text.split(". ") if len(s.split()) > 5]}])
    res = engine.answer_query(question)
    return {
        "answer": res["answer"],
        "evidence": res["evidence"],
    }


def market_basket(quarterly_analyses: List[Dict[str, Any]], transcripts: Optional[List[Dict[str, Any]]] = None) -> pd.DataFrame:
    engine = get_cooccurrence_engine()
    return engine.get_rules()
