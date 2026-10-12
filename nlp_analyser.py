"""
nlp_analyser.py
---------------
Core analysis engine coordinating sentiment scoring, key narrative phrases,
topic classification, forward commitments, and quarterly summaries.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from entity_extractor import build_quarter_scorecard
from topic_model import get_topic_engine

CALL_STOP_WORDS = list(ENGLISH_STOP_WORDS | {
    "quarter", "year", "question", "sort", "think", "said", "thank", "thanks",
    "like", "going", "know", "just", "lot", "want", "let", "need", "kind",
    "yes", "next", "good", "okay", "also", "sir", "mr", "ms", "operator",
})


def _extract_sentences(text: str) -> List[str]:
    cleaned = re.sub(r"\s+", " ", text).strip()
    return [
        s.strip()
        for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", cleaned)
        if len(s.split()) >= 8
    ]


def extract_key_phrases(text: str, top_n: int = 10) -> List[str]:
    sents = _extract_sentences(text)
    if len(sents) < 2:
        return []
    vec = TfidfVectorizer(stop_words=CALL_STOP_WORDS, ngram_range=(1, 2), max_features=300, sublinear_tf=True)
    mat = vec.fit_transform(sents)
    scores = mat.sum(axis=0).A1
    feats = vec.get_feature_names_out()
    return [p for p, _ in sorted(zip(feats, scores), key=lambda x: x[1], reverse=True)[:top_n]]


def generate_summary(text: str, top_n: int = 5) -> str:
    sents = [s for s in _extract_sentences(text) if "operator" not in s.lower() and "conference call" not in s.lower()]
    if len(sents) <= top_n:
        return " ".join(sents)
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
    mat = vec.fit_transform(sents)
    scores = mat.sum(axis=1).A1
    ranked = np.argsort(scores)[::-1][:top_n]
    return " ".join(sents[i] for i in sorted(ranked))


def analyse_tone(text: str) -> Dict[str, Any]:
    """Computes mean sentence-level VADER compound score."""
    analyzer = SentimentIntensityAnalyzer()
    sents = _extract_sentences(text)
    score = float(np.mean([analyzer.polarity_scores(x)["compound"] for x in sents])) if sents else 0.0
    overall = "Positive" if score > 0.15 else "Negative" if score < -0.15 else "Neutral"
    return {"overall": overall, "score": round(score, 3)}


def extract_risks(sentences: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    out = []
    for s in sentences:
        if s.get("stance") == "Cautious" or s.get("risk_severity") in ("High", "Medium"):
            out.append({"risk": s["text"], "speaker": s.get("speaker", "Management")})
    return out[:12]


def extract_commitments(sentences: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    future_pat = re.compile(r"\b(will|shall|plan to|expect to|intend to|aim to|target|committed to|looking to)\b", re.I)
    out = []
    for s in sentences:
        if future_pat.search(s["text"]):
            out.append({"commitment": s["text"], "speaker": s.get("speaker", "Management")})
    return out[:12]


def analyse_quarter_dict(transcript: Dict[str, Any]) -> Dict[str, Any]:
    text = transcript.get("text", "")
    sentences = transcript.get("sentences", [])
    quarter = transcript.get("quarter", "")

    summary_text = generate_summary(text)
    tone_res = analyse_tone(text)
    scorecard = build_quarter_scorecard(transcript)
    key_phrases = extract_key_phrases(text, top_n=6)
    risks = extract_risks(sentences)
    commitments = extract_commitments(sentences)

    return {
        "quarter": quarter,
        "call_date": transcript.get("call_date"),
        "filename": transcript.get("filename"),
        "summary": summary_text,
        "management_tone": tone_res,
        "key_metrics_mentioned": scorecard.get("metrics", {}),
        "earnings_triggers": [{"trigger": p} for p in key_phrases],
        "business_changes": [{"description": p} for p in key_phrases],
        "risks_and_issues": risks,
        "commitments_made": commitments,
        "deviations": [],
        "sentences": sentences,
    }


def run_full_analysis(
    transcripts: List[Dict[str, Any]],
    company_name: str = "",
    verbose: bool = False,
) -> Dict[str, Any]:
    """Runs full NLP analysis pipeline across all transcripts."""
    engine = get_topic_engine()
    classified_transcripts = engine.classify_transcripts(transcripts)

    quarterly_analyses = [analyse_quarter_dict(t) for t in classified_transcripts]

    # Calculate narrative shifts / deviations
    for i in range(1, len(quarterly_analyses)):
        prev_comm = " ".join(c["commitment"] for c in quarterly_analyses[i - 1]["commitments_made"])
        shifts = []
        for c in quarterly_analyses[i]["commitments_made"]:
            if c["commitment"] not in prev_comm:
                shifts.append(c["commitment"])
        quarterly_analyses[i]["deviations"] = shifts[:4]

    return {
        "company_name": company_name,
        "quarterly_analyses": quarterly_analyses,
        "transcripts": classified_transcripts,
    }
