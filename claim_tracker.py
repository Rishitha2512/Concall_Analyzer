"""
claim_tracker.py
----------------
Core engine for claim extraction, cross-quarter evolution tracking,
and narrative stress quantification across corporate earnings calls.

Powers the signature 3D Claim Time Machine and Narrative Stress Map.
Zero internal AI/ML jargon: exposed strictly through institutional financial terms
(e.g., "Reaffirmed Target", "Weakened Stance", "Contradiction Candidate", "Newly Introduced").
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Color Codes & Status Definitions
# -----------------------------------------------------------------------------
STATUS_COLORS = {
    "new": "#6aa8ff",                    # Blue: Newly introduced claim
    "reaffirmed": "#5ee6a0",             # Green: Supported / reaffirmed as continuing
    "weakened": "#f6b95b",               # Amber: Weakened, qualified, or less certain
    "contradiction_candidate": "#ff7a8a" # Red: Possible contradiction requiring investigation
}

STATUS_LABELS = {
    "new": "Newly Introduced",
    "reaffirmed": "Reaffirmed Target",
    "weakened": "Weakened / Qualified",
    "contradiction_candidate": "Contradiction Candidate",
}

# Standardized corporate business topics
TOPIC_NAMES = [
    "Demand and Pipeline",
    "Margins and Costs",
    "AI and Digital",
    "Talent and Attrition",
    "Guidance and Outlook",
    "Capital Allocation",
    "Risk and Macro",
    "Client and Deals",
]

TOPIC_TO_INDEX = {name: i for i, name in enumerate(TOPIC_NAMES)}

# Forward-looking and commitment markers
CLAIM_INDICATORS = [
    r"\bexpect\b", r"\baim\b", r"\btarget\b", r"\bguidance\b", r"\bcommitted\b",
    r"\banticipate\b", r"\bconfident\b", r"\bwill continue\b", r"\bremain on track\b",
    r"\bprojected\b", r"\boutlook\b", r"\bdeal pipeline\b", r"\border book\b",
    r"\bgrowth of\b", r"\bmargin between\b", r"\bdouble-digit\b", r"\bsee strong\b",
    r"\bheadwinds\b", r"\buncertainty\b", r"\bcautious\b", r"\bchallenging\b",
]

CLAIM_REGEX = re.compile("|".join(CLAIM_INDICATORS), re.I)

# Hedging & weakening indicators
HEDGING_TERMS = [
    "subject to", "macro uncertainty", "headwinds", "challenging", "volatility",
    "cautious", "delay", "extended decision cycles", "softness", "slowness",
    "moderation", "recalibrat", "lower than expected", "pushback", "scrutiny",
]

# Contradiction / reversal markers
REVERSAL_TERMS = [
    "revised down", "lowered guidance", "cut guidance", "below expectation",
    "cancelled", "withdrawn", "deteriorat", "decline from earlier", "missed target",
    "reversal", "unfavourable variance", "margin compression", "downward revision",
]


# -----------------------------------------------------------------------------
# Claim Extraction
# -----------------------------------------------------------------------------
def extract_claims_from_transcript(transcript: Dict[str, Any], quarter_idx: int = 0) -> List[Dict[str, Any]]:
    """
    Extracts substantive forward-looking commitments, financial targets,
    and strategic assertions made by management in an earnings call.
    """
    sentences = transcript.get("sentences", [])
    quarter = transcript.get("quarter", "Unknown Quarter")
    company = transcript.get("company", "Company")
    claims: List[Dict[str, Any]] = []

    seen_signatures: Set[str] = set()

    for i, s in enumerate(sentences):
        txt = s.get("text", "").strip()
        if len(txt) < 35 or len(txt) > 450:
            continue

        speaker = s.get("speaker", "Management")
        section = s.get("section", "Prepared Remarks")
        role = s.get("role", "Management")

        # Focus primarily on management assertions
        if role != "Management" and "Analyst" in speaker:
            continue

        # Check for claim markers or numerical commitments
        has_claim_marker = bool(CLAIM_REGEX.search(txt))
        has_number_commitment = bool(re.search(r"\b\d+(?:\.\d+)?\s*(?:%|billion|crore|million|bps)\b", txt, re.I))

        if not (has_claim_marker or has_number_commitment):
            continue

        # Topic attribution
        topic = s.get("topic", "Guidance and Outlook")
        if topic not in TOPIC_TO_INDEX:
            topic = "Guidance and Outlook"

        # Stance determination
        stance = s.get("stance", "Neutral")
        tone_score = float(s.get("tone_score", 0.0))

        # Check hedging and reversal signals
        low = txt.lower()
        hedge_count = sum(1 for term in HEDGING_TERMS if term in low)
        is_reversal = any(term in low for term in REVERSAL_TERMS)

        # Financial metric reference if present
        m_pct = re.search(r"(\d+(?:\.\d+)?\s*%)", txt)
        m_money = re.search(r"(?:₹|Rs\.?|\$|USD)\s*(\d+(?:[.,]\d+)?\s*(?:billion|million|crore|bn)?)", txt, re.I)
        metric_ref = ""
        if m_pct:
            metric_ref = m_pct.group(0)
        elif m_money:
            metric_ref = m_money.group(0)

        # Signature to prevent redundant extraction of near-identical sentences
        norm_words = " ".join(re.findall(r"\b[a-z]{4,}\b", low)[:6])
        if norm_words in seen_signatures:
            continue
        seen_signatures.add(norm_words)

        claim_id = f"CLM-{company}-{quarter}-{len(claims)+1}"

        claims.append({
            "claim_id": claim_id,
            "company": company,
            "quarter": quarter,
            "quarter_idx": quarter_idx,
            "speaker": speaker,
            "section": section,
            "topic": topic,
            "topic_idx": TOPIC_TO_INDEX[topic],
            "text": txt,
            "metric_ref": metric_ref,
            "stance": stance,
            "tone_score": tone_score,
            "hedge_count": hedge_count,
            "is_reversal": is_reversal,
            "status": "new",  # Default before cross-quarter resolution
            "status_reason": f"Initial corporate statement extracted in {quarter}.",
            "previous_claim_id": None,
            "previous_quarter": None,
            "previous_text": None,
            "similarity_score": 0.0,
        })

        if len(claims) >= 20:  # Top salient claims per quarter
            break

    return claims


# -----------------------------------------------------------------------------
# Cross-Quarter Semantic Matching & Status Classification
# -----------------------------------------------------------------------------
def _compute_claim_similarity(c1: Dict[str, Any], c2: Dict[str, Any]) -> float:
    """Computes lexical and semantic overlap between two claims."""
    # Topic bonus
    topic_bonus = 0.35 if c1["topic"] == c2["topic"] else 0.0

    # Token overlap
    tokens1 = set(re.findall(r"\b[a-z]{3,}\b", c1["text"].lower()))
    tokens2 = set(re.findall(r"\b[a-z]{3,}\b", c2["text"].lower()))

    # Filter out ubiquitous filler words
    stopwords = {"that", "this", "with", "have", "from", "will", "been", "were", "they", "their", "quarter", "company"}
    tokens1 -= stopwords
    tokens2 -= stopwords

    if not tokens1 or not tokens2:
        return topic_bonus

    jaccard = len(tokens1 & tokens2) / max(1, len(tokens1 | tokens2))

    # Metric match bonus
    metric_bonus = 0.25 if (c1["metric_ref"] and c2["metric_ref"] and c1["metric_ref"] == c2["metric_ref"]) else 0.0

    return (jaccard * 0.5) + topic_bonus + metric_bonus


def match_claims_across_quarters(quarter_claims_list: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    Traces claims chronologically across consecutive quarters.
    Classifies evolution status into:
    - 'new': Introduced for the first time
    - 'reaffirmed': Maintained or strengthened with consistent management stance
    - 'weakened': Qualified, hedged, delayed, or softened
    - 'contradiction_candidate': Material downgrade, reversal, or potential conflict requiring investigation
    """
    all_resolved_claims: List[Dict[str, Any]] = []

    for q_idx, current_claims in enumerate(quarter_claims_list):
        if q_idx == 0:
            # Baseline quarter: all claims are newly introduced
            for c in current_claims:
                c["status"] = "new"
                c["status_reason"] = f"Baseline target or disclosure established in {c['quarter']}."
                all_resolved_claims.append(c)
            continue

        previous_pool = [c for c in all_resolved_claims if c["quarter_idx"] < q_idx]

        for cur in current_claims:
            best_match: Optional[Dict[str, Any]] = None
            best_score = 0.0

            for prev in previous_pool:
                sim = _compute_claim_similarity(prev, cur)
                if sim > best_score:
                    best_score = sim
                    best_match = prev

            if best_match and best_score >= 0.42:
                cur["previous_claim_id"] = best_match["claim_id"]
                cur["previous_quarter"] = best_match["quarter"]
                cur["previous_text"] = best_match["text"]
                cur["similarity_score"] = round(best_score, 3)

                # Classify status based on stance shift, hedging, and reversals
                prev_tone = best_match.get("tone_score", 0.0)
                cur_tone = cur.get("tone_score", 0.0)
                is_reversal = cur.get("is_reversal", False)
                hedge_count = cur.get("hedge_count", 0)

                if is_reversal or (prev_tone > 0.35 and cur_tone < -0.15):
                    cur["status"] = "contradiction_candidate"
                    cur["status_reason"] = (
                        f"Potential divergence from earlier {best_match['quarter']} assertion. "
                        f"Language shows explicit directional reversal or guidance cut requiring investigation."
                    )
                elif hedge_count >= 2 or (prev_tone > 0.25 and cur_tone <= 0.05):
                    cur["status"] = "weakened"
                    cur["status_reason"] = (
                        f"Management qualified earlier {best_match['quarter']} assertion, introducing "
                        f"hedging vocabulary or extended operational timelines."
                    )
                else:
                    cur["status"] = "reaffirmed"
                    cur["status_reason"] = (
                        f"Management reaffirmed strategic trajectory established in {best_match['quarter']} "
                        f"with consistent stance and execution discipline."
                    )
            else:
                cur["status"] = "new"
                cur["status_reason"] = f"Newly introduced initiative or disclosure in {cur['quarter']}."

            all_resolved_claims.append(cur)

    return all_resolved_claims


# -----------------------------------------------------------------------------
# 3D Claim Time Machine Dataset Builder
# -----------------------------------------------------------------------------
def build_claim_time_machine_dataset(transcripts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Constructs the spatial node-link representation for the 3D Claim Time Machine.
    Nodes: Individual verified claims across quarters.
    Links: Directed narrative lineages between matching claims.
    """
    quarter_claims_list = [
        extract_claims_from_transcript(t, quarter_idx=i)
        for i, t in enumerate(transcripts)
    ]

    resolved_claims = match_claims_across_quarters(quarter_claims_list)

    # Compute 3D Coordinates
    # X: Quarter index (chronological axis)
    # Y: Topic index (thematic domain axis)
    # Z: Narrative elevation (stance & certainty axis)
    nodes = []
    for c in resolved_claims:
        q_x = float(c["quarter_idx"])
        t_y = float(c["topic_idx"])

        # Elevation based on status and tone
        base_z = 2.0 if c["status"] == "reaffirmed" else (
            1.2 if c["status"] == "new" else (
                0.2 if c["status"] == "weakened" else -1.2
            )
        )
        z_elevation = base_z + (c.get("tone_score", 0.0) * 0.8)

        nodes.append({
            **c,
            "x": q_x,
            "y": t_y,
            "z": round(z_elevation, 3),
            "color": STATUS_COLORS.get(c["status"], "#6aa8ff"),
            "status_label": STATUS_LABELS.get(c["status"], "New"),
        })

    # Construct cross-quarter links
    links = []
    node_map = {n["claim_id"]: n for n in nodes}

    for n in nodes:
        prev_id = n.get("previous_claim_id")
        if prev_id and prev_id in node_map:
            prev_node = node_map[prev_id]
            links.append({
                "source_id": prev_id,
                "target_id": n["claim_id"],
                "source_coord": (prev_node["x"], prev_node["y"], prev_node["z"]),
                "target_coord": (n["x"], n["y"], n["z"]),
                "status": n["status"],
                "color": n["color"],
                "label": f"{prev_node['quarter']} → {n['quarter']} ({n['status_label']})",
            })

    quarters = [t.get("quarter", f"Q{i+1}") for i, t in enumerate(transcripts)]

    return {
        "nodes": nodes,
        "links": links,
        "quarters": quarters,
        "topics": TOPIC_NAMES,
        "summary": {
            "total_claims": len(nodes),
            "reaffirmed": sum(1 for n in nodes if n["status"] == "reaffirmed"),
            "weakened": sum(1 for n in nodes if n["status"] == "weakened"),
            "contradiction_candidates": sum(1 for n in nodes if n["status"] == "contradiction_candidate"),
            "new": sum(1 for n in nodes if n["status"] == "new"),
        }
    }


# -----------------------------------------------------------------------------
# Narrative Stress Map Indicators
# -----------------------------------------------------------------------------
def compute_narrative_stress_indicators(transcripts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Computes interpretable risk and investigation-priority indicators:
    - Number of claims whose status changed (weakened or contradiction candidates).
    - Frequency of risk-related sentences.
    - Guidance revisions.
    - Hedging density.
    Produces 3D surface/scatter coordinates for the Narrative Stress Map.
    """
    tm_data = build_claim_time_machine_dataset(transcripts)
    nodes = tm_data["nodes"]
    quarters = tm_data["quarters"]

    records = []
    for q_idx, q_name in enumerate(quarters):
        q_nodes = [n for n in nodes if n["quarter_idx"] == q_idx]

        for t_idx, topic in enumerate(TOPIC_NAMES):
            t_nodes = [n for n in q_nodes if n["topic_idx"] == t_idx]

            n_weak = sum(1 for n in t_nodes if n["status"] == "weakened")
            n_contra = sum(1 for n in t_nodes if n["status"] == "contradiction_candidate")
            n_new = sum(1 for n in t_nodes if n["status"] == "new")
            n_reaffirm = sum(1 for n in t_nodes if n["status"] == "reaffirmed")

            # Interpretable Narrative Stress Score:
            # Weighted sum of contradiction signals, qualifications, and unresolved shifts
            stress_score = (n_contra * 3.5) + (n_weak * 2.0) + (n_new * 0.5) - (n_reaffirm * 0.4)
            stress_score = max(0.0, min(10.0, stress_score))

            priority = "Normal"
            color = "#42d3c5"
            if stress_score >= 5.5 or n_contra > 0:
                priority = "High Priority"
                color = "#ff7a8a"
            elif stress_score >= 2.5 or n_weak > 0:
                priority = "Medium Priority"
                color = "#f6b95b"

            records.append({
                "quarter": q_name,
                "quarter_idx": q_idx,
                "topic": topic,
                "topic_idx": t_idx,
                "stress_score": round(stress_score, 2),
                "weakened_count": n_weak,
                "contradiction_count": n_contra,
                "reaffirmed_count": n_reaffirm,
                "new_count": n_new,
                "priority": priority,
                "color": color,
                "explanation": (
                    f"In {q_name}, {topic} registered {n_contra} potential contradiction candidates "
                    f"and {n_weak} weakened claims. Overall stress index: {stress_score:.1f}/10."
                ),
            })

    df = pd.DataFrame(records)
    return {
        "dataframe": df,
        "quarters": quarters,
        "topics": TOPIC_NAMES,
        "high_priority_count": int(df[df["priority"] == "High Priority"].shape[0]),
        "medium_priority_count": int(df[df["priority"] == "Medium Priority"].shape[0]),
    }
