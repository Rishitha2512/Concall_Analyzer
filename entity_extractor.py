"""
entity_extractor.py
-------------------
Enterprise information extraction and named entity extraction engine.
Pulls money, percentages, dates, periods, people, organizations, technologies,
geographies, and deal terms from conference call transcripts.
Builds the Key Numbers Scorecard with QoQ deltas, citation linkages,
and speaker/section analytics.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Entity Regex & Vocabulary Definitions
# -----------------------------------------------------------------------------
MONEY_PAT = re.compile(
    r"(?:₹|Rs\.?|INR|\$|USD|€|EUR|£|GBP)\s*\d+(?:[.,]\d+)*(?:\s*(?:billion|million|crore|lakh|bn|mn|cr|k|trillion))?\b",
    re.I,
)

PERCENT_PAT = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:%|percent|bps|basis points)(?!\w)",
    re.I,
)

PERIOD_PAT = re.compile(
    r"\b(?:Q[1-4]|first|second|third|fourth)\s*(?:quarter)?\s*(?:of\s+)?(?:FY\s*['\"]?|financial year\s*)?\s*(?:20\d{2}|\d{2})\b",
    re.I,
)

DATE_PAT = re.compile(
    r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+20\d{2}\b",
    re.I,
)

KNOWN_ORGS = [
    "Microsoft", "Amazon", "AWS", "Google", "Google Cloud", "NVIDIA", "SAP",
    "Salesforce", "Oracle", "IBM", "ServiceNow", "Apple", "Meta", "Intel",
    "Cisco", "Accenture", "Infosys", "TCS", "Wipro", "HCLTech", "HCL",
    "Cognizant", "LTIMindtree", "Mphasis", "Coforge", "Tech Mahindra",
    "JPMorgan", "Morgan Stanley", "Goldman Sachs", "Citigroup", "BNP Paribas",
    "HSBC", "Barclays", "Bank of America", "Nomura", "Kotak", "ICICI",
]

KNOWN_GEOS = [
    "North America", "Americas", "United States", "US", "Europe", "UK",
    "United Kingdom", "Continental Europe", "Germany", "France", "Nordics",
    "Asia Pacific", "APAC", "Australia", "Japan", "India", "Middle East",
    "Latin America", "LATAM", "Singapore",
]

KNOWN_TECHS = [
    "Generative AI", "GenAI", "Artificial Intelligence", "AI", "Machine Learning",
    "LLM", "Agentic AI", "Cloud", "Cloud Migration", "AWS", "Azure", "GCP",
    "SAP S/4HANA", "Cybersecurity", "Data Platform", "Data Lake", "Automation",
    "Microservices", "DevOps", "IoT", "Blockchain", "Digital Twin", "Full Stack",
]

DEAL_TERMS = [
    "TCV", "ACV", "Total Contract Value", "Annual Contract Value", "Mega Deal",
    "Large Deal", "Order Book", "Book-to-Bill", "Vendor Consolidation",
    "Renewal", "Net New", "Sole Source", "Pipeline Conversion", "Multi-Year",
]


# -----------------------------------------------------------------------------
# Entity Extraction Functions
# -----------------------------------------------------------------------------
def extract_all_entities(text: str, quarter: str = "", company: str = "") -> pd.DataFrame:
    """
    Extracts all structured entities with context snippet, entity type,
    quarter, and company.
    """
    rows = []
    seen = set()

    # 1. Money
    for m in MONEY_PAT.finditer(text):
        val = m.group(0).strip()
        key = ("Money", val.lower())
        if key not in seen:
            seen.add(key)
            start = max(0, m.start() - 60)
            end = min(len(text), m.end() + 60)
            ctx = text[start:end].replace("\n", " ").strip()
            rows.append({
                "Entity": val,
                "Type": "Money",
                "Context": f"…{ctx}…",
                "Quarter": quarter,
                "Company": company,
            })

    # 2. Percentages
    for m in PERCENT_PAT.finditer(text):
        val = m.group(0).strip()
        key = ("Percentage", val.lower())
        if key not in seen:
            seen.add(key)
            start = max(0, m.start() - 60)
            end = min(len(text), m.end() + 60)
            ctx = text[start:end].replace("\n", " ").strip()
            rows.append({
                "Entity": val,
                "Type": "Percentage",
                "Context": f"…{ctx}…",
                "Quarter": quarter,
                "Company": company,
            })

    # 3. Fiscal Periods
    for m in PERIOD_PAT.finditer(text):
        val = m.group(0).strip()
        key = ("Fiscal Period", val.lower())
        if key not in seen:
            seen.add(key)
            start = max(0, m.start() - 60)
            end = min(len(text), m.end() + 60)
            ctx = text[start:end].replace("\n", " ").strip()
            rows.append({
                "Entity": val,
                "Type": "Fiscal Period",
                "Context": f"…{ctx}…",
                "Quarter": quarter,
                "Company": company,
            })

    # 4. Dates
    for m in DATE_PAT.finditer(text):
        val = m.group(0).strip()
        key = ("Date", val.lower())
        if key not in seen:
            seen.add(key)
            start = max(0, m.start() - 60)
            end = min(len(text), m.end() + 60)
            ctx = text[start:end].replace("\n", " ").strip()
            rows.append({
                "Entity": val,
                "Type": "Date",
                "Context": f"…{ctx}…",
                "Quarter": quarter,
                "Company": company,
            })

    # 5. Organizations & Clients
    for org in KNOWN_ORGS:
        for m in re.finditer(r"\b" + re.escape(org) + r"\b", text, re.I):
            val = org
            key = ("Organization", val.lower())
            if key not in seen:
                seen.add(key)
                start = max(0, m.start() - 60)
                end = min(len(text), m.end() + 60)
                ctx = text[start:end].replace("\n", " ").strip()
                rows.append({
                    "Entity": val,
                    "Type": "Organization / Ecosystem",
                    "Context": f"…{ctx}…",
                    "Quarter": quarter,
                    "Company": company,
                })
            break

    # 6. Geographies
    for geo in KNOWN_GEOS:
        for m in re.finditer(r"\b" + re.escape(geo) + r"\b", text, re.I):
            val = geo
            key = ("Geography", val.lower())
            if key not in seen:
                seen.add(key)
                start = max(0, m.start() - 60)
                end = min(len(text), m.end() + 60)
                ctx = text[start:end].replace("\n", " ").strip()
                rows.append({
                    "Entity": val,
                    "Type": "Geography",
                    "Context": f"…{ctx}…",
                    "Quarter": quarter,
                    "Company": company,
                })
            break

    # 7. Technologies
    for tech in KNOWN_TECHS:
        for m in re.finditer(r"\b" + re.escape(tech) + r"\b", text, re.I):
            val = tech
            key = ("Technology", val.lower())
            if key not in seen:
                seen.add(key)
                start = max(0, m.start() - 60)
                end = min(len(text), m.end() + 60)
                ctx = text[start:end].replace("\n", " ").strip()
                rows.append({
                    "Entity": val,
                    "Type": "Technology",
                    "Context": f"…{ctx}…",
                    "Quarter": quarter,
                    "Company": company,
                })
            break

    # 8. Deal Terms
    for dt in DEAL_TERMS:
        for m in re.finditer(r"\b" + re.escape(dt) + r"\b", text, re.I):
            val = dt
            key = ("Deal Term", val.lower())
            if key not in seen:
                seen.add(key)
                start = max(0, m.start() - 60)
                end = min(len(text), m.end() + 60)
                ctx = text[start:end].replace("\n", " ").strip()
                rows.append({
                    "Entity": val,
                    "Type": "Deal Term",
                    "Context": f"…{ctx}…",
                    "Quarter": quarter,
                    "Company": company,
                })
            break

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Key Numbers Scorecard per Quarter
# -----------------------------------------------------------------------------
def build_quarter_scorecard(transcript: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extracts the eight core corporate metrics from the transcript sentences:
    1. Revenue Growth (%)
    2. Operating Margin (%)
    3. Net Profit / PAT ($ or ₹)
    4. TCV / Order Book ($ or ₹)
    5. Headcount (associates)
    6. Attrition Rate (%)
    7. Large Deal Wins (count)
    8. Guidance Outlook (%)
    Each metric includes the exact supporting sentence, speaker, and section.
    """
    sentences = transcript.get("sentences", [])
    quarter = transcript.get("quarter", "")
    company = transcript.get("company", "")

    # Default metrics structure
    scorecard: Dict[str, Dict[str, Any]] = {
        "Revenue Growth": {"value": "N/A", "raw_num": None, "unit": "%", "sentence": "", "speaker": "", "section": ""},
        "Operating Margin": {"value": "N/A", "raw_num": None, "unit": "%", "sentence": "", "speaker": "", "section": ""},
        "Net Profit": {"value": "N/A", "raw_num": None, "unit": "", "sentence": "", "speaker": "", "section": ""},
        "TCV / Order Book": {"value": "N/A", "raw_num": None, "unit": "$", "sentence": "", "speaker": "", "section": ""},
        "Headcount": {"value": "N/A", "raw_num": None, "unit": "", "sentence": "", "speaker": "", "section": ""},
        "Attrition": {"value": "N/A", "raw_num": None, "unit": "%", "sentence": "", "speaker": "", "section": ""},
        "Deal Wins": {"value": "N/A", "raw_num": None, "unit": "deals", "sentence": "", "speaker": "", "section": ""},
        "Guidance": {"value": "N/A", "raw_num": None, "unit": "%", "sentence": "", "speaker": "", "section": ""},
    }

    for s in sentences:
        txt = s["text"]
        low = txt.lower()

        # 1. Operating Margin
        if scorecard["Operating Margin"]["value"] == "N/A" and ("operating margin" in low or "ebit margin" in low):
            m = re.search(r"(\d+(?:\.\d+)?)\s*(?:%|percent)", txt)
            if m:
                scorecard["Operating Margin"] = {
                    "value": f"{m.group(1)}%",
                    "raw_num": float(m.group(1)),
                    "unit": "%",
                    "sentence": txt,
                    "speaker": s.get("speaker", "Management"),
                    "section": s.get("section", "Prepared Remarks"),
                }

        # 2. Revenue Growth
        if scorecard["Revenue Growth"]["value"] == "N/A" and ("revenue growth" in low or "grew by" in low or "constant currency" in low):
            m = re.search(r"(-?\d+(?:\.\d+)?)\s*(?:%|percent)", txt)
            if m and float(m.group(1)) < 40.0:  # sanity check
                scorecard["Revenue Growth"] = {
                    "value": f"{m.group(1)}%",
                    "raw_num": float(m.group(1)),
                    "unit": "%",
                    "sentence": txt,
                    "speaker": s.get("speaker", "Management"),
                    "section": s.get("section", "Prepared Remarks"),
                }

        # 3. TCV / Order Book
        if scorecard["TCV / Order Book"]["value"] == "N/A" and ("tcv" in low or "total contract value" in low or "order book" in low or "deal booking" in low):
            m = re.search(r"(?:₹|Rs\.?|\$|USD)\s*(\d+(?:[.,]\d+)?)\s*(?:billion|million|crore|bn|mn)?\b", txt, re.I)
            if m:
                scorecard["TCV / Order Book"] = {
                    "value": m.group(0).strip(),
                    "raw_num": float(m.group(1).replace(",", "")),
                    "unit": "$",
                    "sentence": txt,
                    "speaker": s.get("speaker", "Management"),
                    "section": s.get("section", "Prepared Remarks"),
                }

        # 4. Attrition
        if scorecard["Attrition"]["value"] == "N/A" and ("attrition" in low):
            m = re.search(r"(\d+(?:\.\d+)?)\s*(?:%|percent)", txt)
            if m:
                scorecard["Attrition"] = {
                    "value": f"{m.group(1)}%",
                    "raw_num": float(m.group(1)),
                    "unit": "%",
                    "sentence": txt,
                    "speaker": s.get("speaker", "Management"),
                    "section": s.get("section", "Prepared Remarks"),
                }

        # 5. Headcount
        if scorecard["Headcount"]["value"] == "N/A" and ("headcount" in low or "employees" in low or "associates" in low):
            m = re.search(r"(\d{2,3}(?:,\d{3})+|\d{5,6})", txt)
            if m:
                scorecard["Headcount"] = {
                    "value": m.group(1),
                    "raw_num": float(m.group(1).replace(",", "")),
                    "unit": "associates",
                    "sentence": txt,
                    "speaker": s.get("speaker", "Management"),
                    "section": s.get("section", "Prepared Remarks"),
                }

        # 6. Deal Wins
        if scorecard["Deal Wins"]["value"] == "N/A" and ("large deal" in low or "mega deal" in low or "deals signed" in low):
            m = re.search(r"\b(\d{1,3})\s+(?:large deal|mega deal|strategic deal)", txt, re.I)
            if m:
                scorecard["Deal Wins"] = {
                    "value": f"{m.group(1)} deals",
                    "raw_num": float(m.group(1)),
                    "unit": "deals",
                    "sentence": txt,
                    "speaker": s.get("speaker", "Management"),
                    "section": s.get("section", "Prepared Remarks"),
                }

        # 7. Guidance
        if scorecard["Guidance"]["value"] == "N/A" and ("guidance" in low and ("revenue" in low or "margin" in low)):
            m = re.search(r"(\d+(?:\.\d+)?\s*(?:to|-)\s*\d+(?:\.\d+)?\s*%)", txt, re.I)
            if m:
                scorecard["Guidance"] = {
                    "value": m.group(1),
                    "raw_num": None,
                    "unit": "%",
                    "sentence": txt,
                    "speaker": s.get("speaker", "Management"),
                    "section": s.get("section", "Prepared Remarks"),
                }

        # 8. Net Profit
        if scorecard["Net Profit"]["value"] == "N/A" and ("net profit" in low or "pat" in low):
            m = re.search(r"(?:₹|Rs\.?|\$|USD)\s*(\d+(?:[.,]\d+)?\s*(?:crore|billion|million|cr|mn)?)", txt, re.I)
            if m:
                scorecard["Net Profit"] = {
                    "value": m.group(0).strip(),
                    "raw_num": float(re.search(r"\d+", m.group(1)).group(0)),
                    "unit": "",
                    "sentence": txt,
                    "speaker": s.get("speaker", "Management"),
                    "section": s.get("section", "Prepared Remarks"),
                }

    return {
        "quarter": quarter,
        "company": company,
        "metrics": scorecard,
    }


def build_qoq_change_table(scorecards: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Computes quarter-on-quarter changes across key numbers with deltas.
    """
    rows = []
    if not scorecards:
        return pd.DataFrame()

    metric_names = [
        "Revenue Growth", "Operating Margin", "Net Profit",
        "TCV / Order Book", "Headcount", "Attrition", "Deal Wins", "Guidance"
    ]

    for i, sc in enumerate(scorecards):
        q = sc["quarter"]
        prev_sc = scorecards[i - 1] if i > 0 else None

        for m_name in metric_names:
            curr_data = sc["metrics"].get(m_name, {})
            curr_val = curr_data.get("value", "N/A")
            curr_raw = curr_data.get("raw_num")
            evidence = curr_data.get("sentence", "")
            speaker = curr_data.get("speaker", "")

            delta_str = "—"
            if prev_sc and curr_raw is not None:
                prev_data = prev_sc["metrics"].get(m_name, {})
                prev_raw = prev_data.get("raw_num")
                if prev_raw is not None:
                    diff = curr_raw - prev_raw
                    if "%" in curr_val:
                        delta_str = f"+{diff:.1f}%" if diff > 0 else f"{diff:.1f}%"
                    else:
                        delta_str = f"+{diff:,.0f}" if diff > 0 else f"{diff:,.0f}"

            rows.append({
                "Quarter": q,
                "Metric": m_name,
                "Reported Value": curr_val,
                "QoQ Shift": delta_str,
                "Speaker": speaker,
                "Citation": evidence,
            })

    return pd.DataFrame(rows)


def analyze_speaker_distribution(transcripts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Computes speaker breakdown:
    - Management statements vs Analyst statements
    - Prepared Remarks statements vs Q&A statements
    """
    mgmt_count = 0
    analyst_count = 0
    remarks_count = 0
    qa_count = 0

    speaker_counts: Dict[str, int] = {}

    for t in transcripts:
        for s in t.get("sentences", []):
            role = s.get("speaker_role", "Management")
            sec = s.get("section", "Prepared Remarks")
            spk = s.get("speaker", "Management")

            if role == "Management":
                mgmt_count += 1
            elif role == "Analyst":
                analyst_count += 1

            if sec == "Prepared Remarks":
                remarks_count += 1
            else:
                qa_count += 1

            speaker_counts[spk] = speaker_counts.get(spk, 0) + 1

    top_speakers = sorted(speaker_counts.items(), key=lambda x: x[1], reverse=True)[:10]

    return {
        "management_count": mgmt_count,
        "analyst_count": analyst_count,
        "prepared_remarks_count": remarks_count,
        "qa_count": qa_count,
        "top_speakers": top_speakers,
    }
