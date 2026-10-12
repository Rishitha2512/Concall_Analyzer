"""
reports.py
----------
Multi-format reporting engine for ConcallIQ:
1. Multi-sheet Excel Workbook (openpyxl)
2. CSV Quarter Monitor
3. Structured JSON Analysis
4. Styled One-Page Analyst Briefing (HTML / Printable)
Strictly adheres to corporate terminology with zero internal jargon.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


def generate_excel_report(
    company_name: str,
    quarterly_data: List[Dict[str, Any]],
    scorecard_df: pd.DataFrame,
    topic_distribution_df: pd.DataFrame,
    cooccurrence_df: Any,
    output_path: Path,
) -> Path:
    """
    Convenience alias for generate_multisheet_excel.
    """
    if isinstance(cooccurrence_df, list):
        cooccurrence_df = pd.DataFrame(cooccurrence_df)
    elif not isinstance(cooccurrence_df, pd.DataFrame):
        cooccurrence_df = pd.DataFrame()
    return generate_multisheet_excel(
        company_name=company_name,
        quarterly_data=quarterly_data,
        scorecard_df=scorecard_df,
        topic_distribution_df=topic_distribution_df,
        cooccurrence_df=cooccurrence_df,
        output_path=output_path,
    )


def generate_multisheet_excel(
    company_name: str,
    quarterly_data: List[Dict[str, Any]],
    scorecard_df: pd.DataFrame,
    topic_distribution_df: pd.DataFrame,
    cooccurrence_df: Any,
    output_path: Path,
) -> Path:
    """
    Generates a professional 5-sheet financial intelligence workbook.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(cooccurrence_df, list):
        cooccurrence_df = pd.DataFrame(cooccurrence_df)
    elif not isinstance(cooccurrence_df, pd.DataFrame):
        cooccurrence_df = pd.DataFrame()

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        # Sheet 1: Executive Overview
        overview_rows = []
        for q in quarterly_data:
            sents = q.get("sentences", [])
            risks = [s for s in sents if s.get("stance") == "Cautious" or s.get("risk_severity") in ("High", "Medium")]
            overview_rows.append({
                "Quarter": q.get("quarter", ""),
                "Call Date": q.get("call_date", ""),
                "Total Statements": len(sents),
                "Disclosed Risks": len(risks),
                "Source Document": q.get("filename", ""),
            })
        pd.DataFrame(overview_rows).to_excel(writer, sheet_name="Executive Summary", index=False)

        # Sheet 2: Key Numbers Scorecard
        if not scorecard_df.empty:
            scorecard_df.to_excel(writer, sheet_name="Key Numbers Scorecard", index=False)
        else:
            pd.DataFrame([{"Notice": "No scorecard metrics extracted"}]).to_excel(writer, sheet_name="Key Numbers Scorecard", index=False)

        # Sheet 3: Strategic Priorities & Topics
        if not topic_distribution_df.empty:
            topic_distribution_df.to_excel(writer, sheet_name="Topic Mix", index=False)
        else:
            pd.DataFrame([{"Notice": "No topic distributions calculated"}]).to_excel(writer, sheet_name="Topic Mix", index=False)

        # Sheet 4: Strategic Co-occurrences
        if not cooccurrence_df.empty:
            cooccurrence_df.to_excel(writer, sheet_name="Thematic Linkages", index=False)
        else:
            pd.DataFrame([{"Notice": "No thematic linkages mined"}]).to_excel(writer, sheet_name="Thematic Linkages", index=False)

        # Sheet 5: Sample Executive Citations
        stmt_rows = []
        for q in quarterly_data:
            for s in q.get("sentences", [])[:25]:
                stmt_rows.append({
                    "Quarter": q.get("quarter", ""),
                    "Speaker": s.get("speaker", ""),
                    "Speaker Role": s.get("speaker_role", ""),
                    "Section": s.get("section", ""),
                    "Topic": s.get("topic", ""),
                    "Stance": s.get("stance", ""),
                    "Statement": s.get("text", ""),
                })
        pd.DataFrame(stmt_rows).to_excel(writer, sheet_name="Statement Citations", index=False)

    return output_path


def generate_csv_export(
    quarterly_data: List[Dict[str, Any]],
    output_path: Path,
) -> Path:
    """Convenience alias for generate_csv_monitor."""
    return generate_csv_monitor(quarterly_data=quarterly_data, output_path=output_path)


def generate_csv_monitor(
    quarterly_data: List[Dict[str, Any]],
    output_path: Path,
) -> Path:
    """Generates standard CSV monitor of quarter KPIs."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for q in quarterly_data:
        sents = q.get("sentences", [])
        cautious_cnt = sum(1 for s in sents if s.get("stance") == "Cautious")
        confident_cnt = sum(1 for s in sents if s.get("stance") == "Confident")
        rows.append({
            "Company": q.get("company", ""),
            "Quarter": q.get("quarter", ""),
            "Call Date": q.get("call_date", ""),
            "Total Statements": len(sents),
            "Confident Signals": confident_cnt,
            "Cautious Signals": cautious_cnt,
            "Source File": q.get("filename", ""),
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False, encoding="utf-8")
    return output_path


def generate_json_export(
    company_name: str,
    quarterly_data: List[Dict[str, Any]],
    scorecard_df: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Exports complete structured JSON payload."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    clean_quarters = []
    for q in quarterly_data:
        clean_quarters.append({
            "quarter": q.get("quarter", ""),
            "call_date": q.get("call_date", ""),
            "filename": q.get("filename", ""),
            "page_count": q.get("page_count", 0),
            "total_sentences": len(q.get("sentences", [])),
            "sample_statements": q.get("sentences", [])[:15],
        })

    payload = {
        "company": company_name,
        "exported_at": datetime.now().isoformat(),
        "quarters_analyzed": len(quarterly_data),
        "data": clean_quarters,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return output_path


def generate_html_briefing(
    company_name: str,
    latest_quarter: Dict[str, Any],
    scorecard_row: Optional[Dict[str, Any]],
    plain_insights: List[str],
) -> str:
    """
    Renders a styled one-page executive briefing in HTML format suitable
    for equity analyst morning notes or investment committee memos.
    """
    q_name = latest_quarter.get("quarter", "Latest Quarter")
    call_dt = latest_quarter.get("call_date", "Recent")
    sents = latest_quarter.get("sentences", [])

    # Extract top confident and top cautious quotes
    top_confident = [s["text"] for s in sents if s.get("stance") == "Confident"][:2]
    top_cautious = [s["text"] for s in sents if s.get("stance") == "Cautious"][:2]

    insights_html = "".join(f"<li>{ins}</li>" for ins in plain_insights[:3])
    confident_html = "".join(f'<div class="quote-box pos">“{t}”</div>' for t in top_confident)
    cautious_html = "".join(f'<div class="quote-box neg">“{t}”</div>' for t in top_cautious)

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{company_name} — Analyst Briefing ({q_name})</title>
<style>
  body {{ font-family: 'Segoe UI', -apple-system, sans-serif; margin: 30px; background: #ffffff; color: #1e293b; line-height: 1.5; }}
  .header {{ border-bottom: 2px solid #0f172a; padding-bottom: 12px; margin-bottom: 20px; }}
  .title {{ font-size: 26px; font-weight: 800; color: #0f172a; }}
  .meta {{ font-size: 13px; color: #64748b; margin-top: 4px; text-transform: uppercase; letter-spacing: 0.5px; }}
  .disclaimer {{ background: #f8fafc; border-left: 3px solid #64748b; padding: 8px 12px; font-size: 11px; color: #475569; margin-bottom: 20px; }}
  .section {{ margin-bottom: 22px; }}
  .section-title {{ font-size: 16px; font-weight: 700; color: #0f172a; border-bottom: 1px solid #e2e8f0; padding-bottom: 4px; margin-bottom: 10px; }}
  .quote-box {{ background: #f1f5f9; border-radius: 6px; padding: 10px 14px; margin-bottom: 8px; font-size: 13px; font-style: italic; }}
  .quote-box.pos {{ border-left: 4px solid #10b981; }}
  .quote-box.neg {{ border-left: 4px solid #ef4444; }}
  ul {{ margin: 0; padding-left: 20px; font-size: 13px; }}
  li {{ margin-bottom: 6px; }}
  .footer {{ font-size: 11px; color: #94a3b8; text-align: center; border-top: 1px solid #e2e8f0; padding-top: 12px; margin-top: 30px; }}
</style>
</head>
<body>
  <div class="header">
    <div class="title">{company_name} — Quarterly Analyst Briefing</div>
    <div class="meta">Period: {q_name} | Call Date: {call_dt} | Source: Official Earnings Transcript</div>
  </div>

  <div class="disclaimer">
    RESEARCH MEMORANDUM: This briefing is prepared automatically by ConcallIQ for research and decision-support purposes. It does not constitute investment advice or securities recommendations.
  </div>

  <div class="section">
    <div class="section-title">Key Executive Strategic Takeaways</div>
    <ul>{insights_html}</ul>
  </div>

  <div class="section">
    <div class="section-title">Growth & Execution Highlights (Management Remarks)</div>
    {confident_html if confident_html else '<p style="font-size:13px;color:#64748b">No primary growth highlights flagged.</p>'}
  </div>

  <div class="section">
    <div class="section-title">Operating Risks & Headwinds Disclosed</div>
    {cautious_html if cautious_html else '<p style="font-size:13px;color:#64748b">No primary operational headwinds flagged.</p>'}
  </div>

  <div class="footer">
    Generated by ConcallIQ Earnings Intelligence Engine · Strict Auditability & Verification Standards
  </div>
</body>
</html>"""
    return html
