"""
csv_reporter.py
---------------
Compatibility wrapper delegating to reports.py for Excel, CSV, and JSON generation.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from reports import generate_csv_monitor, generate_json_export, generate_multisheet_excel


def generate_excel_report(
    quarterly_analyses: List[Dict[str, Any]],
    company_name: str,
    output_path: Path | str,
    scorecard_df: pd.DataFrame | None = None,
    topic_df: pd.DataFrame | None = None,
    cooccurrence_df: pd.DataFrame | None = None,
) -> Path:
    return generate_multisheet_excel(
        company_name=company_name,
        quarterly_data=quarterly_analyses,
        scorecard_df=scorecard_df if scorecard_df is not None else pd.DataFrame(),
        topic_distribution_df=topic_df if topic_df is not None else pd.DataFrame(),
        cooccurrence_df=cooccurrence_df if cooccurrence_df is not None else pd.DataFrame(),
        output_path=Path(output_path),
    )


def generate_csv_report(
    quarterly_analyses: List[Dict[str, Any]],
    company_name: str,
    output_path: Path | str,
) -> Path:
    return generate_csv_monitor(
        quarterly_data=quarterly_analyses,
        output_path=Path(output_path),
    )


def generate_json_report(
    quarterly_analyses: List[Dict[str, Any]],
    company_name: str,
    output_path: Path | str,
) -> Path:
    return generate_json_export(
        company_name=company_name,
        quarterly_data=quarterly_analyses,
        scorecard_df=pd.DataFrame(),
        output_path=Path(output_path),
    )