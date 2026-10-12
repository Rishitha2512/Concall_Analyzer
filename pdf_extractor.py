"""
pdf_extractor.py
----------------
Compatibility bridge delegating to data_loader.py for enterprise PDF text extraction,
period detection, deduplication, and caching.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import data_loader


def extract_text_from_pdf(pdf_path: Path | str, max_pages: int = 80) -> str:
    text, _ = data_loader.extract_raw_pdf_text(Path(pdf_path), max_pages=max_pages)
    return text


def detect_quarter_from_text(text: str) -> Optional[str]:
    dt = data_loader.detect_call_date(text)
    period, _ = data_loader.detect_fiscal_period(text, dt)
    return period


def load_company_pdfs(company_name: str) -> List[Dict[str, Any]]:
    res = data_loader.load_company_data(company_name, use_cache=True)
    return res["valid_transcripts"]


def list_companies() -> List[str]:
    return data_loader.list_available_companies()
