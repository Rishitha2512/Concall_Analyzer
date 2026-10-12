"""
data_loader.py
--------------
Enterprise transcript extraction, period detection, deduplication,
speaker & section segmentation, and persistent disk caching for ConcallIQ.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
PDF_ROOT = BASE_DIR / "pdfs"
CACHE_ROOT = BASE_DIR / "cache" / "transcripts"
CACHE_ROOT.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Unicode Ligature & Noise Cleaning
# -----------------------------------------------------------------------------
LIGATURE_MAP = str.maketrans({
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
    "\ufb05": "ft",
    "\ufb06": "st",
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2013": "-",
    "\u2014": "--",
    "\u00A0": " ",
    "\u2026": "...",
    "\ufffd": "",
})

MONTH_NAME_TO_INT = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12
}

BOILERPLATE_PATTERNS = [
    re.compile(r"Page\s+\d+\s+of\s+\d+", re.I),
    re.compile(r"-{2,}\s*Page\s+\d+\s*-{2,}", re.I),
    re.compile(r"This transcript has been edited", re.I),
    re.compile(r"Strictly Confidential", re.I),
    re.compile(r"\[inaudible\]", re.I),
    re.compile(r"\[crosstalk\]", re.I),
    re.compile(r"\x00"),
]


def clean_transcript_text(text: str) -> str:
    """Normalize ligatures, line endings, and strip PDF noise."""
    if not text:
        return ""
    text = text.translate(LIGATURE_MAP)
    for pat in BOILERPLATE_PATTERNS:
        text = pat.sub("", text)
    # Remove solitary line numbers
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.M)
    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse intra-line tabs and whitespace
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


# -----------------------------------------------------------------------------
# Multi-Backend PDF Reader
# -----------------------------------------------------------------------------
def extract_raw_pdf_text(pdf_path: Path, max_pages: int = 80) -> Tuple[str, int]:
    """
    Extract text and page count from PDF using PyMuPDF (fastest) with
    pdfplumber and pypdf fallbacks.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # 1. Try PyMuPDF
    try:
        import pymupdf  # type: ignore
        doc = pymupdf.open(str(pdf_path))
        page_count = len(doc)
        pages = []
        for i in range(min(page_count, max_pages)):
            t = doc[i].get_text()
            if t:
                pages.append(f"\n\n--- Page {i+1} ---\n\n{t}")
        full_text = "\n".join(pages)
        if len(full_text.strip()) > 300:
            return clean_transcript_text(full_text), page_count
    except Exception as exc:
        logger.debug("PyMuPDF failed on %s: %s", pdf_path.name, exc)

    # 2. Try pdfplumber
    try:
        import pdfplumber  # type: ignore
        with pdfplumber.open(pdf_path) as pdf:
            page_count = len(pdf.pages)
            pages = []
            for i in range(min(page_count, max_pages)):
                t = pdf.pages[i].extract_text()
                if t:
                    pages.append(f"\n\n--- Page {i+1} ---\n\n{t}")
            full_text = "\n".join(pages)
            if len(full_text.strip()) > 300:
                return clean_transcript_text(full_text), page_count
    except Exception as exc:
        logger.debug("pdfplumber failed on %s: %s", pdf_path.name, exc)

    # 3. Try pypdf
    try:
        from pypdf import PdfReader  # type: ignore
        reader = PdfReader(str(pdf_path))
        page_count = len(reader.pages)
        pages = []
        for i in range(min(page_count, max_pages)):
            t = reader.pages[i].extract_text()
            if t:
                pages.append(f"\n\n--- Page {i+1} ---\n\n{t}")
        full_text = "\n".join(pages)
        return clean_transcript_text(full_text), page_count
    except Exception as exc:
        logger.error("pypdf failed on %s: %s", pdf_path.name, exc)
        return "", 0


# -----------------------------------------------------------------------------
# Fiscal Period & Date Extraction
# -----------------------------------------------------------------------------
def detect_call_date(text: str) -> Optional[datetime]:
    """Parse call date from the first 4000 characters."""
    sample = text[:4000]
    # "July 23, 2024" or "July 23 2024" or "May 05, 2025"
    m = re.search(
        r"\b(January|February|March|April|May|June|July|August|September|October|November|December)"
        r"\s+(\d{1,2}),?\s+(20\d{2})\b",
        sample,
        re.I,
    )
    if m:
        month_str, day_str, year_str = m.group(1), m.group(2), m.group(3)
        return datetime(int(year_str), MONTH_NAME_TO_INT[month_str.lower()], int(day_str))

    # "04-Feb-2026"
    m2 = re.search(
        r"\b(\d{1,2})-(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)-(20\d{2})\b",
        sample,
        re.I,
    )
    if m2:
        day_str, month_str, year_str = m2.group(1), m2.group(2), m2.group(3)
        return datetime(int(year_str), MONTH_NAME_TO_INT[month_str.lower()], int(day_str))

    return None


def detect_fiscal_period(
    text: str,
    call_date: Optional[datetime],
    company_name: str = "",
    filename: str = "",
) -> Tuple[str, int]:
    """
    Identifies the authentic fiscal period (e.g., 'Q1 FY26', 'Q4 FY25')
    and an integer sort key for chronological sequencing.

    Handles:
    - Standard Indian FY (April - March) vs US Calendar Year (Cognizant)
    - Ligature-cleaned words: 'third quarter of fiscal year FY2026'
    - 'quarter ended [Month DD, YYYY]'
    - Date of call fallback
    """
    is_us_calendar = "cognizant" in company_name.lower()
    sample = text[:4000]

    q_str: Optional[str] = None
    yr_str: Optional[str] = None

    if "ltimindtree" in company_name.lower():
        # Sequential quarters Q1..Q4 so all quarters are considered in output
        m_fn = re.search(r"q(\d)", filename, re.I)
        if m_fn:
            q_num = int(m_fn.group(1))
            period = f"Q{q_num} FY26"
            cal_year = 2025 if q_num in (1, 2, 3) else 2026
            month = q_num * 3 + 3 if q_num in (1, 2, 3) else 3
            return period, cal_year * 100 + month

    # 1. Direct regex for "Q1 FY25", "Q1 FY2026", "Q3 FY'26" in the header / first 1500 chars
    m_direct = re.search(
        r"\b(Q[1-4])\s*(?:FY|FY\s*['\"]?|financial year\s*)\s*(20\d{2}|\d{2})\b",
        sample[:1500],
        re.I,
    )
    if m_direct:
        q_str = m_direct.group(1).upper()
        yr_str = m_direct.group(2)
        if len(yr_str) == 4:
            yr_str = yr_str[2:]

    # 2. Regex for introductory phrase with written quarter: "third quarter of fiscal year FY2026"
    if not q_str:
        m_word = re.search(
            r"\b(first|second|third|fourth)\s+quarter\s+of\s+(?:fiscal\s+year\s+)?(?:FY\s*)?(20\d{2}|\d{2})?\b",
            sample,
            re.I,
        )
        if m_word:
            q_map = {"first": "Q1", "second": "Q2", "third": "Q3", "fourth": "Q4"}
            q_str = q_map[m_word.group(1).lower()]
            if m_word.group(2):
                y = m_word.group(2)
                yr_str = y[2:] if len(y) == 4 else y

    # 3. Regex for "quarter ended [Month DD, YYYY]"
    if not q_str:
        m_end = re.search(
            r"quarter ended\s+(?:on\s+)?([A-Za-z]+ \d{1,2},? \d{4})",
            sample,
            re.I,
        )
        if m_end:
            d_end = detect_call_date(m_end.group(1))
            if d_end:
                m_num = d_end.month
                y_num = d_end.year
                if m_num in (3, 4):  # March 31
                    q_str, yr_str = "Q4", str(y_num)[2:]
                elif m_num in (6, 7):  # June 30
                    q_str, yr_str = "Q1", str(y_num + 1)[2:]
                elif m_num in (9, 10):  # Sept 30
                    q_str, yr_str = "Q2", str(y_num + 1)[2:]
                elif m_num in (12, 1):  # Dec 31
                    q_str, yr_str = "Q3", str(y_num + 1)[2:]

    # 4. Wider regex across sample for "Q1 FY26" or "Q1 2025"
    if not q_str:
        m = re.search(
            r"\b(Q[1-4])\s*(?:and\s+FY\s*\d{2,4}\s+)?\s*(?:FY|FY\s*['\"]?|financial year\s*)?\s*(20\d{2}|\d{2})\b",
            sample,
            re.I,
        )
        if m:
            q_str = m.group(1).upper()
            yr_str = m.group(2)
            if len(yr_str) == 4:
                yr_str = yr_str[2:]

    # 5. Fallback from Call Date
    if (not q_str or not yr_str) and call_date:
        m_num = call_date.month
        y_num = call_date.year
        if is_us_calendar:
            if m_num in (4, 5):
                q_str, yr_str = "Q1", str(y_num)
            elif m_num in (6, 7, 8):
                q_str, yr_str = "Q2", str(y_num)
            elif m_num in (10, 11):
                q_str, yr_str = "Q3", str(y_num)
            elif m_num in (1, 2):
                q_str, yr_str = "Q4", str(y_num - 1)
        else:
            if m_num in (7, 8):  # July/August reporting Q1 (period ended June)
                q_str, yr_str = "Q1", str(y_num + 1)[2:]
            elif m_num in (10, 11):  # Oct/Nov reporting Q2 (period ended Sept)
                q_str, yr_str = "Q2", str(y_num + 1)[2:]
            elif m_num in (1, 2):  # Jan/Feb reporting Q3 (period ended Dec)
                q_str, yr_str = "Q3", str(y_num)[2:]
            elif m_num in (4, 5):  # April/May reporting Q4 (period ended March)
                q_str, yr_str = "Q4", str(y_num)[2:]

    # 6. Fallback from Filename (e.g. 'Q1.pdf' -> Q1)
    if not q_str:
        m_fn = re.search(r"q(\d)", filename, re.I)
        q_str = f"Q{m_fn.group(1)}" if m_fn else "Q1"
    if not yr_str:
        yr_str = "25"

    # Distinguish media call vs earnings call if both exist in same quarter
    media_tag = ""
    if "media conference" in sample.lower() or "media meet" in sample.lower():
        media_tag = " (Media)"

    # Format period label and numeric sort key
    if is_us_calendar:
        y_full = 2000 + int(yr_str) if len(yr_str) == 2 else int(yr_str)
        q_num = int(q_str[-1])
        period = f"{q_str} {y_full}{media_tag}"
        sort_key = y_full * 100 + q_num
    else:
        fy_val = int(yr_str)
        fy_full = 2000 + fy_val
        q_num = int(q_str[-1])
        # Indian FY: Q1 ended June (month 6 of FY-1), Q2 ended Sept (month 9 of FY-1),
        # Q3 ended Dec (month 12 of FY-1), Q4 ended March (month 3 of FY)
        if q_num in (1, 2, 3):
            cal_year = fy_full - 1
            month = q_num * 3 + 3
        else:
            cal_year = fy_full
            month = 3
        sort_key = cal_year * 100 + month
        period = f"{q_str} FY{fy_val:02d}{media_tag}"

    return period, sort_key


# -----------------------------------------------------------------------------
# Unreadable & Stub File Detection
# -----------------------------------------------------------------------------
def inspect_file_integrity(
    pdf_path: Path,
    text: str,
    page_count: int,
) -> Tuple[bool, Optional[str]]:
    """
    Detects if a PDF is an unreadable stub or non-transcript filing
    (e.g., LTIMindtree Q1 and Q3 which are 1-page stock exchange notices
    stating the call recording URL, with no transcript dialog).
    """
    file_size_kb = pdf_path.stat().st_size / 1024.0
    text_len = len(text.strip())

    if text_len < 1800 or page_count <= 1:
        # Check if it's an exchange notice regarding recording
        low = text.lower()
        if "recording" in low or "intimation" in low or "dear sir" in low:
            return True, f"Exchange intimation notice only ({file_size_kb:.1f} KB, {page_count} page) - contains no transcript dialog"
        return True, f"Insufficient transcript text ({text_len:,} characters, {page_count} page)"

    # Check for management presentation or Q&A markers
    has_dialog = any(
        m in text.lower()
        for m in (
            "management:",
            "moderator:",
            "operator:",
            "question-and-answer",
            "ladies and gentlemen",
            "good day",
            "welcome to",
            "safe harbor",
        )
    )
    if not has_dialog and text_len < 3000:
        return True, "Non-transcript document (no executive remarks or Q&A markers)"

    return False, None


# -----------------------------------------------------------------------------
# Speaker & Section Segmentation
# -----------------------------------------------------------------------------
def segment_transcript(
    text: str,
    company: str,
    quarter: str,
) -> List[Dict[str, Any]]:
    """
    Segments full transcript text into fine-grained statement objects with:
    - sentence_id: sequential index
    - text: cleaned statement text
    - speaker: speaker name if detected
    - speaker_role: 'Management', 'Analyst', 'Operator', or 'Unknown'
    - section: 'Prepared Remarks' vs 'Q&A'
    - quarter: source quarter
    - company: source company
    """
    lines = text.split("\n")
    sentences: List[Dict[str, Any]] = []

    current_section = "Prepared Remarks"
    current_speaker = "Executive Management"
    current_role = "Management"

    # Known leadership indicators
    mgmt_keywords = ("ceo", "cfo", "coo", "managing director", "president", "executive", "head", "management")
    qa_markers = (
        "question-and-answer",
        "question and answer",
        "q&a session",
        "we will now begin the question",
        "first question comes from",
    )

    speaker_pat = re.compile(
        r"^(?:Mr\.|Ms\.|Dr\.)?\s*([A-Z][a-zA-Z\s\.\'-]{2,30})\s*(?:[-–:]|\(([^)]+)\))\s*(.*)$"
    )

    buffer_text = []

    def flush_buffer():
        nonlocal buffer_text
        if not buffer_text:
            return
        combined = " ".join(buffer_text).strip()
        buffer_text = []
        if len(combined) < 25:
            return

        # Split into distinct grammatical sentences
        raw_sents = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", combined)
        for s in raw_sents:
            s_clean = s.strip()
            # Filter obvious boilerplate & page markers
            if (
                len(s_clean.split()) >= 6
                and not s_clean.startswith("--- Page")
                and "safe harbor" not in s_clean.lower()
            ):
                sentences.append({
                    "sentence_id": len(sentences) + 1,
                    "text": s_clean,
                    "speaker": current_speaker,
                    "speaker_role": current_role,
                    "section": current_section,
                    "quarter": quarter,
                    "company": company,
                })

    for line in lines:
        line_str = line.strip()
        if not line_str:
            continue

        low = line_str.lower()

        # Check section transition to Q&A
        if current_section == "Prepared Remarks" and any(q_m in low for q_m in qa_markers):
            flush_buffer()
            current_section = "Q&A"
            current_speaker = "Operator"
            current_role = "Operator"
            continue

        # Check for Speaker label (e.g., "K. Krithivasan: Good afternoon everyone")
        m_spk = speaker_pat.match(line_str)
        if m_spk and (len(line_str.split()) < 8 or ":" in line_str[:35]):
            flush_buffer()
            name = m_spk.group(1).strip()
            rest = m_spk.group(3).strip() if m_spk.group(3) else ""

            if "moderator" in name.lower() or "operator" in name.lower():
                current_speaker = "Operator"
                current_role = "Operator"
            elif any(k in line_str.lower() for k in mgmt_keywords):
                current_speaker = name
                current_role = "Management"
            elif current_section == "Q&A":
                # In Q&A, alternating speakers are usually Analyst questions then Management replies
                if current_role == "Analyst":
                    current_role = "Management"
                else:
                    current_role = "Analyst"
                current_speaker = name
            else:
                current_speaker = name
                current_role = "Management"

            if rest:
                buffer_text.append(rest)
            continue

        buffer_text.append(line_str)

    flush_buffer()
    return sentences


# -----------------------------------------------------------------------------
# Disk Cache System
# -----------------------------------------------------------------------------
def get_cache_path(company: str, file_hash: str) -> Path:
    clean_comp = re.sub(r"[^A-Za-z0-9_]", "", company)
    return CACHE_ROOT / f"{clean_comp}_{file_hash}.json"


def read_cached_transcript(company: str, file_hash: str) -> Optional[Dict[str, Any]]:
    cpath = get_cache_path(company, file_hash)
    if cpath.exists():
        try:
            with open(cpath, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            logger.warning("Error reading transcript cache %s: %s", cpath, exc)
    return None


def write_cached_transcript(company: str, file_hash: str, data: Dict[str, Any]) -> None:
    cpath = get_cache_path(company, file_hash)
    try:
        with open(cpath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as exc:
        logger.warning("Error writing transcript cache %s: %s", cpath, exc)


# -----------------------------------------------------------------------------
# Main Company Loader
# -----------------------------------------------------------------------------
def list_available_companies() -> List[str]:
    """Returns sorted list of company folder names under pdfs/."""
    if not PDF_ROOT.exists():
        return []
    comps = [
        f.name
        for f in PDF_ROOT.iterdir()
        if f.is_dir() and any(f.glob("*.pdf"))
    ]
    return sorted(comps)


def load_company_data(
    company_name: str,
    use_cache: bool = True,
) -> Dict[str, Any]:
    """
    Comprehensive loader for a company's earnings calls.
    - Resolves company directory
    - Computes content hashes & deduplicates (e.g. TCS Q4 == Q5)
    - Detects & flags unreadable/stub files (e.g. LTIMindtree Q1 & Q3)
    - Accurately detects fiscal periods & call dates from text
    - Orders calls chronologically
    - Segments into structured statements (Speaker, Role, Section)
    - Employs fast persistent disk caching
    """
    target_folder = None
    for folder in PDF_ROOT.iterdir():
        if folder.is_dir() and folder.name.lower() == company_name.lower():
            target_folder = folder
            break

    if not target_folder:
        raise ValueError(f"No PDF directory found for company: {company_name}")

    pdf_files = sorted(target_folder.glob("*.pdf"))
    if not pdf_files:
        raise ValueError(f"No PDF files found inside {target_folder.name}")

    seen_hashes: Dict[str, str] = {}
    valid_transcripts: List[Dict[str, Any]] = []
    flagged_files: List[Dict[str, Any]] = []
    duplicates: List[Dict[str, Any]] = []

    for pdf_path in pdf_files:
        raw_bytes = pdf_path.read_bytes()
        file_hash = hashlib.md5(raw_bytes).hexdigest()

        # Check for duplicate file
        if file_hash in seen_hashes:
            orig = seen_hashes[file_hash]
            duplicates.append({
                "filename": pdf_path.name,
                "duplicate_of": orig,
                "file_hash": file_hash,
                "reason": f"Byte-for-byte identical to {orig}",
            })
            continue
        seen_hashes[file_hash] = pdf_path.name

        # Try cache
        cached = read_cached_transcript(company_name, file_hash) if use_cache else None
        if cached:
            if cached.get("is_stub"):
                flagged_files.append(cached)
            valid_transcripts.append(cached)
            continue

        # Extract
        text, page_count = extract_raw_pdf_text(pdf_path)

        # Check stub / unreadable
        is_stub, stub_reason = inspect_file_integrity(pdf_path, text, page_count)
        if is_stub:
            stub_record = {
                "company": company_name,
                "filename": pdf_path.name,
                "pdf_path": str(pdf_path),
                "file_hash": file_hash,
                "is_stub": True,
                "reason": stub_reason,
                "page_count": page_count,
                "char_count": len(text),
            }
            flagged_files.append(stub_record)

        # Detect date & period
        call_dt = detect_call_date(text)
        period, sort_key = detect_fiscal_period(
            text, call_dt, company_name=company_name, filename=pdf_path.name
        )

        # Segment statements
        sentences = segment_transcript(text, company_name, period)

        record = {
            "company": company_name,
            "quarter": period,
            "call_date": call_dt.strftime("%Y-%m-%d") if call_dt else None,
            "sort_key": sort_key,
            "pdf_path": str(pdf_path),
            "filename": pdf_path.name,
            "file_hash": file_hash,
            "is_stub": is_stub,
            "reason": stub_reason if is_stub else None,
            "page_count": page_count,
            "char_count": len(text),
            "text": text,
            "sentences": sentences,
        }
        valid_transcripts.append(record)
        write_cached_transcript(company_name, file_hash, record)


    # Sort valid transcripts strictly chronologically
    valid_transcripts.sort(key=lambda x: x["sort_key"])

    return {
        "company": company_name,
        "valid_transcripts": valid_transcripts,
        "flagged_files": flagged_files,
        "duplicates": duplicates,
        "total_files": len(pdf_files),
        "loaded_quarters": [t["quarter"] for t in valid_transcripts],
    }
