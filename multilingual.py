"""
multilingual.py
---------------
Multilingual text intelligence, writing system / script detection,
and cross-lingual document ingestion engine for ConcallIQ.
Supports automated language identification, script distribution profiling,
and bidirectional translation of user-uploaded transcripts.
"""
from __future__ import annotations

import io
import logging
import re
from typing import Any, Dict, List, Tuple

import pandas as pd
from langdetect import DetectorFactory, detect_langs

from translator import LANGUAGE_CODES, translate_passage

logger = logging.getLogger(__name__)

# Enforce deterministic language detection
DetectorFactory.seed = 42

SCRIPT_RANGES = {
    "Latin / English": (0x0041, 0x007A),
    "Devanagari / Hindi / Marathi": (0x0900, 0x097F),
    "Tamil": (0x0B80, 0x0BFF),
    "Telugu": (0x0C00, 0x0C7F),
    "Kannada": (0x0C80, 0x0CFF),
    "Malayalam": (0x0D00, 0x0D7F),
    "Bengali": (0x0980, 0x09FF),
    "Gujarati": (0x0A80, 0x0AFF),
    "Gurmukhi / Punjabi": (0x0A00, 0x0A7F),
    "Arabic": (0x0600, 0x06FF),
    "East Asian (Han / Hangul / Kana)": (0x4E00, 0x9FFF),
}

ISO_TO_LANGUAGE = {
    "en": "English",
    "hi": "Hindi",
    "ta": "Tamil",
    "te": "Telugu",
    "kn": "Kannada",
    "ml": "Malayalam",
    "bn": "Bengali",
    "mr": "Marathi",
    "gu": "Gujarati",
    "pa": "Punjabi",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "pt": "Portuguese",
    "zh-cn": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "ar": "Arabic",
}


def analyze_script_profile(text: str) -> Dict[str, Any]:
    """
    Analyzes Unicode character ranges to determine the distribution of
    writing systems present in the transcript.
    """
    sample = text[:50000] if text else ""
    counts = {script: 0 for script in SCRIPT_RANGES}

    for ch in sample:
        code = ord(ch)
        for script, (low, high) in SCRIPT_RANGES.items():
            if low <= code <= high:
                counts[script] += 1
                break

    total_matched = sum(counts.values())
    primary_script = "Latin / English"
    if total_matched > 0:
        primary_script = max(counts, key=counts.get)

    words = re.findall(r"\b[\w'-]+\b", sample)
    distinct_words = len(set(w.lower() for w in words))

    return {
        "primary_script": primary_script,
        "script_counts": counts,
        "total_characters": len(sample),
        "word_count": len(words),
        "unique_words": distinct_words,
    }


def detect_document_language(text: str) -> Tuple[str, float]:
    """
    Detects language with confidence using langdetect.
    Returns (Language_Name, Confidence_Probability).
    """
    sample = text[:15000].strip() if text else ""
    if not sample or len(sample) < 20:
        return "English", 1.0

    try:
        langs = detect_langs(sample)
        if langs:
            top = langs[0]
            lang_name = ISO_TO_LANGUAGE.get(top.lang, "English")
            return lang_name, round(float(top.prob), 3)
    except Exception as exc:
        logger.debug("Language detection fallback: %s", exc)

    # Fallback to script inspection
    profile = analyze_script_profile(sample)
    script = profile["primary_script"]
    if "Devanagari" in script:
        return "Hindi", 0.95
    elif "Tamil" in script:
        return "Tamil", 0.95
    elif "Telugu" in script:
        return "Telugu", 0.95
    elif "Kannada" in script:
        return "Kannada", 0.95
    elif "Malayalam" in script:
        return "Malayalam", 0.95
    elif "Bengali" in script:
        return "Bengali", 0.95
    elif "Gujarati" in script:
        return "Gujarati", 0.95

    return "English", 1.0


def ingest_user_transcript(
    file_bytes: bytes,
    file_name: str,
) -> Dict[str, Any]:
    """
    Ingests a user-uploaded PDF or TXT transcript in any language.
    1. Extracts raw text.
    2. Detects writing script and language.
    3. If non-English, translates to English for analysis while retaining
       the original text alongside.
    4. Generates source language badge & quality note.
    """
    raw_text = ""
    low_name = file_name.lower()

    if low_name.endswith(".pdf"):
        try:
            import pymupdf  # type: ignore
            doc = pymupdf.open(stream=file_bytes, filetype="pdf")
            pages = [doc[i].get_text() for i in range(len(doc))]
            raw_text = "\n".join(pages)
        except Exception:
            from pypdf import PdfReader  # type: ignore
            reader = PdfReader(io.BytesIO(file_bytes))
            raw_text = "\n".join(p.extract_text() or "" for p in reader.pages)
    else:
        raw_text = file_bytes.decode("utf-8", errors="replace")

    raw_text = raw_text.strip()
    script_prof = analyze_script_profile(raw_text)
    lang_name, confidence = detect_document_language(raw_text)

    is_multilingual = lang_name != "English"
    english_text = raw_text
    quality_note = "Original English corporate transcript ready for analysis."

    if is_multilingual:
        quality_note = (
            f"Detected source language: {lang_name} ({script_prof['primary_script']}) "
            f"with {confidence*100:.1f}% confidence. Sentence-batched translation applied "
            f"for intelligence extraction while keeping the original text alongside."
        )
        try:
            english_text = translate_passage(
                raw_text[:8000],
                source_language=lang_name,
                target_language="English",
            )
        except Exception as exc:
            logger.warning("Translation error on upload: %s", exc)
            english_text = raw_text
            quality_note += f" (Translation engine alert: {exc})"

    return {
        "file_name": file_name,
        "detected_language": lang_name,
        "confidence": confidence,
        "primary_script": script_prof["primary_script"],
        "script_distribution": script_prof["script_counts"],
        "word_count": script_prof["word_count"],
        "original_text": raw_text,
        "english_text": english_text,
        "is_translated": is_multilingual,
        "quality_note": quality_note,
        "badge_text": f"{lang_name} ({script_prof['primary_script'].split('/')[0].strip()}) · {confidence*100:.0f}%",
    }
