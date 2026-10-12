"""
text_filters.py
---------------
Shared clean-up helpers so that PDF boilerplate (page markers, participant lists,
legal disclaimers, exchange letters) does not leak into summaries, risk lists,
commitments, entity extraction or question answering.
"""
from __future__ import annotations

import re

_PAGE = re.compile(r"-{2,}\s*Page\s*\d+\s*-{2,}", re.I)

BOILERPLATE_MARKERS = (
    "©", "external document", "scrip code", "regulation 30", "listing obligations", "exchange plaza",
    "listen-only", "listen only", "participant lines", "being recorded", "safe harbor", "safe harbour",
    "forward-looking statements", "forward looking statements", "dear sirs", "transcript of the",
    "disclosure requirements", "corporate participants", "kindly take the same on record",
    "webcast", "anything said on this call", "reflects our outlook", "archive including the transcript",
    "don't provide any specific", "do not provide any specific", "moderator", "next question", "thank you for joining",
)


def strip_markers(text: str) -> str:
    """Remove '--- Page N ---' markers left by the PDF extractor."""
    return _PAGE.sub(" ", text or "")


def is_boilerplate(sentence: str) -> bool:
    low = sentence.lower()
    if any(m in low for m in BOILERPLATE_MARKERS):
        return True
    words = sentence.split()
    # lists of names / organisations (mostly capitalised words) are not management statements
    if len(words) >= 6 and sum(w[:1].isupper() for w in words) / len(words) > 0.6:
        return True
    return False
