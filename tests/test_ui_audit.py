"""
test_ui_audit.py
----------------
Strict audit verifying that forbidden technique names and technical jargon
appear nowhere in the UI strings, dashboard labels, tooltips, or assistant replies.
"""
from __future__ import annotations

import re
from pathlib import Path
import pytest

BASE_DIR = Path(__file__).resolve().parent.parent

FORBIDDEN_TERMS = [
    r"\bner\b",
    r"\bclassifier\b",
    r"\btf-idf\b",
    r"\blogistic regression\b",
    r"\bnllb\b",
    r"\bmarket basket\b",
    r"\bassociation rules\b",
    r"\blift\b",
    r"\bnlp module\b",
    r"\blab\b",
]


def test_ui_labels_free_of_jargon():
    """Verifies that app.py and ui_components.py do not expose forbidden jargon in UI."""
    files_to_check = [
        BASE_DIR / "app.py",
        BASE_DIR / "ui_components.py",
        BASE_DIR / "voice_assistant.py",
    ]

    violations = []

    for file_path in files_to_check:
        content = file_path.read_text(encoding="utf-8")
        # Split into lines and inspect string literals or markdown blocks
        for line_num, line in enumerate(content.splitlines(), 1):
            # Exclude code comments or variable names like tfidf_vectorizer in backend logic
            clean_line = line.strip()
            if clean_line.startswith("#") or clean_line.startswith('"""') or clean_line.startswith("from ") or clean_line.startswith("import "):
                continue

            # Only check user-facing strings (HTML tags, markdown, st.write, st.markdown, labels)
            if any(ui_kw in clean_line for ui_kw in ["st.", "ui.", "html", "markdown", "kpi_card", "TOUR", "PAGE_HELP", "badge"]):
                for term in FORBIDDEN_TERMS:
                    match = re.search(term, clean_line, re.I)
                    if match:
                        violations.append(f"{file_path.name}:{line_num} -> matches '{match.group(0)}' in: {clean_line}")

    assert len(violations) == 0, f"Found forbidden jargon in UI strings:\n" + "\n".join(violations)


def test_no_promotional_3d_badge():
    """Verifies that no promotional 'New' badge is placed next to 3D visualizations."""
    content = (BASE_DIR / "app.py").read_text(encoding="utf-8")
    for line in content.splitlines():
        if "3D" in line and "badge" in line:
            assert not re.search(r"badge.*new", line, re.I), f"Promotional 'New' badge found beside 3D in: {line}"


def test_exact_voice_assistant_footer_present():
    """Verifies that the exact required line: 'To activate voice assistant, say “Concall”.' is present."""
    ui_content = (BASE_DIR / "ui_components.py").read_text(encoding="utf-8")
    expected_line = "To activate voice assistant, say “Concall”."
    assert expected_line in ui_content, f"Expected footer line '{expected_line}' not found in ui_components.py"

