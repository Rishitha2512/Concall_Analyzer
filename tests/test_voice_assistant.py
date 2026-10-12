"""
test_voice_assistant.py
------------------------
Unit tests verifying voice assistant wake-word processing, phonetic variants,
explicit "Concall Wokeup" acknowledgment, and Web Speech listener HTML generation.
"""
from __future__ import annotations

import pytest
import voice_assistant as va


def test_wake_word_alone_triggers_wokeup():
    """Verifies that saying 'Concall' or phonetic variants responds with 'Concall Wokeup'."""
    engine = va.VoiceAssistantEngine()

    variants = [
        "concall",
        "Concall",
        "con call",
        "Con Call",
        "hey concall",
        "can call",
        "corn call",
        "call call",
    ]

    for variant in variants:
        res = engine.process_query(variant, current_page="Command Center", current_company="TCS")
        assert "Concall Wokeup" in res["text"], f"Failed for variant: {variant}"
        assert "Concall Wokeup" in res["speech"], f"Failed speech for variant: {variant}"


def test_wake_word_with_navigation_command():
    """Verifies that saying 'Concall open timeline' responds with 'Concall Wokeup' and navigates."""
    engine = va.VoiceAssistantEngine()

    res = engine.process_query("Concall open timeline", current_page="Command Center", current_company="TCS")
    assert "Concall Wokeup" in res["text"]
    assert "Navigating to Call Timeline" in res["text"]
    assert res["navigate_to"] == "Call Timeline"


def test_wake_word_with_claims_query():
    """Verifies that asking changed claims via wake word includes 'Concall Wokeup'."""
    engine = va.VoiceAssistantEngine()
    fake_claim_data = {
        "nodes": [
            {
                "id": "c1",
                "quarter": "Q1 FY26",
                "topic": "Margins and Costs",
                "text": "Operating margin expansion will be constrained by near-term investments.",
                "status": "weakened",
                "status_reason": "Tone softened compared to prior quarter",
            }
        ]
    }

    res = engine.process_query(
        "Concall which claims changed",
        current_page="Command Center",
        current_company="TCS",
        claim_data=fake_claim_data,
    )
    assert "Concall Wokeup" in res["text"]
    assert "Identified 1 claims" in res["text"]
    assert res["navigate_to"] == "Call Timeline"


def test_listener_html_generation():
    """Verifies that the generated listener contains wake regex, parent dispatch, acoustic mute, and tour guide."""
    html = va.generate_global_wake_word_listener(company="TCS", current_page="Command Center")
    assert "SpeechRecognition" in html
    assert "WAKE_REGEX" in html
    assert "Concall Wokeup" in html
    assert "playWakeChime" in html
    assert "playSleepChime" in html
    assert "speakWakeUp" in html
    assert "concallTriggerManualWake" in html
    assert "isAssistantSpeaking" in html
    assert "echoGuardUntil" in html
    assert "startGuidedTour" in html
    assert "navigateToPage" in html
    assert "TOUR_STEPS" in html


def test_tour_guide_command_triggers_command_center_navigation():
    """Verifies that asking for a tour guide sets navigate_to to Command Center."""
    engine = va.VoiceAssistantEngine()
    res = engine.process_query("Concall start tour", current_page="Alerts & Watchlist", current_company="TCS")
    assert "Concall Wokeup" in res["text"]
    assert res["navigate_to"] == "Command Center"
    assert "tour guide" in res["speech"].lower() or "walkthrough" in res["speech"].lower()

