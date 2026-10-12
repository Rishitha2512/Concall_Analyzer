"""
test_visualization_3d.py
------------------------
Unit tests for the 3D visualization builders:
- The 3D Claim Time Machine
- The 2D Claim Timeline Fallback
- The 3D Corporate Narrative Galaxy
- The 3D Narrative Stress Map
- The 3D Financial Peer Space
- Audio & Narrative Exploration View
"""
import pandas as pd
import pytest
import plotly.graph_objects as go

import claim_tracker
import visualization_3d as v3d


@pytest.fixture
def mock_claim_data():
    sample_transcripts = [
        {
            "company": "Infosys",
            "quarter": "Q1 FY26",
            "sentences": [
                {
                    "text": "We expect operating margins between 20% and 22% for the fiscal year.",
                    "speaker": "CFO",
                    "role": "Management",
                    "section": "Prepared Remarks",
                    "topic": "Margins and Costs",
                    "stance": "Confident",
                    "tone_score": 0.35,
                }
            ],
        },
        {
            "company": "Infosys",
            "quarter": "Q2 FY26",
            "sentences": [
                {
                    "text": "Operating margins remain on track between 20% and 22%.",
                    "speaker": "CFO",
                    "role": "Management",
                    "section": "Prepared Remarks",
                    "topic": "Margins and Costs",
                    "stance": "Confident",
                    "tone_score": 0.35,
                }
            ],
        },
    ]
    return claim_tracker.build_claim_time_machine_dataset(sample_transcripts)


def test_build_3d_claim_time_machine(mock_claim_data):
    fig = v3d.build_3d_claim_time_machine(mock_claim_data)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 1
    # Check scene configuration
    assert "scene" in fig.layout
    assert "xaxis" in fig.layout.scene


def test_build_2d_claim_timeline(mock_claim_data):
    fig = v3d.build_2d_claim_timeline(mock_claim_data)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 1


def test_build_3d_narrative_galaxy():
    topics = ["Demand and Pipeline", "Margins and Costs", "AI and Digital"]
    techs = ["GenAI", "Cloud", "S/4HANA"]
    geos = ["Americas", "Europe", "APAC"]
    rules = [{"antecedent": "GenAI", "consequent": "Margins and Costs", "support": 0.2, "confidence": 0.8, "lift": 1.4}]

    fig = v3d.build_3d_narrative_galaxy("Infosys", topics, techs, geos, rules)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 1
    assert "scene" in fig.layout


def test_build_3d_narrative_stress_map():
    df = pd.DataFrame([
        {
            "quarter": "Q1 FY26",
            "quarter_idx": 0,
            "topic": "Margins and Costs",
            "topic_idx": 1,
            "stress_score": 6.5,
            "priority": "High Priority",
            "color": "#ff7a8a",
            "explanation": "High margin pressure.",
            "weakened_count": 1,
            "contradiction_count": 1,
        }
    ])
    stress_data = {
        "dataframe": df,
        "quarters": ["Q1 FY26"],
        "topics": ["Margins and Costs"],
    }
    fig = v3d.build_3d_narrative_stress_map(stress_data)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 1


def test_build_3d_financial_peer_space():
    peer_df = pd.DataFrame([
        {"Company": "TCS", "Quarter": "Q1", "Revenue Growth": "3.5%", "Operating Margin": "26.0%", "Revenue Growth Num": 3.5, "Operating Margin Num": 26.0, "Net Profit Num": 15.0},
        {"Company": "Infosys", "Quarter": "Q1", "Revenue Growth": "2.5%", "Operating Margin": "21.1%", "Revenue Growth Num": 2.5, "Operating Margin Num": 21.1, "Net Profit Num": 12.0},
    ])
    fig = v3d.build_3d_financial_peer_space(peer_df)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 1


def test_build_audio_narrative_chart():
    audio_meta = {
        "speaker_turns": [
            {"turn_index": 1, "duration_seconds": 45, "speaker_name": "CEO", "speaker_role": "Management", "section": "Remarks", "topic": "Demand", "snippet": "Strong growth."},
            {"turn_index": 2, "duration_seconds": 30, "speaker_name": "Analyst", "speaker_role": "Analyst", "section": "Q&A", "topic": "Margins", "snippet": "Question on margins."},
        ]
    }
    fig = v3d.build_audio_narrative_chart(audio_meta)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 1
