"""
test_claim_tracker.py
---------------------
Unit tests for the claim extraction, cross-quarter evolution tracking,
and 3D Claim Time Machine data generation engine.
"""
import pytest
import claim_tracker


@pytest.fixture
def sample_transcripts():
    return [
        {
            "company": "TCS",
            "quarter": "Q1 FY25",
            "sentences": [
                {
                    "text": "We expect operating margins to remain between 26% and 28% driven by execution discipline.",
                    "speaker": "Chief Financial Officer",
                    "role": "Management",
                    "section": "Prepared Remarks",
                    "topic": "Margins and Costs",
                    "stance": "Confident",
                    "tone_score": 0.45,
                },
                {
                    "text": "We are targeting double-digit growth in cloud migration and AI engagements over the year.",
                    "speaker": "Chief Executive Officer",
                    "role": "Management",
                    "section": "Prepared Remarks",
                    "topic": "Demand and Pipeline",
                    "stance": "Confident",
                    "tone_score": 0.50,
                },
            ],
        },
        {
            "company": "TCS",
            "quarter": "Q2 FY25",
            "sentences": [
                {
                    "text": "We remain on track to maintain operating margins around 26% to 28% as planned.",
                    "speaker": "Chief Financial Officer",
                    "role": "Management",
                    "section": "Prepared Remarks",
                    "topic": "Margins and Costs",
                    "stance": "Confident",
                    "tone_score": 0.40,
                },
                {
                    "text": "Due to macro uncertainty and headwinds, cloud migration decision cycles have extended and softened.",
                    "speaker": "Chief Executive Officer",
                    "role": "Management",
                    "section": "Q&A",
                    "topic": "Demand and Pipeline",
                    "stance": "Cautious",
                    "tone_score": -0.10,
                },
            ],
        },
        {
            "company": "TCS",
            "quarter": "Q3 FY25",
            "sentences": [
                {
                    "text": "We have lowered guidance and revised down operating margin targets to 24% due to unforeseen cost inflation.",
                    "speaker": "Chief Financial Officer",
                    "role": "Management",
                    "section": "Prepared Remarks",
                    "topic": "Margins and Costs",
                    "stance": "Cautious",
                    "tone_score": -0.30,
                },
            ],
        },
    ]


def test_claim_extraction(sample_transcripts):
    claims = claim_tracker.extract_claims_from_transcript(sample_transcripts[0], quarter_idx=0)
    assert len(claims) >= 1
    assert claims[0]["company"] == "TCS"
    assert claims[0]["quarter"] == "Q1 FY25"
    assert claims[0]["topic"] in claim_tracker.TOPIC_NAMES


def test_claim_evolution_status_classification(sample_transcripts):
    claims_q1 = claim_tracker.extract_claims_from_transcript(sample_transcripts[0], quarter_idx=0)
    claims_q2 = claim_tracker.extract_claims_from_transcript(sample_transcripts[1], quarter_idx=1)
    claims_q3 = claim_tracker.extract_claims_from_transcript(sample_transcripts[2], quarter_idx=2)

    resolved = claim_tracker.match_claims_across_quarters([claims_q1, claims_q2, claims_q3])

    statuses = [c["status"] for c in resolved]
    assert "new" in statuses
    # Check that weakened and/or reaffirmed claims were recognized
    assert any(s in ("reaffirmed", "weakened") for s in statuses)
    assert any(s in ("weakened", "contradiction_candidate") for s in statuses)


def test_3d_claim_time_machine_dataset_structure(sample_transcripts):
    data = claim_tracker.build_claim_time_machine_dataset(sample_transcripts)
    assert "nodes" in data
    assert "links" in data
    assert "quarters" in data
    assert "topics" in data
    assert "summary" in data

    assert len(data["quarters"]) == 3
    assert len(data["nodes"]) >= 3
    for n in data["nodes"]:
        assert "x" in n
        assert "y" in n
        assert "z" in n
        assert "color" in n
        assert "status_label" in n


def test_narrative_stress_indicators(sample_transcripts):
    stress_res = claim_tracker.compute_narrative_stress_indicators(sample_transcripts)
    assert "dataframe" in stress_res
    df = stress_res["dataframe"]
    assert not df.empty
    assert "stress_score" in df.columns
    assert "priority" in df.columns
    assert "explanation" in df.columns
