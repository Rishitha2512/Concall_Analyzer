"""
test_smoke_pages.py
-------------------
Headless smoke test verifying all 11 enterprise pages, chart generation,
and report builders without UI errors.
"""
from __future__ import annotations

from pathlib import Path
import pytest
import pandas as pd

import data_loader
import entity_extractor
import peer_analysis
import qa_engine
import reports
import topic_model
import ui_components as ui


@pytest.fixture(scope="module")
def loaded_pipeline():
    res = data_loader.load_company_data("TCS", use_cache=True)
    vts = res["valid_transcripts"]
    engine = topic_model.get_topic_engine()
    classified_vts = engine.classify_transcripts(vts)
    scorecards = [entity_extractor.build_quarter_scorecard(t) for t in classified_vts]
    qoq_df = entity_extractor.build_qoq_change_table(scorecards)
    topic_df = topic_model.compute_topic_distribution(classified_vts)
    return {
        "transcripts": classified_vts,
        "scorecards": scorecards,
        "qoq_df": qoq_df,
        "topic_df": topic_df,
    }


def test_smoke_page_1_command_center(loaded_pipeline):
    """Smoke test: Command Center metrics, gauge, 3D Claim Time Machine, and trend chart."""
    vts = loaded_pipeline["transcripts"]
    gauge_fig = ui.build_tone_gauge(0.24, "Q1 FY26")
    trend_fig = ui.build_tone_trend_chart([{"quarter": t["quarter"], "management_tone": {"score": 0.2}} for t in vts])
    vol_fig = ui.build_signal_volume_bar_chart(vts)
    assert gauge_fig is not None
    assert trend_fig is not None
    assert vol_fig is not None

    # Verify 3D Claim Time Machine generation with pipeline data
    import claim_tracker
    import visualization_3d as v3d
    claim_dataset = claim_tracker.build_claim_time_machine_dataset(vts)
    fig_3d = v3d.build_3d_claim_time_machine(claim_dataset)
    assert fig_3d is not None
    assert len(fig_3d.data) >= 1



def test_smoke_page_2_alerts_watchlist(loaded_pipeline):
    """Smoke test: Alerts & Watchlist topic alert detection."""
    topic_df = loaded_pipeline["topic_df"]
    alerts = topic_model.detect_topic_alerts(topic_df)
    assert isinstance(alerts, list)


def test_smoke_page_3_call_timeline(loaded_pipeline):
    """Smoke test: Call Timeline chronological sequences."""
    vts = loaded_pipeline["transcripts"]
    assert len(vts) >= 2
    for t in vts:
        assert "quarter" in t
        assert len(t.get("sentences", [])) > 0


def test_smoke_page_4_topic_radar(loaded_pipeline):
    """Smoke test: Topic Radar and Reliability metrics."""
    topic_df = loaded_pipeline["topic_df"]
    radar_fig = ui.build_topic_radar_chart(topic_df, ["Q1 FY26", "Q4 FY26"])
    assert radar_fig is not None
    eval_metrics = topic_model.get_model_evaluation()
    cm_fig = ui.build_confusion_matrix_chart(eval_metrics["confusion_matrix"], eval_metrics["classes"])
    assert cm_fig is not None


def test_smoke_page_5_key_numbers(loaded_pipeline):
    """Smoke test: Key Numbers Scorecard and QoQ trends."""
    qoq_df = loaded_pipeline["qoq_df"]
    assert not qoq_df.empty
    bar_fig = ui.build_scorecard_metric_trend(qoq_df, "Operating Margin")
    assert bar_fig is not None


def test_smoke_page_6_peer_benchmark(loaded_pipeline):
    """Smoke test: Peer Benchmark multi-company comparison and network graph."""
    res = peer_analysis.compare_peer_companies(["TCS", "INFOSYS"])
    assert not res["summary_table"].empty
    co_engine = peer_analysis.get_cooccurrence_engine()
    net_fig = peer_analysis.build_peer_linkage_network(co_engine.get_rules())
    assert net_fig is not None


def test_smoke_page_7_ask_the_call(loaded_pipeline):
    """Smoke test: Ask the Call cited retrieval and comparison queries."""
    vts = loaded_pipeline["transcripts"]
    engine = qa_engine.get_qa_engine(vts)
    res = engine.answer_query("What were the operating margins in Q1 FY26?", current_company="TCS")
    assert "answer" in res
    assert len(res["citations"]) > 0


def test_smoke_page_8_language_desk():
    """Smoke test: Language Desk writing system analysis."""
    import multilingual
    prof = multilingual.analyze_script_profile("Hello world from quarterly earnings call.")
    assert prof["primary_script"] == "Latin / English"


def test_smoke_page_9_evidence_explorer(loaded_pipeline):
    """Smoke test: Evidence Explorer filtering."""
    vts = loaded_pipeline["transcripts"]
    first = vts[0]
    mgmt_sents = [s for s in first.get("sentences", []) if s.get("speaker_role") == "Management"]
    assert len(mgmt_sents) > 0


def test_smoke_page_10_compare_quarters(loaded_pipeline):
    """Smoke test: Compare Quarters diff."""
    vts = loaded_pipeline["transcripts"]
    assert len(vts) >= 2
    q1 = vts[0]["quarter"]
    q2 = vts[1]["quarter"]
    assert q1 != q2


def test_smoke_page_11_reports(loaded_pipeline, tmp_path):
    """Smoke test: Excel, CSV, JSON, and HTML report export."""
    vts = loaded_pipeline["transcripts"]
    qoq_df = loaded_pipeline["qoq_df"]
    topic_df = loaded_pipeline["topic_df"]
    co_engine = peer_analysis.get_cooccurrence_engine()

    xl_file = tmp_path / "test_report.xlsx"
    reports.generate_multisheet_excel("TCS", vts, qoq_df, topic_df, co_engine.get_rules(), xl_file)
    assert xl_file.exists()

    csv_file = tmp_path / "test_monitor.csv"
    reports.generate_csv_monitor(vts, csv_file)
    assert csv_file.exists()

    json_file = tmp_path / "test_data.json"
    reports.generate_json_export("TCS", vts, qoq_df, json_file)
    assert json_file.exists()

    html_briefing = reports.generate_html_briefing("TCS", vts[0], None, ["Insight 1"])
    assert "TCS" in html_briefing
    assert "Quarterly Analyst Briefing" in html_briefing
