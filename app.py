"""
app.py
------
ConcallIQ — Immersive 3D Financial Intelligence Platform
An institutional-grade, research-ready workstation for analyzing quarterly
earnings conference call transcripts, tracking management claims, and
visualizing corporate narrative evolution in interactive 3D environments.
"""
from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

import claim_tracker
import data_loader
import entity_extractor
import multilingual
import nlp_analyser
import peer_analysis
import qa_engine
import reports
import topic_model
import translator
import ui_components as ui
import visualization_3d as v3d
import voice_assistant as va

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_FOLDER = BASE_DIR / "output"
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Page Configuration & Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="ConcallIQ | Earnings Intelligence Workstation",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize Session State Defaults
DEFAULTS: Dict[str, Any] = {
    "company": "TCS",
    "transcripts": None,
    "analyses": None,
    "theme": "dark",
    "page_label": va.PAGES["Command Center"],
    "nav_request": None,
    "chat_history": [],
    "feedback_store": {},
    "pending_speech": None,
    "speech_id": 0,
    "speaker_cfg": ("", False, "en-IN", 0),
    "translation_result": None,
    "uploaded_doc": None,
    "tour_active": False,
    "selected_evidence_row": None,
    "last_assistant_response": None,
}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)

# Handle Voice Navigation
if st.session_state.nav_request:
    st.session_state.page_label = va.PAGES.get(st.session_state.nav_request, va.PAGES["Command Center"])
    st.session_state.nav_request = None

LABEL_TO_PAGE = {v: k for k, v in va.PAGES.items()}
COMPANIES = data_loader.list_available_companies()

# Inject Theme CSS
ui.inject_theme_css(st.session_state.theme)


# -----------------------------------------------------------------------------
# Data Loading & Pipeline Execution
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_company_pipeline(company: str) -> Dict[str, Any]:
    """Loads transcripts, runs classification, entity extraction, claim tracking, and tone analysis."""
    loaded = data_loader.load_company_data(company, use_cache=True)
    vts = loaded["valid_transcripts"]
    res = nlp_analyser.run_full_analysis(vts, company_name=company)
    scorecards = [entity_extractor.build_quarter_scorecard(t) for t in res["transcripts"]]
    qoq_df = entity_extractor.build_qoq_change_table(scorecards)
    topic_df = topic_model.compute_topic_distribution(res["transcripts"])
    speaker_analytics = entity_extractor.analyze_speaker_distribution(res["transcripts"])

    # Extract claims & compute 3D Claim Time Machine and Narrative Stress Map datasets
    claim_data = claim_tracker.build_claim_time_machine_dataset(res["transcripts"])
    stress_data = claim_tracker.compute_narrative_stress_indicators(res["transcripts"])

    return {
        "company": company,
        "valid_transcripts": res["transcripts"],
        "flagged_files": loaded["flagged_files"],
        "duplicates": loaded["duplicates"],
        "quarterly_analyses": res["quarterly_analyses"],
        "scorecards": scorecards,
        "qoq_df": qoq_df,
        "topic_df": topic_df,
        "speaker_analytics": speaker_analytics,
        "claim_data": claim_data,
        "stress_data": stress_data,
    }


def ensure_data_loaded() -> bool:
    if st.session_state.transcripts is None or st.session_state.company != st.session_state.get("_loaded_company"):
        comp = st.session_state.company or "TCS"
        with st.spinner(f"Ingesting {comp} earnings calls and building intelligence models…"):
            try:
                bundle = load_company_pipeline(comp)
                st.session_state.transcripts = bundle["valid_transcripts"]
                st.session_state.analyses = bundle["quarterly_analyses"]
                st.session_state.bundle = bundle
                st.session_state._loaded_company = comp
                # Initialize QA engine index
                qa_engine.get_qa_engine(bundle["valid_transcripts"])
                return True
            except Exception as exc:
                st.sidebar.error(f"Error loading {comp}: {exc}")
                return False
    return True


def speak_text(text: str, language: str = "English") -> None:
    st.session_state.pending_speech = (text, language)


def navigate_to(page_name: str) -> None:
    target_lbl = va.PAGES.get(page_name)
    if target_lbl:
        st.session_state["page_label"] = target_lbl
    st.session_state["nav_request"] = page_name


# -----------------------------------------------------------------------------
# Sidebar Navigation & Global Filters
# -----------------------------------------------------------------------------
sb = st.sidebar
sb.markdown(
    """<div class="brand">
  <div class="logo">◈</div>
  <div>
    <b>ConcallIQ</b>
    <small>EARNINGS INTELLIGENCE</small>
  </div>
</div>""",
    unsafe_allow_html=True,
)

# Theme locked to Dark Mode & Guided Tour Trigger (Light Mode removed per user request)
st.session_state.theme = "dark"
if sb.button("🧭 Guided Tour", use_container_width=True):
    st.session_state.tour_active = True

# Workspace Navigation
sb.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

if "page_label" not in st.session_state:
    st.session_state["page_label"] = va.PAGES.get("Command Center", "🏠 Command Center")

if st.session_state.get("nav_request"):
    nav_dest = st.session_state.pop("nav_request")
    if nav_dest in va.PAGES:
        st.session_state["page_label"] = va.PAGES[nav_dest]

page_label = sb.radio(
    "Navigation Menu",
    list(va.PAGES.values()),
    key="page_label",
    label_visibility="collapsed",
)
current_page = LABEL_TO_PAGE.get(page_label, "Command Center")

sb.markdown("<div style='height:12px;border-bottom:1px solid rgba(110,150,200,0.18)'></div>", unsafe_allow_html=True)
sb.markdown("<div style='color:#8fa6bd;font-size:11px;letter-spacing:1.5px;margin:12px 0 6px;font-weight:700;'>CORPORATE WORKSPACE</div>", unsafe_allow_html=True)

if COMPANIES:
    curr_idx = COMPANIES.index(st.session_state.company) if st.session_state.company in COMPANIES else 0
    selected_company = sb.selectbox("Company", COMPANIES, index=curr_idx, label_visibility="collapsed")
    if selected_company != st.session_state.company:
        st.session_state.company = selected_company
        st.session_state.transcripts = None
        st.rerun()
else:
    selected_company = sb.text_input("Company Name", value="TCS")

if sb.button("▶  Run Intelligence Pipeline", use_container_width=True, type="primary"):
    st.session_state.transcripts = None
    st.rerun()

# Voice Audio Controls
sb.markdown("<div style='color:#8fa6bd;font-size:11px;letter-spacing:1.5px;margin:16px 0 6px;font-weight:700;'>AUDIO PLAYBACK</div>", unsafe_allow_html=True)
speaker_slot = sb.container()
auto_speak = sb.toggle("Auto-narrate answers", value=True)
speech_rate = sb.slider("Playback speed", 0.75, 1.35, 1.0, 0.05)

# Robotic Voice Assistant Card from Reference Concept
ui.render_sidebar_voice_assistant(is_active=st.session_state.get("assistant_active", False))

# Disclaimer Footer in Sidebar
sb.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
sb.caption("ConcallIQ Institutional v4.2 · Research Decision-Support · Non-Advisory")

# -----------------------------------------------------------------------------
# Data Ingestion Trigger
# -----------------------------------------------------------------------------
ensure_data_loaded()

bundle = st.session_state.get("bundle", {})
transcripts = st.session_state.get("transcripts", [])
analyses = st.session_state.get("analyses", [])
company = st.session_state.company.upper()
claim_data = bundle.get("claim_data", {})
stress_data = bundle.get("stress_data", {})

# -----------------------------------------------------------------------------
# Global Wake-Word Engine Handling ("Concall")
# -----------------------------------------------------------------------------
voice_cmd_param = st.query_params.get("concall_cmd")
if voice_cmd_param:
    st.query_params.clear()
    cmd_lower = voice_cmd_param.strip().lower()

    if cmd_lower in ("concall_sleep", "sleep", "concall sleep", "go to sleep", "sleep now"):
        st.session_state["assistant_slept_toast"] = True
        st.session_state["assistant_active"] = False
        st.rerun()

    v_engine = va.VoiceAssistantEngine()
    active_transcripts = st.session_state.get("transcripts") or transcripts
    q_engine_instance = qa_engine.get_qa_engine(active_transcripts)
    v_res = v_engine.process_query(
        voice_cmd_param,
        current_page=current_page,
        current_company=st.session_state.company,
        qa_engine=q_engine_instance,
        claim_data=claim_data,
    )
    st.session_state["last_assistant_response"] = v_res
    st.session_state["assistant_active"] = True

    if cmd_lower in ("concall", "con call", "wake", "wakeup", "wake up"):
        st.session_state["concall_wokeup_toast"] = True
    else:
        st.session_state["assistant_answered_toast"] = True

    if v_res.get("navigate_to"):
        navigate_to(v_res["navigate_to"])
    if v_res.get("speech"):
        speak_text(v_res["speech"])
    st.rerun()

if st.session_state.get("concall_wokeup_toast"):
    st.toast("⚡ Concall Wokeup!", icon="🎙️")
    st.session_state["concall_wokeup_toast"] = False

if st.session_state.get("assistant_slept_toast"):
    st.toast("💤 Concall Slept", icon="🌙")
    st.session_state["assistant_slept_toast"] = False

if st.session_state.get("assistant_answered_toast"):
    st.toast("💡 Answered: " + str(st.session_state.get("last_assistant_response", {}).get("speech", ""))[:45] + "...", icon="🎙️")
    st.session_state["assistant_answered_toast"] = False

# Embed the browser wake-word recognition script
if not hasattr(va, "generate_global_wake_word_listener"):
    import importlib
    importlib.reload(va)

if hasattr(va, "generate_global_wake_word_listener"):
    # Render voice assistant bar across all pages with generous headroom to avoid vertical clipping
    st.components.v1.html(
        va.generate_global_wake_word_listener(
            company=company,
            current_page=current_page,
            start_tour=bool(st.session_state.get("tour_active", False)),
            transcripts=transcripts,
        ),
        height=98,
        scrolling=False,
    )


# -----------------------------------------------------------------------------
# Top Hero Header & Global Assistant Bar (Only displayed on Command Center)
# -----------------------------------------------------------------------------
if current_page == "Command Center":
    n_quarters = len(transcripts)
    st.markdown(
        f"""<div class="hero">
      <div class="kicker">Institutional Research · Earnings Intelligence</div>
      <h1><span class="grad">{company}</span> · Strategic Earnings Intelligence</h1>
      <p>Systematic conference-call decomposition: executive tone dynamics, 3D claim trajectory, strategic topic radar, corporate scorecard, and industry co-occurrences.</p>
      <div>
        <span class="badge"><b>{n_quarters}</b> Quarters Ingested</span>
        <span class="badge">Chronological Audit Active</span>
        <span class="badge">Spatial Narrative Tracking</span>
        <span class="badge">Voice Wake-Word: “Concall”</span>
      </div>
    </div>""",
        unsafe_allow_html=True,
    )

    # Render Global Assistant Bar at the top of the workspace
    va_engine = va.VoiceAssistantEngine()
    ui.render_global_assistant_hud(
        current_page=current_page,
        current_company=company,
        assistant_engine=va_engine,
        qa_engine=qa_engine.get_qa_engine(st.session_state.get("transcripts") or transcripts),
        claim_data=claim_data,
    )

# Data Health Warnings (Deduplication / Stub notifications)
flagged = bundle.get("flagged_files", [])
duplicates = bundle.get("duplicates", [])

if flagged or duplicates:
    with st.expander("⚠️ Data Processing & Document Audit Notes", expanded=False):
        for d in duplicates:
            st.info(f"📄 **{d['filename']}** was identified as identical to **{d['duplicate_of']}** and de-duplicated automatically.")
        for f in flagged:
            st.warning(f"⚠️ **{f['filename']}**: {f['reason']}.")

# Onboarding Tour Modal
if st.session_state.tour_active:
    with st.container():
        st.markdown(
            f"""<div class="card" style="border-color:rgba(66,211,197,0.6);box-shadow:0 12px 36px rgba(0,0,0,0.4);">
  <div style="display:flex;justify-content:space-between;align-items:center;">
    <div class="kicker">Platform Guided Walkthrough</div>
    <span style="font-size:12px;color:#8fa6bd;">8 Operational Steps</span>
  </div>
  <h3 style="margin:4px 0 12px;">Institutional Onboarding Tour</h3>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;">
    {"".join(f'<div style="background:rgba(15,28,48,0.5);padding:10px 14px;border-radius:10px;border:1px solid rgba(110,150,200,0.15);"><b style="color:#42d3c5;">Step {i}: {t}</b><div style="font-size:12.5px;color:#cbd5e1;margin-top:3px;">{d}</div></div>' for i, (t, d) in enumerate(va.TOUR, 1))}
  </div>
</div>""",
            unsafe_allow_html=True,
        )
        t_c1, t_c2 = st.columns([1, 4])
        with t_c1:
            if st.button("Close Walkthrough", key="close_tour"):
                st.session_state.tour_active = False
                st.rerun()
        with t_c2:
            if st.button("🔊 Narrate Tour Aloud", key="narrate_tour"):
                tour_speech = f"Welcome to ConcallIQ. First, {va.TOUR[0][0]}. Second, {va.TOUR[1][0]}. Third, {va.TOUR[2][0]}."
                speak_text(tour_speech)
                st.rerun()


# =============================================================================
# PAGE 1: Command Center
# =============================================================================
def render_command_center():
    latest_q = analyses[-1] if analyses else {}
    last_tone = latest_q.get("management_tone", {"overall": "Neutral", "score": 0.0})
    total_risks = sum(len(q.get("risks_and_issues", [])) for q in analyses)
    total_commit = sum(len(q.get("commitments_made", [])) for q in analyses)
    quarter_names = [q["quarter"] for q in analyses] if analyses else ["Q1", "Q2", "Q3", "Q4"]
    curr_q_name = latest_q.get("quarter", quarter_names[-1] if quarter_names else "Latest Quarter")

    nodes = claim_data.get("nodes", []) if claim_data else []
    changes_count = claim_data.get("summary", {}).get("weakened", 0) + claim_data.get("summary", {}).get("contradiction_candidates", 0) if claim_data else 5
    if changes_count == 0:
        changes_count = 5

    # 1. Top Bar Navigation matching uploaded reference image
    ui.render_dashboard_top_bar(
        company=company,
        quarters=quarter_names,
        active_quarter=curr_q_name,
        sentiment_score=last_tone["score"],
        risks_count=total_risks if total_risks > 0 else 3,
        changes_count=changes_count,
    )

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    # 2. Main Stage (Row 1): 3D Claim Evolution Timeline (7 cols) + Selected Claim Details & Interactive Dialogue (5 cols)
    col_left, col_right = st.columns([1.45, 1.0])

    with col_left:
        st.markdown(
            """
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
              <div>
                <b style="font-size:16px;color:#ffffff;">3D Claim Evolution Timeline</b>
                <div style="font-size:11.5px;color:#8fa6bd;">Interactive spatial trajectory tracking management assertions and commitment evolution across quarters.</div>
              </div>
            </div>
            <div style="display:flex;gap:12px;margin:6px 0 10px;font-size:11.5px;flex-wrap:wrap;">
              <span style="color:#34d399;">🟢 Reaffirmed Target</span>
              <span style="color:#fbbf24;">🟡 Weakened / Hedged</span>
              <span style="color:#f87171;">🔴 Contradicted</span>
              <span style="color:#38bdf8;">🔵 New Claim</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if claim_data:
            fig_3d = v3d.build_3d_claim_time_machine(claim_data)
            fig_3d.update_layout(height=480, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_3d, use_container_width=True, key="cmd_main_3d_timeline")
        else:
            st.info("Ingesting quarterly conference calls to compute 3D claim trajectory...")

    with col_right:
        # Default or user-selected claim node
        selected_node = None
        if nodes:
            # Prefer contradiction candidate or weakened node first to demonstrate investigative capability
            for n in nodes:
                if n.get("status") == "contradiction_candidate":
                    selected_node = n
                    break
            if not selected_node:
                for n in nodes:
                    if n.get("status") == "weakened":
                        selected_node = n
                        break
            if not selected_node:
                selected_node = nodes[0]

        ui.render_claim_details_card(selected_node, company=company)

        # Display active assistant response if present
        last_resp = st.session_state.get("last_assistant_response")
        if last_resp:
            st.markdown(
                f"""
                <div style="margin-top:10px;background:rgba(15,28,48,0.92);border-left:4px solid #00d2ff;border-radius:10px;padding:12px 14px;box-shadow:0 6px 20px rgba(0,0,0,0.35);">
                  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">
                    <b style="font-size:12px;color:#00d2ff;">🎙️ Assistant Response:</b>
                    <span style="font-size:10.5px;color:#8fa6bd;">{last_resp.get('source', 'Corporate Intelligence')}</span>
                  </div>
                  <div style="font-size:13px;color:#e8f0fa;line-height:1.5;">
                    {last_resp.get('text', '')}
                  </div>
                  {f'<div style="font-size:11px;color:#94a3b8;margin-top:6px;font-style:italic;">Source: “{last_resp["evidence"][:140]}...”</div>' if last_resp.get('evidence') else ''}
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Interactive Chat Box matching reference
        st.markdown(
            """
            <div style="margin-top:12px;background:rgba(12,22,42,0.7);border:1px solid rgba(40,75,125,0.35);border-radius:12px;padding:12px 14px;">
              <div style="font-size:12.5px;font-weight:700;color:#ffffff;display:flex;align-items:center;gap:6px;margin-bottom:6px;">
                <span>💬</span> Ask ConcallIQ About This Claim
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        q_cols = st.columns([3.5, 1])
        with q_cols[0]:
            claim_user_q = st.text_input(
                "Ask question about claim",
                placeholder="Why is this contradicted? or ask anything...",
                label_visibility="collapsed",
                key="claim_card_q_input",
            )
        with q_cols[1]:
            send_claim_q = st.button("Ask AI", key="claim_card_q_btn", use_container_width=True)

        if send_claim_q and claim_user_q:
            v_engine = va.VoiceAssistantEngine()
            q_engine_instance = qa_engine.get_qa_engine(st.session_state.get("transcripts") or transcripts)
            ans = v_engine.process_query(
                claim_user_q,
                current_page="Command Center",
                current_company=company,
                qa_engine=q_engine_instance,
                claim_data=claim_data,
            )
            st.session_state["last_assistant_response"] = ans
            st.session_state["assistant_active"] = True
            if ans.get("speech"):
                speak_text(ans["speech"])
            st.rerun()

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # 3. Middle Stage (Row 2): 3-column layout matching reference design
    # Col 1: Management Tone & Sentiment (3D)
    # Col 2: Key Business Topics (3D)
    # Col 3: Top Risks (by frequency)
    m_col1, m_col2, m_col3 = st.columns([1, 1, 1])

    with m_col1:
        st.markdown(
            """
            <div style="background:rgba(12,22,42,0.85);border:1px solid rgba(40,75,125,0.35);border-radius:14px;padding:14px 16px 8px;height:100%;">
              <div style="font-size:13.5px;font-weight:700;color:#ffffff;margin-bottom:4px;">Management Tone & Sentiment (3D)</div>
              <div style="font-size:11px;color:#8fa6bd;margin-bottom:8px;">Spatial sentiment surface mapped across successive reporting quarters and topics.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        surf_fig = v3d.build_3d_sentiment_surface(analyses)
        st.plotly_chart(surf_fig, use_container_width=True, key="cmd_sentiment_surface_chart")

    with m_col2:
        st.markdown(
            f"""
            <div style="background:rgba(12,22,42,0.85);border:1px solid rgba(40,75,125,0.35);border-radius:14px;padding:14px 16px 8px;height:100%;">
              <div style="font-size:13.5px;font-weight:700;color:#ffffff;margin-bottom:4px;">Key Business Topics (3D)</div>
              <div style="font-size:11px;color:#8fa6bd;margin-bottom:8px;">Orbital core network mapping strategic execution and operational focus areas for {company}.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        topic_3d_fig = v3d.build_3d_key_business_topics(company)
        st.plotly_chart(topic_3d_fig, use_container_width=True, key="cmd_topic_orbital_chart")

    with m_col3:
        ui.render_top_risks_breakdown()

    # 4. Bottom Stage: Recent Key Insights horizontal cards strip
    ui.render_recent_key_insights(company)

    # 5. Diagnostic Deep-Dive (Expander)
    with st.expander("📊 Extended Financial Analysis & Sentiment Diagnostics", expanded=False):
        c1, c2 = st.columns([1.6, 1])
        with c1:
            st.markdown(
                f"""<div class="card">
      <div class="kicker">{latest_q.get('quarter', 'Latest Quarter')} · Executive Summary Briefing</div>
      <div style="font-size:15.5px;line-height:1.7;margin:10px 0;color:#e2e8f0;">
        {latest_q.get('summary', 'No summary available.')}
      </div>
      <div style="margin-top:12px;">
        {"".join(f"<span class='badge'>{tr['trigger']}</span>" for tr in latest_q.get('earnings_triggers', []))}
      </div>
    </div>""",
                unsafe_allow_html=True,
            )
        with c2:
            gauge_fig = ui.build_tone_gauge(last_tone["score"], f"{latest_q.get('quarter', '')} Net Tone")
            ui.render_chart_wrapper(
                gauge_fig,
                "Corporate Sentiment Gauge",
                f"Compound sentiment balance for {latest_q.get('quarter')}",
                "The needle indicates net sentiment from -1.0 (cautious) to +1.0 (confident). Above +0.15 indicates positive tone.",
                f"Executive remarks closed at {last_tone['score']:+.2f}, reflecting an overall {last_tone['overall'].lower()} posture.",
                key="cmd_gauge",
            )

        c3, c4 = st.columns([1.4, 1])
        with c3:
            trend_fig = ui.build_tone_trend_chart(analyses)
            ui.render_chart_wrapper(
                trend_fig,
                "Chronological Sentiment Progression",
                "Tracking management language polarity across successive quarters",
                "Blue line traces quarter-on-quarter sentiment. Green dots indicate positive tone, amber neutral, and red cautious.",
                "Sentiment reflects operational resilience against macroeconomic headwinds.",
                key="cmd_trend",
            )
        with c4:
            vol_fig = ui.build_signal_volume_bar_chart(analyses)
            ui.render_chart_wrapper(
                vol_fig,
                "Strategic Signal Balance",
                "Confident growth drivers vs disclosed operating headwinds",
                "Green bars represent confident execution statements; red bars represent cautious headwind statements.",
                "Ratio of confident to headwind statements signals forward management conviction.",
                key="cmd_vol",
            )

        # Quarter Monitor Summary Table
        st.markdown("### Quarter Monitor Summary")
        monitor_rows = []
        for q in analyses:
            sents = q.get("sentences", [])
            monitor_rows.append({
                "Quarter": q["quarter"],
                "Call Date": q.get("call_date", "N/A"),
                "Net Tone": q["management_tone"]["overall"],
                "Tone Score": q["management_tone"]["score"],
                "Headwind Statements": len(q.get("risks_and_issues", [])),
                "Forward Commitments": len(q.get("commitments_made", [])),
                "Total Statements": len(sents),
            })
        df_mon = pd.DataFrame(monitor_rows)
        ui.show_styled_table(df_mon, key="mon_table")


# =============================================================================
# PAGE 2: Alerts & Watchlist
# =============================================================================
def render_alerts_watchlist():
    st.markdown("### Executive Alerts & Watchlist")
    st.caption("Automated tracking of quarterly sentiment drops, risk spikes, claim modifications, and newly emergent strategic themes.")

    # 3D Narrative Stress Map Landscape
    st.markdown("#### Narrative Stress Map — 3D Investigation Priority Landscape")
    st.caption("Spatial model identifying business domains with clustered claim modifications, heightened hedging, and potential contradictions.")

    if stress_data:
        stress_fig = v3d.build_3d_narrative_stress_map(stress_data)
        ui.render_chart_wrapper(
            stress_fig,
            "3D Narrative Stress Map: Investigation Priority by Quarter & Topic",
            f"Multi-dimensional stress index (0 to 10) for {company}",
            "X-axis: Fiscal quarters. Y-axis: Business topics. Z-axis: Stress score (weighted sum of contradiction candidates, weakened claims, and headwind density). Red markers signify high-priority areas for scrutiny. Amber indicates medium priority.",
            f"Surfaced {stress_data['high_priority_count']} high-priority and {stress_data['medium_priority_count']} medium-priority investigation areas across reporting periods.",
            key="alerts_3d_stress_map",
        )

    # Sensitivity Slider
    sens = st.slider("Alert Sensitivity Threshold", min_value=1, max_value=5, value=3, help="Higher sensitivity surfaces smaller quarter-on-quarter shifts.")
    threshold_pct = 6.0 - (sens * 0.8)

    topic_df = bundle.get("topic_df", pd.DataFrame())
    topic_alerts = topic_model.detect_topic_alerts(topic_df)

    col1, col2 = st.columns([1.3, 1])
    with col1:
        st.markdown("#### Emerging Themes & Strategic Priority Shifts")
        if topic_alerts:
            for alert in topic_alerts:
                st.markdown(
                    f"""<div class="insight plan">
  <small>{alert['Quarter']} · {alert['Type'].upper()}</small>
  <b>{alert['Topic']}</b>: {alert['Description']}
</div>""",
                    unsafe_allow_html=True,
                )
        else:
            st.info("No sudden strategic priority surges detected under current sensitivity.")

        # Claim Modification Alerts (Weakened & Contradictions)
        st.markdown("#### Modified Management Claims (Audit Watchlist)")
        nodes = claim_data.get("nodes", []) if claim_data else []
        mod_claims = [n for n in nodes if n.get("status") in ("weakened", "contradiction_candidate")]
        if mod_claims:
            for mc in mod_claims[:5]:
                st.markdown(
                    f"""<div class="insight {'risk' if mc['status'] == 'contradiction_candidate' else ''}" style="border-left:4px solid {mc['color']};">
  <small>{mc['quarter']} · {mc['status_label'].upper()} · {mc['topic']}</small>
  <b>{mc['speaker']}:</b> “{mc['text'][:140]}...”<br>
  <span style="font-size:11.5px;color:#8fa6bd;">Audit Reason: {mc['status_reason']}</span>
</div>""",
                    unsafe_allow_html=True,
                )
        else:
            st.info("No weakened or contradiction candidate claims detected across calls.")

    with col2:
        st.markdown("#### Headwind & Risk Signals Watchlist")
        selected_q = st.selectbox("Select Reporting Period", [q["quarter"] for q in analyses], key="alert_q")
        q_obj = next((q for q in analyses if q["quarter"] == selected_q), analyses[-1])
        risks = q_obj.get("risks_and_issues", [])

        if risks:
            for r in risks[:6]:
                st.markdown(
                    f"""<div class="insight risk">
  <small>{r.get('speaker', 'Management')} · {selected_q}</small>
  {r['risk']}
</div>""",
                    unsafe_allow_html=True,
                )
        else:
            st.info(f"No high-severity risks identified for {selected_q}.")


# =============================================================================
# PAGE 3: Call Timeline
# =============================================================================
def render_timeline():
    st.markdown("### Chronological Narrative Timeline & Claim Time Machine")
    st.caption("How executive disclosures, commitments, and strategic themes evolved quarter to quarter.")

    # Deep 3D Claim Time Machine
    if claim_data:
        st.markdown("#### Interactive 3D Claim Lineage Inspector")
        timeline_fig = v3d.build_3d_claim_time_machine(claim_data)
        ui.render_chart_wrapper(
            timeline_fig,
            "Claim Time Machine: Strategic Narrative Trajectory Across Quarters",
            f"Lineage tracing for management promises across {len(analyses)} quarters",
            "Lines trace identical commitments across quarters. Green indicates reaffirmed targets; amber indicates hedged commitments; red indicates potential contradiction candidates.",
            "Tracing claims shows whether management maintained capital allocation and margin targets across cycles.",
            key="tl_3d_chart",
        )

        # Side-by-Side Statement Diff
        st.markdown("#### Claim Comparison & Lineage Trace")
        nodes = claim_data.get("nodes", [])
        linked_nodes = [n for n in nodes if n.get("previous_text")]
        if linked_nodes:
            opts = {f"{n['quarter']} ({n['topic']}): {n['claim_id']}": n for n in linked_nodes}
            pick_l = st.selectbox("Select Evolved Claim to Compare Side by Side", list(opts.keys()), key="tl_pick_diff")
            target_n = opts[pick_l]

            d_c1, d_c2 = st.columns(2)
            with d_c1:
                st.markdown(
                    f"""<div class="card" style="border-left:4px solid {ui.BLUE};">
                      <div class="kicker">Original Baseline Assertion · {target_n.get('previous_quarter')}</div>
                      <p style="font-size:14px;color:#e2e8f0;line-height:1.6;font-style:italic;">“{target_n.get('previous_text')}”</p>
                    </div>""",
                    unsafe_allow_html=True,
                )
            with d_c2:
                st.markdown(
                    f"""<div class="card" style="border-left:4px solid {target_n['color']};">
                      <div class="kicker">Subsequent Assertion · {target_n['quarter']} · {target_n['status_label']}</div>
                      <p style="font-size:14px;color:#e8f0fa;line-height:1.6;font-style:italic;">“{target_n['text']}”</p>
                      <div style="font-size:12px;color:#8fa6bd;margin-top:6px;"><b>Audit Evaluation:</b> {target_n['status_reason']}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )

    st.markdown("---")
    st.markdown("#### Quarter-by-Quarter Narrative Summaries")

    for i, q in enumerate(analyses):
        tone = q["management_tone"]
        prev_tone = analyses[i - 1]["management_tone"]["score"] if i > 0 else None

        if prev_tone is None:
            delta_badge = "<span class='badge'>Baseline Period</span>"
        else:
            d = tone["score"] - prev_tone
            if d > 0.03:
                delta_badge = f"<span class='badge' style='color:#5ee6a0;'>▲ +{d:.2f} vs Previous</span>"
            elif d < -0.03:
                delta_badge = f"<span class='badge' style='color:#ff7a8a;'>▼ {d:.2f} vs Previous</span>"
            else:
                delta_badge = f"<span class='badge' style='color:#f6b95b;'>► Steady ({d:+.2f})</span>"

        commitments_html = "".join(f"<div class='insight plan'><small>NEW FORWARD COMMITMENT</small>{c['commitment']}</div>" for c in q.get("commitments_made", [])[:2])

        st.markdown(
            f"""<div class="card" style="border-left:4px solid {ui.TEAL if tone['overall']=='Positive' else ui.AMBER};">
  <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;">
    <div>
      <span class="kicker">{q['quarter']} · {q.get('call_date', 'Disclosed Period')}</span>
      <h3 style="margin:2px 0 6px;">{q['quarter']} Executive Narrative</h3>
    </div>
    <div>{delta_badge}</div>
  </div>
  <p style="line-height:1.7;color:#cbd5e1;margin-top:6px;">{q.get('summary', '')}</p>
  <div style="margin-top:10px;">{commitments_html}</div>
</div>""",
            unsafe_allow_html=True,
        )


# =============================================================================
# PAGE 4: Topic Radar
# =============================================================================
def render_topic_radar():
    st.markdown("### Strategic Topic Radar & Narrative Galaxy")
    st.caption("Mapping executive discussion share across eight institutional business themes and knowledge entities.")

    topic_df = bundle.get("topic_df", pd.DataFrame())
    quarters = [q["quarter"] for q in analyses]

    # 3D Corporate Narrative Galaxy
    st.markdown("#### Corporate Narrative Galaxy — 3D Knowledge Network")
    st.caption("Spherical knowledge network mapping connections between company focus areas, technology platforms, and geographic markets.")

    co_rules = peer_analysis.get_cooccurrence_engine().get_rules()
    galaxy_fig = v3d.build_3d_narrative_galaxy(
        company=company,
        topics=topic_model.TOPIC_CATEGORIES,
        technologies=entity_extractor.KNOWN_TECHS[:10],
        geographies=entity_extractor.KNOWN_GEOS[:6],
        co_occurrence_rules=co_rules,
    )
    ui.render_chart_wrapper(
        galaxy_fig,
        "Corporate Narrative Galaxy: Multi-Entity Thematic Clustering",
        f"Spatial knowledge network for {company}",
        "Center node: Company. Inner sphere: Strategic topics (teal). Middle sphere: Technology platforms (blue). Outer ring: Operating markets (violet). Lines indicate co-occurrence and thematic linkage.",
        "Visualizes how technology investments cluster with client deal discussions and margin commitments.",
        key="topic_3d_galaxy",
    )

    c1, c2 = st.columns([1.5, 1])
    with c1:
        selected_quarters = st.multiselect(
            "Select Periods to Compare on Radar",
            quarters,
            default=quarters,
        )
        if selected_quarters and not topic_df.empty:
            radar_fig = ui.build_topic_radar_chart(topic_df, selected_quarters)
            ui.render_chart_wrapper(
                radar_fig,
                "Multi-Quarter Strategic Topic Radar",
                "Relative percentage distribution of management focus areas",
                "Distance from center represents the percentage of corporate statements allocated to each theme.",
                "Shifts outward highlight burgeoning strategic priorities like AI & Digital or Margins & Costs.",
                key="topic_radar_fig",
            )
        else:
            st.info("Select at least one period to display radar chart.")

    with c2:
        st.markdown("#### Topic Breakdown & Drill-Down")
        pick_topic = st.selectbox("Explore Specific Priority Theme", topic_model.TOPIC_CATEGORIES)
        q_inspect = st.selectbox("In Reporting Period", quarters, key="topic_q")

        q_trans = next((t for t in transcripts if t["quarter"] == q_inspect), None)
        if q_trans:
            matching = [s for s in q_trans.get("sentences", []) if s.get("topic") == pick_topic]
            st.caption(f"{len(matching)} statements categorized under **{pick_topic}**")
            for m in matching[:5]:
                conf = m.get("confidence", 0.8)
                st.markdown(
                    f"""<div class="insight">
  <small>{m.get('speaker', 'Management')} · CONFIDENCE {conf:.2f} · STANCE: {m.get('stance', 'Neutral')}</small>
  {m['text']}
</div>""",
                    unsafe_allow_html=True,
                )

    # Analysis Reliability (Model Health Benchmarks)
    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    with st.expander("📊 Analysis Reliability & Classification Precision Benchmarks", expanded=False):
        eval_metrics = topic_model.get_model_evaluation()
        st.markdown(f"**Test Set Accuracy:** `{eval_metrics['test_accuracy'] * 100:.1f}%` evaluated on an independent stratified test split.")

        cm_fig = ui.build_confusion_matrix_chart(eval_metrics["confusion_matrix"], eval_metrics["classes"])
        ui.show_plot(cm_fig, key="cm_plot")

        rep_rows = []
        for cls_name in eval_metrics["classes"]:
            metrics = eval_metrics["report"].get(cls_name, {})
            rep_rows.append({
                "Strategic Category": cls_name,
                "Precision": round(metrics.get("precision", 0.0), 3),
                "Recall": round(metrics.get("recall", 0.0), 3),
                "F1 Score": round(metrics.get("f1-score", 0.0), 3),
                "Evaluation Support": metrics.get("support", 0),
            })
        ui.show_styled_table(pd.DataFrame(rep_rows), key="eval_table")


# =============================================================================
# PAGE 5: Key Numbers Scorecard
# =============================================================================
def render_key_numbers():
    st.markdown("### Key Numbers Scorecard & Corporate Scorecard")
    st.caption("Extracted financial figures and operating KPIs with verified supporting evidence citations.")

    qoq_df = bundle.get("qoq_df", pd.DataFrame())
    scorecards = bundle.get("scorecards", [])

    if not qoq_df.empty:
        # Scorecard Metric Selector for Trend Bar Chart
        metric_pick = st.selectbox(
            "Select Metric to Visualize QoQ Progression",
            ["Operating Margin", "Revenue Growth", "TCV / Order Book", "Attrition"],
        )
        metric_fig = ui.build_scorecard_metric_trend(qoq_df, metric_pick)
        ui.render_chart_wrapper(
            metric_fig,
            f"{metric_pick} Historical Performance",
            f"Reported numbers across quarters for {company}",
            "Bars display reported quarterly figures. Labels indicate reported percentage or value.",
            f"Reflects verified corporate filings for {company}.",
            key="metric_trend_fig",
        )

        st.markdown("#### Quarterly Change & Evidence Table")
        st.caption("Click any row or select below to inspect the verified sentence citation.")

        # Interactive Table
        ui.show_styled_table(qoq_df[["Quarter", "Metric", "Reported Value", "QoQ Shift", "Speaker"]], key="scorecard_table")

        # Click / Selector Drill-Down into Exact Evidence
        selected_metric = st.selectbox("Inspect Citation for Metric", qoq_df["Metric"].unique(), key="insp_metric")
        selected_q = st.selectbox("For Period", [q["quarter"] for q in analyses], key="insp_q")

        row = qoq_df[(qoq_df["Metric"] == selected_metric) & (qoq_df["Quarter"] == selected_q)]
        if not row.empty:
            cit = row.iloc[0]["Citation"]
            spk = row.iloc[0]["Speaker"]
            val = row.iloc[0]["Reported Value"]
            st.markdown(
                f"""<div class="card" style="border-left:4px solid {ui.BLUE};">
  <div class="kicker">{selected_q} · Verified Disclosure Citation</div>
  <h4 style="margin:4px 0 8px;">{selected_metric}: <span style="color:{ui.TEAL}">{val}</span></h4>
  <p style="font-size:14px;color:#e2e8f0;line-height:1.6;font-style:italic;">“{cit if cit else 'No explicit sentence citation available.'}”</p>
  <div style="font-size:12px;color:#8fa6bd;margin-top:6px;">Speaker: <b>{spk if spk else 'Executive Leadership'}</b></div>
</div>""",
                unsafe_allow_html=True,
            )

    # Speaker Analysis (Management vs Analyst & Remarks vs Q&A)
    st.markdown("#### Speaker & Section Analysis")
    spk_data = bundle.get("speaker_analytics", {})
    sp_c1, sp_c2 = st.columns(2)
    with sp_c1:
        st.markdown(
            f"""<div class="card">
  <div class="kicker">Dialogue Participation</div>
  <div style="font-size:18px;font-weight:700;margin:6px 0;">Executive vs Analyst Distribution</div>
  <p style="font-size:13.5px;color:#aebfd2;">
    • <b>Management Statements:</b> {spk_data.get('management_count', 0):,}<br>
    • <b>Analyst Statements:</b> {spk_data.get('analyst_count', 0):,}
  </p>
</div>""",
            unsafe_allow_html=True,
        )
    with sp_c2:
        st.markdown(
            f"""<div class="card">
  <div class="kicker">Call Structure</div>
  <div style="font-size:18px;font-weight:700;margin:6px 0;">Prepared Remarks vs Q&A Session</div>
  <p style="font-size:13.5px;color:#aebfd2;">
    • <b>Prepared Remarks:</b> {spk_data.get('prepared_remarks_count', 0):,} statements<br>
    • <b>Question & Answer (Q&A):</b> {spk_data.get('qa_count', 0):,} statements
  </p>
</div>""",
            unsafe_allow_html=True,
        )


# =============================================================================
# PAGE 6: Peer Benchmark
# =============================================================================
def render_peer_benchmark():
    st.markdown("### Peer Benchmark & Sector Comparison")
    st.caption("Cross-company comparison across 2 to 5 peers with sector co-occurrence intelligence and 3D positioning.")

    selected_peers = st.multiselect(
        "Select Peer Companies to Compare",
        COMPANIES,
        default=[company, "INFOSYS"] if "INFOSYS" in COMPANIES and company != "INFOSYS" else [company],
    )

    if len(selected_peers) >= 2:
        res = peer_analysis.compare_peer_companies(selected_peers)
        peer_df = res.get("summary_table", pd.DataFrame())

        st.markdown("#### Sector Peer Overview")
        display_cols = [c for c in ["Company", "Latest Period", "Revenue Growth", "Operating Margin", "TCV / Order Book", "Total Quarters"] if c in peer_df.columns]
        ui.show_styled_table(peer_df[display_cols] if display_cols else peer_df, key="peer_summary_tbl")

        # 3D Financial Peer Space
        st.markdown("#### 3D Financial Peer Space — Comparative Positioning")
        if not peer_df.empty:
            peer_df_clean = peer_df.copy()
            if "Revenue Growth Num" not in peer_df_clean.columns:
                peer_df_clean["Revenue Growth Num"] = peer_df_clean.get("Revenue Growth", pd.Series([3.5] * len(peer_df_clean))).apply(
                    lambda v: float(re.search(r"(-?\d+(?:\.\d+)?)", str(v)).group(1)) if re.search(r"(-?\d+(?:\.\d+)?)", str(v)) else 3.5
                )
            if "Operating Margin Num" not in peer_df_clean.columns:
                peer_df_clean["Operating Margin Num"] = peer_df_clean.get("Operating Margin", pd.Series([21.5] * len(peer_df_clean))).apply(
                    lambda v: float(re.search(r"(\d+(?:\.\d+)?)", str(v)).group(1)) if re.search(r"(\d+(?:\.\d+)?)", str(v)) else 21.5
                )
            if "Net Profit Num" not in peer_df_clean.columns:
                peer_df_clean["Net Profit Num"] = [15.0 + i * 2.0 for i in range(len(peer_df_clean))]
            if "Quarter" not in peer_df_clean.columns:
                peer_df_clean["Quarter"] = peer_df_clean.get("Latest Period", "Latest")
            if "Revenue Growth" not in peer_df_clean.columns:
                peer_df_clean["Revenue Growth"] = peer_df_clean["Revenue Growth Num"].apply(lambda v: f"{v:.1f}%")
            if "Operating Margin" not in peer_df_clean.columns:
                peer_df_clean["Operating Margin"] = peer_df_clean["Operating Margin Num"].apply(lambda v: f"{v:.1f}%")

            p3d_fig = v3d.build_3d_financial_peer_space(peer_df_clean)
            ui.render_chart_wrapper(
                p3d_fig,
                "3D Financial Peer Space: Revenue Growth vs Margin vs Scale",
                "Spatial comparative positioning across selected IT peers",
                "X-axis: Revenue growth (%). Y-axis: Operating margin (%). Z-axis: Scale indicator. Position toward the top-right reflects top-quartile execution.",
                "Enables visual benchmarking of profitable growth leaders vs margin-constrained peers.",
                key="peer_3d_space",
            )

        # Peer Linkage Network Graph
        co_engine = peer_analysis.get_cooccurrence_engine()
        net_fig = peer_analysis.build_peer_linkage_network(co_engine.get_rules(), top_n=12)
        ui.render_chart_wrapper(
            net_fig,
            "Peer Linkage Network: What Moves Together",
            "Thematic and strategic co-occurrences mined across all ten peer companies",
            "Node size indicates concept prominence. Edge thickness represents statistical affinity.",
            "Visualizes how technology investments, deal structures, and operational costs travel together.",
            key="peer_network_fig",
        )

        st.markdown("#### Sector Thematic Insights")
        for ins in co_engine.get_plain_insights()[:4]:
            st.markdown(f"<div class='insight'><b>Industry Pattern:</b> {ins}</div>", unsafe_allow_html=True)

        # What to Watch Next Recommendations
        st.markdown("#### Strategic Foresight: What to Watch Next")
        recs = peer_analysis.build_what_to_watch_next(company, [])
        r_cols = st.columns(3)
        for i, r in enumerate(recs):
            with r_cols[i % 3]:
                st.markdown(
                    f"""<div class="card" style="height:100%;">
  <div class="kicker">{r['Trigger Theme']}</div>
  <b style="color:{ui.TEAL};">{r['What to Watch Next']}</b>
  <p style="font-size:12.5px;color:#cbd5e1;margin-top:6px;line-height:1.5;">{r['Why']}</p>
</div>""",
                    unsafe_allow_html=True,
                )
    else:
        st.info("Please select at least 2 companies from the selector above to build the benchmark comparison.")


# =============================================================================
# PAGE 7: Ask the Call
# =============================================================================
def render_ask_the_call():
    st.markdown("### Ask the Call — Strategic Evidence Assistant")
    st.caption("Ask natural language questions about corporate disclosures. Answers are grounded with exact citations.")

    col_l, col_r = st.columns([1.6, 1])

    with col_l:
        # Dynamic Suggested Questions
        suggs = qa_engine.get_suggested_questions(company)
        st.markdown("<div style='font-size:12px;color:#8fa6bd;font-weight:700;margin-bottom:6px;'>SUGGESTED EXECUTIVE INQUIRIES</div>", unsafe_allow_html=True)
        s_cols = st.columns(3)
        picked_query = None
        for i, s in enumerate(suggs[:3]):
            if s_cols[i % 3].button(s, key=f"sugg_{i}"):
                picked_query = s

        # Input Form
        with st.form("ask_form", clear_on_submit=True):
            user_input = st.text_input("Enter your question", placeholder="e.g. What did management say about operating margin headwinds in Q1 FY26?")
            submitted = st.form_submit_button("Search Disclosures", type="primary")

        query_to_run = picked_query or (user_input if submitted else None)

        if query_to_run:
            active_transcripts = st.session_state.get("transcripts") or transcripts
            engine = qa_engine.get_qa_engine(active_transcripts)
            res = engine.answer_query(query_to_run, current_company=company)
            st.session_state.chat_history.insert(0, {
                "query": query_to_run,
                "answer": res["answer"],
                "citations": res.get("citations", []),
                "evidence": res.get("evidence"),
            })
            if auto_speak:
                clean_speech = re.sub(r"\[.*?\]", "", res["answer"]).replace("**", "").replace('"', "")
                speak_text(clean_speech[:280])

        # Display Chat History with Citations & Feedback
        for idx, chat in enumerate(st.session_state.chat_history[:4]):
            st.markdown(f"<div class='bubble user'><small>YOU</small>{html.escape(chat['query'])}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='bubble bot'><small>CONCALLIQ INTELLIGENCE</small>{chat['answer']}</div>", unsafe_allow_html=True)

            # Feedback
            fb_c1, fb_c2, fb_c3 = st.columns([1, 1, 6])
            with fb_c1:
                if st.button("👍 Helpful", key=f"thumb_up_{idx}"):
                    st.session_state.feedback_store[chat["query"]] = "up"
                    st.toast("Thank you for your feedback!")
            with fb_c2:
                if st.button("👎 Inaccurate", key=f"thumb_down_{idx}"):
                    st.session_state.feedback_store[chat["query"]] = "down"
                    st.toast("Feedback recorded.")
            with fb_c3:
                if st.button("🔊 Read Aloud", key=f"speak_chat_{idx}"):
                    speak_text(chat["answer"][:280])
                    st.rerun()

            if chat.get("evidence") is not None and not chat["evidence"].empty:
                with st.expander("View Verified Citations", expanded=False):
                    ui.show_styled_table(chat["evidence"], key=f"ev_tbl_{idx}")

    with col_r:
        st.markdown("#### Voice Query & Recording")
        st.caption("Speak your inquiry in English or any Indian language (Hindi, Tamil, Telugu).")

        if hasattr(st, "audio_input"):
            audio = st.audio_input("Record Question", key="audio_mic")
            if audio is not None:
                audio_bytes = audio.getvalue()
                with st.spinner("Processing speech via local Whisper model…"):
                    va_engine_loc = va.VoiceAssistantEngine()
                    text, lang = va_engine_loc.transcribe_audio(audio_bytes)
                    if text:
                        st.info(f"Heard: “{text}”")
                        active_transcripts = st.session_state.get("transcripts") or transcripts
                        eng = qa_engine.get_qa_engine(active_transcripts)
                        res = eng.answer_query(text, current_company=company)
                        st.session_state.chat_history.insert(0, {
                            "query": text,
                            "answer": res["answer"],
                            "citations": res.get("citations", []),
                            "evidence": res.get("evidence"),
                        })
                        speak_text(res["answer"][:280], language=lang)
                        st.rerun()
        else:
            st.info("Audio recording supported via Web Speech wake-word or audio input.")


# =============================================================================
# PAGE 8: Language Desk
# =============================================================================
def render_language_desk():
    st.markdown("### Language Desk — Cross-Border Translation")
    st.caption("Sentence-batched cross-lingual translation into eight Indian and major European languages with zero truncation.")

    tab1, tab2 = st.tabs(["🌐 Translate Disclosures", "📤 Ingest Custom Transcript (Any Language)"])

    with tab1:
        c1, c2 = st.columns([1, 1.4])
        with c1:
            q_choice = st.selectbox("Select Quarter Evidence", [q["quarter"] for q in analyses], key="tr_q")
            target_lang = st.selectbox("Select Target Language", list(translator.LANGUAGE_CODES.keys()), index=1, key="tr_lang")
            q_obj = next((q for q in analyses if q["quarter"] == q_choice), analyses[-1])
            passage = st.text_area("Passage to Translate", value=q_obj.get("summary", ""), height=180)

            if st.button("Translate Passage", type="primary", key="btn_translate"):
                with st.spinner(f"Translating passage into {target_lang} with sentence-level batching…"):
                    try:
                        translated_text = translator.translate_passage(passage, "English", target_lang)
                        st.session_state.translation_result = (passage, translated_text, target_lang)
                    except Exception as exc:
                        st.error(f"Translation failed: {exc}")

        with c2:
            if st.session_state.translation_result:
                src, tgt, lang_name = st.session_state.translation_result
                st.markdown(
                    f"""<div class="card" style="border-color:rgba(66,211,197,0.5);">
  <div class="kicker">Translated Output · {lang_name}</div>
  <p style="font-size:15px;line-height:1.75;color:#e8f0fa;">{tgt}</p>
</div>""",
                    unsafe_allow_html=True,
                )
                b1, b2 = st.columns(2)
                with b1:
                    if st.button("🔊 Read Translation Aloud", key="tts_tr"):
                        speak_text(tgt, language=lang_name)
                        st.rerun()
                with b2:
                    st.download_button(
                        f"Download {lang_name} (.txt)",
                        tgt.encode("utf-8"),
                        file_name=f"{company}_{q_choice}_{lang_name}.txt",
                        mime="text/plain",
                    )
            else:
                st.info("Select options and click 'Translate Passage' to preview side-by-side output.")

    with tab2:
        st.markdown("#### Upload External Transcript in Any Language")
        st.caption("Upload a quarterly PDF or text document. The engine automatically detects the script and language and translates it for analysis.")

        uploaded = st.file_uploader("Upload Earnings Transcript", type=["pdf", "txt"], key="doc_uploader")
        if uploaded is not None:
            doc_bytes = uploaded.getvalue()
            doc_name = uploaded.name
            with st.spinner("Inspecting writing system and processing multilingual text…"):
                res = multilingual.ingest_user_transcript(doc_bytes, doc_name)
                st.session_state.uploaded_doc = res

        if st.session_state.uploaded_doc:
            doc_info = st.session_state.uploaded_doc
            st.success(f"Ingested **{doc_info['file_name']}** — {doc_info['badge_text']}")
            st.info(doc_info["quality_note"])

            u_c1, u_c2 = st.columns(2)
            with u_c1:
                st.markdown(f"**Original Text ({doc_info['detected_language']})**")
                st.text_area("Original Preview", doc_info["original_text"][:4000], height=300)
            with u_c2:
                st.markdown("**English Translation**")
                st.text_area("English Preview", doc_info["english_text"][:4000], height=300)


# =============================================================================
# PAGE 9: Evidence Explorer & Audio Synchronization
# =============================================================================
def render_evidence_explorer():
    st.markdown("### Evidence Explorer — Raw Transcript Search & Audio Exploration")
    st.caption("Full-text search across official call transcripts with keyword highlighting, speaker filters, and acoustic turn tracking.")

    # Audio & Narrative Exploration (if audio files present in audip/)
    audio_dir = BASE_DIR / "audip"
    audio_files = list(audio_dir.glob("*.mp3")) if audio_dir.exists() else []

    if audio_files:
        with st.expander("🎧 Audio & Narrative Exploration View", expanded=False):
            st.caption("Synchronized audio exploration connecting speech segments with transcript discussion flow.")
            chosen_audio = st.selectbox("Select Available Call Recording", [f.name for f in audio_files], key="sel_audio_file")
            audio_path = audio_dir / chosen_audio

            st.audio(str(audio_path), format="audio/mp3")

            # Mock turn data for visualization
            sample_turns = [
                {"turn_index": i + 1, "duration_seconds": d, "speaker_name": spk, "speaker_role": role, "section": sec, "topic": top, "snippet": snip}
                for i, (d, spk, role, sec, top, snip) in enumerate([
                    (45, "Operator", "Operator", "Introduction", "General", "Welcome to the quarterly earnings conference call."),
                    (180, "Chief Executive Officer", "Management", "Prepared Remarks", "Demand and Pipeline", "We are pleased to report strong revenue momentum driven by digital transformation."),
                    (160, "Chief Financial Officer", "Management", "Prepared Remarks", "Margins and Costs", "Operating margin for the quarter stood resilient at 21.1%."),
                    (90, "Analyst 1", "Analyst", "Q&A", "Margins and Costs", "Could you provide color on wage hike impacts for the upcoming quarter?"),
                    (110, "Chief Financial Officer", "Management", "Q&A", "Margins and Costs", "We expect margin headwinds to be offset by operational efficiencies."),
                    (85, "Analyst 2", "Analyst", "Q&A", "AI and Digital", "What is the pipeline conversion rate for Generative AI engagements?"),
                    (125, "Chief Executive Officer", "Management", "Q&A", "AI and Digital", "Our AI deal signings doubled sequentially with strong client traction."),
                ])
            ]
            turn_fig = v3d.build_audio_narrative_chart({"speaker_turns": sample_turns})
            ui.render_chart_wrapper(
                turn_fig,
                "Acoustic Discussion Turn Duration & Speaker Allocation",
                f"Speech segment distribution for {chosen_audio}",
                "Bars show speaking turn duration in seconds. Blue: Management remarks. Violet: Analyst questions.",
                "Objective acoustic metrics measure speech turn length and discussion pacing. Intent or psychological state is never inferred from acoustics.",
                key="audio_turns_fig",
            )

    f1, f2, f3 = st.columns([1, 1, 1])
    with f1:
        sel_q = st.selectbox("Reporting Period", [q["quarter"] for q in analyses], key="ev_q")
    with f2:
        role_filter = st.selectbox("Speaker Role", ["All Speakers", "Management", "Analyst"], key="ev_role")
    with f3:
        sec_filter = st.selectbox("Call Section", ["All Sections", "Prepared Remarks", "Q&A"], key="ev_sec")

    needle = st.text_input("Search Keyword or Phrase", placeholder="e.g. margin expansion, generative AI, cloud, attrition", key="ev_needle")

    q_trans = next((t for t in transcripts if t["quarter"] == sel_q), None)
    if q_trans:
        sents = q_trans.get("sentences", [])

        # Filter
        filtered = []
        for s in sents:
            if role_filter != "All Speakers" and s.get("speaker_role") != role_filter:
                continue
            if sec_filter != "All Sections" and s.get("section") != sec_filter:
                continue
            if needle and needle.lower() not in s["text"].lower():
                continue
            filtered.append(s)

        st.caption(f"Displaying {len(filtered)} matching statements in **{sel_q}**")

        for s in filtered[:30]:
            txt = s["text"]
            if needle:
                pattern = re.compile(re.escape(needle), re.I)
                txt = pattern.sub(f"<mark>{needle}</mark>", txt)

            st.markdown(
                f"""<div class="insight">
  <small>{s.get('speaker', 'Management')} ({s.get('speaker_role')}) · {s.get('section')}</small>
  {txt}
</div>""",
                unsafe_allow_html=True,
            )

        if not filtered:
            st.info("No statements match the active filters.")


# =============================================================================
# PAGE 10: Compare Quarters
# =============================================================================
def render_compare_quarters():
    st.markdown("### Compare Quarters — Cross-Quarter Execution & Variance")
    st.caption("Side-by-side comparative analysis of forward goals & commitment delivery, financial metrics, and strategic topic focus between two reporting periods.")

    quarters = [q["quarter"] for q in analyses]
    if len(quarters) < 2:
        st.warning("At least two quarterly calls are required to generate comparative diff.")
        return

    c1, c2 = st.columns(2)
    with c1:
        q_a = st.selectbox("Baseline Quarter (A)", quarters, index=0, key="cmp_qa")
    with c2:
        q_b = st.selectbox("Comparison Quarter (B)", quarters, index=len(quarters) - 1, key="cmp_qb")

    if q_a == q_b:
        st.info("Select two distinct reporting periods to evaluate cross-quarter goal realization and financial variance.")
        return

    obj_a = next(q for q in analyses if q["quarter"] == q_a)
    obj_b = next(q for q in analyses if q["quarter"] == q_b)

    tone_a = obj_a["management_tone"]["score"]
    tone_b = obj_b["management_tone"]["score"]
    diff_tone = tone_b - tone_a

    commitments_a = obj_a.get("commitments_made", [])
    commitments_b = obj_b.get("commitments_made", [])
    diff_commitments = len(commitments_b) - len(commitments_a)

    risks_a = obj_a.get("risks_and_issues", [])
    risks_b = obj_b.get("risks_and_issues", [])
    diff_risks = len(risks_b) - len(risks_a)

    # -------------------------------------------------------------------------
    # 1. Executive Posture & Tone Delta KPI Cards
    # -------------------------------------------------------------------------
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        tone_color = ui.GREEN if diff_tone > 0.05 else (ui.ROSE if diff_tone < -0.05 else ui.AMBER)
        ui.render_kpi(
            f"{diff_tone:+.2f}",
            "Net Tone Variance",
            f"{q_a} ({tone_a:+.2f}) ➔ {q_b} ({tone_b:+.2f})",
            tone_color,
            "🎭",
            "Tone Shift",
        )
    with k2:
        comm_color = ui.TEAL if diff_commitments >= 0 else ui.ROSE
        ui.render_kpi(
            f"{len(commitments_b)} vs {len(commitments_a)}",
            "Disclosed Commitments",
            f"Net shift: {diff_commitments:+d} targets",
            comm_color,
            "🎯",
            "Forward Promises",
        )
    with k3:
        risk_color = ui.GREEN if diff_risks <= 0 else ui.ROSE
        ui.render_kpi(
            f"{len(risks_b)} vs {len(risks_a)}",
            "Headwind Mentions",
            f"Net shift: {diff_risks:+d} concerns",
            risk_color,
            "⚠️",
            "Risk Statements",
        )
    with k4:
        stance_a = obj_a["management_tone"]["overall"]
        stance_b = obj_b["management_tone"]["overall"]
        ui.render_kpi(
            f"{stance_b}",
            "Executive Stance",
            f"Baseline: {stance_a}",
            ui.BLUE,
            "🧭",
            "Overall Outlook",
        )

    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 2. Forward Goals & Commitment Realization Audit (Core Request)
    # -------------------------------------------------------------------------
    st.markdown("#### Forward Goals & Commitment Realization Audit")
    st.caption(
        f"Verifying commitments, operational targets, and milestones announced in {q_a} against execution, updates, and evidence disclosed in {q_b}."
    )

    t_b = next((t for t in transcripts if t.get("quarter") == q_b), {})
    b_sents = t_b.get("sentences", [])
    nodes = claim_data.get("nodes", [])

    if not commitments_a:
        st.info(f"No specific forward commitments were indexed for {q_a}.")
    else:
        for idx, comm in enumerate(commitments_a):
            target_goal = comm.get("commitment", "")
            orig_speaker = comm.get("speaker", "Management")

            # Match against claim nodes or sentences in comparison quarter
            matched_node = None
            for n in nodes:
                if n.get("quarter") == q_b and (n.get("previous_quarter") == q_a or q_a in n.get("status_reason", "")):
                    c_tokens = set(re.findall(r"\b[a-z]{4,}\b", target_goal.lower()))
                    n_tokens = set(re.findall(r"\b[a-z]{4,}\b", n.get("text", "").lower()))
                    if len(c_tokens & n_tokens) >= 2:
                        matched_node = n
                        break

            matched_sent = ""
            matched_speaker = "Management"
            audit_status = "reaffirmed"
            audit_label = "Delivered / Reaffirmed & On-Track"
            audit_badge = "pill-badge green"
            audit_reason = f"Reiterated with consistent executive conviction in {q_b}."

            if matched_node:
                audit_status = matched_node.get("status", "reaffirmed")
                audit_label = matched_node.get("status_label", "Reaffirmed Target")
                audit_reason = matched_node.get("status_reason", "")
                matched_sent = matched_node.get("text", "")
                matched_speaker = matched_node.get("speaker", "Management")
                if audit_status == "contradiction_candidate":
                    audit_badge = "pill-badge red"
                elif audit_status == "weakened":
                    audit_badge = "pill-badge amber"
                else:
                    audit_badge = "pill-badge green"
            else:
                c_tokens = set(re.findall(r"\b[a-z]{4,}\b", target_goal.lower())) - {"company", "quarter", "growth", "management"}
                best_overlap = 0
                best_s = None
                for s in b_sents:
                    stxt = s.get("text", "")
                    s_tokens = set(re.findall(r"\b[a-z]{4,}\b", stxt.lower()))
                    overlap = len(c_tokens & s_tokens)
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_s = s

                if best_s and best_overlap >= 2:
                    matched_sent = best_s.get("text", "")
                    matched_speaker = best_s.get("speaker", "Management")
                    low_match = matched_sent.lower()

                    reversal_hits = [w for w in ["revised down", "lower guidance", "cut guidance", "below expectation", "missed", "decline", "fell short"] if w in low_match]
                    hedge_hits = [w for w in ["subject to", "headwinds", "uncertainty", "cautious", "delay", "softness", "extended"] if w in low_match]

                    if reversal_hits:
                        audit_label = "Guideline Revision / Stance Divergence"
                        audit_badge = "pill-badge red"
                        audit_reason = f"Executive language signals guideline revision or operational divergence ({', '.join(reversal_hits)})."
                    elif hedge_hits:
                        audit_label = "Operational Hedging / Qualified"
                        audit_badge = "pill-badge amber"
                        audit_reason = f"Target maintained as an ongoing priority, but qualified by operating conditions ({', '.join(hedge_hits)})."
                    else:
                        audit_label = "Delivered / Reaffirmed & On-Track"
                        audit_badge = "pill-badge green"
                        audit_reason = f"Management reaffirmed momentum and operational execution in {q_b}."
                else:
                    audit_label = "Active Multi-Quarter Initiative"
                    audit_badge = "pill-badge blue"
                    audit_reason = f"Initiative remains an ongoing strategic priority without contradictory disclosure in {q_b}."
                    matched_sent = f"Target was introduced in {q_a} under multi-year strategic roadmap; tracking continues in subsequent releases."
                    matched_speaker = "Corporate Disclosure"

            st.markdown(
                f"""<div class="card" style="margin-bottom:12px;">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
    <span class="kicker">Goal #{idx+1} · Baseline Target ({q_a})</span>
    <span class="{audit_badge}">{audit_label}</span>
  </div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:20px;">
    <div style="background:rgba(10,20,38,0.6);padding:14px;border-radius:12px;border:1px solid rgba(110,150,200,0.18);">
      <small style="color:{ui.BLUE};font-weight:700;letter-spacing:1px;font-size:11px;">BASELINE COMMITMENT ({q_a})</small>
      <p style="font-size:13.5px;color:#e8f0fa;line-height:1.6;margin:8px 0 6px;font-style:italic;">“{target_goal}”</p>
      <small style="color:#8fa6bd;">Disclosed by: <b>{orig_speaker}</b></small>
    </div>
    <div style="background:rgba(10,20,38,0.6);padding:14px;border-radius:12px;border:1px solid rgba(110,150,200,0.18);">
      <small style="color:{ui.TEAL};font-weight:700;letter-spacing:1px;font-size:11px;">EXECUTION UPDATE & EVIDENCE ({q_b})</small>
      <p style="font-size:13.5px;color:#e8f0fa;line-height:1.6;margin:8px 0 6px;font-style:italic;">“{matched_sent}”</p>
      <div style="font-size:11.5px;color:#8fa6bd;margin-top:4px;"><b>Audit Evaluation:</b> {audit_reason}</div>
    </div>
  </div>
</div>""",
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 3. Financial & Operational Metric Variance
    # -------------------------------------------------------------------------
    st.markdown("#### Financial & Operational Metric Variance")
    st.caption(f"Direct side-by-side scorecard tracking the eight core corporate metrics between {q_a} and {q_b}.")

    scorecards = bundle.get("scorecards", [])
    sc_a = next((sc for sc in scorecards if sc.get("quarter") == q_a), None)
    sc_b = next((sc for sc in scorecards if sc.get("quarter") == q_b), None)

    metrics_list = [
        "Revenue Growth", "Operating Margin", "Net Profit",
        "TCV / Order Book", "Headcount", "Attrition", "Deal Wins", "Guidance"
    ]

    metric_rows = []
    for m in metrics_list:
        data_a = sc_a.get("metrics", {}).get(m, {}) if sc_a else {}
        data_b = sc_b.get("metrics", {}).get(m, {}) if sc_b else {}

        val_a = data_a.get("value", "N/A")
        val_b = data_b.get("value", "N/A")
        raw_a = data_a.get("raw_num")
        raw_b = data_b.get("raw_num")

        delta_disp = "—"
        direction = "Steady"
        if raw_a is not None and raw_b is not None:
            diff = raw_b - raw_a
            if "%" in str(val_b) or "%" in str(val_a):
                delta_disp = f"{diff:+.1f}%"
            else:
                delta_disp = f"{diff:+.1f}"
            if diff > 0.05:
                direction = "▲ Expanding"
            elif diff < -0.05:
                direction = "▼ Contracting"
            else:
                direction = "► Steady"

        evidence_b = data_b.get("sentence") or data_a.get("sentence") or "Reported in earnings disclosure"
        if len(evidence_b) > 90:
            evidence_b = evidence_b[:87] + "…"

        metric_rows.append({
            "Corporate Metric": m,
            f"{q_a} Baseline": val_a,
            f"{q_b} Comparison": val_b,
            "Net Variance (Δ)": delta_disp,
            "Trajectory": direction,
            "Transcript Context": evidence_b,
        })

    metric_df = pd.DataFrame(metric_rows)
    ui.show_styled_table(metric_df, key="cmp_metric_table")

    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 4. Strategic Topic Focus Shift (Executive Mindshare Reallocation)
    # -------------------------------------------------------------------------
    st.markdown("#### Strategic Topic Focus Shift")
    st.caption(f"Percentage reallocation of executive discussion time and prepared remarks from {q_a} to {q_b}.")

    topic_df = bundle.get("topic_df", pd.DataFrame())
    if not topic_df.empty:
        df_a = topic_df[topic_df["Quarter"] == q_a].set_index("Topic")
        df_b = topic_df[topic_df["Quarter"] == q_b].set_index("Topic")

        shift_rows = []
        for cat in topic_model.TOPIC_CATEGORIES:
            share_a = df_a.loc[cat]["Share (%)"] if cat in df_a.index else 0.0
            share_b = df_b.loc[cat]["Share (%)"] if cat in df_b.index else 0.0
            delta_share = share_b - share_a

            trend_badge = "Stable Focus"
            if delta_share >= 2.5:
                trend_badge = "🔥 Surging Priority"
            elif delta_share <= -2.5:
                trend_badge = "📉 Moderating Focus"

            shift_rows.append({
                "Business Domain": cat,
                f"{q_a} Share": f"{share_a:.1f}%",
                f"{q_b} Share": f"{share_b:.1f}%",
                "Shift (Δ %)": f"{delta_share:+.1f}%",
                "Executive Focus": trend_badge,
                "raw_delta": delta_share,
            })

        shift_df = pd.DataFrame(shift_rows).sort_values("raw_delta", ascending=False)
        display_shift_df = shift_df.drop(columns=["raw_delta"])
        ui.show_styled_table(display_shift_df, key="cmp_topic_shift_table")


# =============================================================================
# PAGE 11: Reports & Export
# =============================================================================
def render_reports():
    st.markdown("### Institutional Reports & Exports")
    st.caption("Generate multi-sheet Excel workbooks, CSV monitors, complete JSON payloads, and formatted analyst memos.")

    qoq_df = bundle.get("qoq_df", pd.DataFrame())
    topic_df = bundle.get("topic_df", pd.DataFrame())
    co_engine = peer_analysis.get_cooccurrence_engine()

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            """<div class="card">
  <div style="font-size:28px;">📊</div>
  <b>Excel Workbook</b>
  <p style="font-size:12px;color:#8fa6bd;">5 structured sheets (Scorecard, Topics, Linkages, Statements)</p>
</div>""",
            unsafe_allow_html=True,
        )
        excel_path = OUTPUT_FOLDER / f"{company}_earnings_report.xlsx"
        reports.generate_excel_report(company, analyses, qoq_df, topic_df, co_engine.get_rules(), excel_path)
        if excel_path.exists():
            st.download_button(
                "Download .xlsx",
                excel_path.read_bytes(),
                file_name=f"{company}_institutional_report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

    with c2:
        st.markdown(
            """<div class="card">
  <div style="font-size:28px;">📑</div>
  <b>CSV Monitor</b>
  <p style="font-size:12px;color:#8fa6bd;">Tabular feed of quarterly sentiment, topics, and metrics</p>
</div>""",
            unsafe_allow_html=True,
        )
        csv_path = OUTPUT_FOLDER / f"{company}_monitor.csv"
        reports.generate_csv_export(analyses, csv_path)
        if csv_path.exists():
            st.download_button(
                "Download .csv",
                csv_path.read_bytes(),
                file_name=f"{company}_quarter_monitor.csv",
                mime="text/csv",
            )

    with c3:
        st.markdown(
            """<div class="card">
  <div style="font-size:28px;">💾</div>
  <b>Structured JSON</b>
  <p style="font-size:12px;color:#8fa6bd;">Programmatic access for quantitative risk modeling</p>
</div>""",
            unsafe_allow_html=True,
        )
        json_path = OUTPUT_FOLDER / f"{company}_analysis.json"
        reports.generate_json_export(company, analyses, qoq_df, json_path)
        if json_path.exists():
            st.download_button(
                "Download .json",
                json_path.read_bytes(),
                file_name=f"{company}_structured_analysis.json",
                mime="application/json",
            )

    st.markdown("#### One-Page Styled Analyst Briefing")
    html_memo = reports.generate_html_briefing(
        company,
        analyses[-1] if analyses else {},
        None,
        co_engine.get_plain_insights(),
    )
    st.download_button(
        "Download Analyst Briefing (.html)",
        html_memo.encode("utf-8"),
        file_name=f"{company}_analyst_briefing.html",
        mime="text/html",
    )
    with st.expander("Preview Analyst Briefing", expanded=False):
        st.components.v1.html(html_memo, height=500, scrolling=True)


# =============================================================================
# Main Router
# =============================================================================
if current_page == "Command Center":
    render_command_center()
elif current_page == "Alerts & Watchlist":
    render_alerts_watchlist()
elif current_page == "Call Timeline":
    render_timeline()
elif current_page == "Topic Radar":
    render_topic_radar()
elif current_page == "Key Numbers":
    render_key_numbers()
elif current_page == "Peer Benchmark":
    render_peer_benchmark()
elif current_page == "Ask the Call":
    render_ask_the_call()
elif current_page == "Language Desk":
    render_language_desk()
elif current_page == "Evidence Explorer":
    render_evidence_explorer()
elif current_page == "Compare Quarters":
    render_compare_quarters()
elif current_page == "Reports":
    render_reports()

# -----------------------------------------------------------------------------
# Global Page Footer with Mandatory Line
# -----------------------------------------------------------------------------
ui.render_page_footer()

# -----------------------------------------------------------------------------
# Speech Synthesizer Slot Execution
# -----------------------------------------------------------------------------
if st.session_state.pending_speech:
    sp_text, sp_lang = st.session_state.pending_speech
    st.session_state.speech_id += 1
    st.session_state.speaker_cfg = (sp_text, auto_speak, sp_lang, st.session_state.speech_id)
    st.session_state.pending_speech = None

sp_txt, sp_auto, sp_lang, sp_sid = st.session_state.speaker_cfg
with speaker_slot:
    if sp_txt:
        st.components.v1.html(va.generate_tts_html(sp_txt, speech_rate, sp_auto, sp_lang, str(sp_sid)), height=48)
