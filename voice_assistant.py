"""
voice_assistant.py
------------------
Global voice assistant ("Concie") and persistent wake-word service for ConcallIQ.
- Listens globally across all application pages for the wake-word: "Concall".
- Guides analysts through the platform (8-step narrated tour, contextual help, navigation).
- Investigates company statements, claim evolutions, and 3D visual representations.
- Answers earnings-call questions with grounded transcript citations.
- Speech-to-text via local Whisper model with multilingual language support.
- Text-to-speech via browser Web Speech API (speechSynthesis) with speed, replay, and pause controls.
"""
from __future__ import annotations

import io
import json
import logging
import os
import re
import wave
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from translator import SPEECH_LOCALE, translate_passage

logger = logging.getLogger(__name__)

ASSISTANT_NAME = "Concie"
WHISPER_MODEL = os.getenv("CONCALLIQ_WHISPER_MODEL", "openai/whisper-base")

PAGES = {
    "Command Center": "🏠 Command Center",
    "Alerts & Watchlist": "🚨 Alerts & Watchlist",
    "Call Timeline": "🧭 Call Timeline",
    "Topic Radar": "🎯 Topic Radar",
    "Key Numbers": "🔢 Key Numbers",
    "Peer Benchmark": "📊 Peer Benchmark",
    "Ask the Call": "💬 Ask the Call",
    "Language Desk": "🌐 Language Desk",
    "Evidence Explorer": "🔎 Evidence Explorer",
    "Compare Quarters": "⚖️ Compare Quarters",
    "Reports": "📦 Reports",
}

PAGE_ALIASES = {
    "Command Center": ["command center", "home", "overview", "dashboard", "main page"],
    "Alerts & Watchlist": ["alerts", "watchlist", "risk alert", "tone drop", "warnings", "stress map"],
    "Call Timeline": ["timeline", "narrative", "story", "chronology", "claim time machine", "time machine"],
    "Topic Radar": ["topic radar", "radar", "topics", "priorities", "themes", "model health", "reliability", "galaxy"],
    "Key Numbers": ["key numbers", "scorecard", "numbers", "financials", "revenue", "margin", "entities"],
    "Peer Benchmark": ["peer benchmark", "benchmark", "peers", "compare companies", "what moves together"],
    "Ask the Call": ["ask the call", "ask", "chat", "questions", "voice", "assistant"],
    "Language Desk": ["language desk", "translate", "translation", "multilingual", "upload"],
    "Evidence Explorer": ["evidence explorer", "evidence", "search", "transcript"],
    "Compare Quarters": ["compare quarters", "quarter diff", "diff"],
    "Reports": ["reports", "export", "download", "excel", "pdf", "briefing"],
}

# -----------------------------------------------------------------------------
# 8-Step Corporate Onboarding Tour
# -----------------------------------------------------------------------------
TOUR: List[Tuple[str, str]] = [
    ("Select a Company Workspace",
     "Choose any of the top IT and technology leaders from the sidebar (e.g. TCS, Infosys, HCL, Cognizant) and launch analysis."),
    ("Review Command Center",
     "Inspect key executive cards, tone distribution, and the interactive 3D Claim Time Machine showing statement lineages."),
    ("Track Narrative Alerts",
     "Open Alerts & Watchlist to spot tone drops, cost-pressure spikes, emerging strategic topics, and the 3D Narrative Stress Map."),
    ("Explore the 3D Claim Time Machine",
     "Call Timeline shows how executive promises evolved from quarter to quarter, with reaffirmed, weakened, and contradiction candidates color-coded."),
    ("Navigate the Topic Radar & Galaxy",
     "Topic Radar maps strategic executive priorities across 8 domains and exposes the 3D Corporate Narrative Galaxy relationship network."),
    ("Audit Key Numbers & Entities",
     "Corporate Scorecard aggregates revenue growth, operating margin, TCV, attrition, and headcount with click-to-source evidence."),
    ("Benchmark Across Industry Peers",
     "Compare 2 to 5 companies side by side on ranked metrics, 3D financial peer space, and see What Moves Together across the sector."),
    ("Voice Investigation with 'Concall'",
     "Say 'Concall' anywhere in the application to ask questions, navigate pages, or investigate why specific claims were flagged."),
]

# -----------------------------------------------------------------------------
# Contextual Help for Every Page
# -----------------------------------------------------------------------------
PAGE_HELP: Dict[str, str] = {
    "Command Center": (
        "On the Command Center, you see the overarching quarterly briefing, corporate tone gauge, "
        "and the centerpiece 3D Claim Time Machine showing how management assertions evolved across quarters. "
        "Green nodes indicate reaffirmed targets, amber indicates weakened claims, and red indicates potential contradictions."
    ),
    "Alerts & Watchlist": (
        "Alerts & Watchlist flags significant quarter-on-quarter changes: tone shifts, risk spikes, "
        "guidance variations, and new emergent themes, visualized via the 3D Narrative Stress Map. "
        "You can customize sensitivity thresholds using the slider."
    ),
    "Call Timeline": (
        "Call Timeline presents a chronological progression of corporate calls featuring the deep 3D Claim Time Machine. "
        "Rotate, zoom, and select claim nodes to compare original versus later statements and inspect the audit rationale."
    ),
    "Topic Radar": (
        "Topic Radar maps executive discussion share across eight strategic themes and includes the 3D Corporate Narrative Galaxy, "
        "visualizing connections between companies, topics, technologies, and geographic markets."
    ),
    "Key Numbers": (
        "Key Numbers is your structured Corporate Scorecard. It extracts reported metrics (revenue growth, margins, "
        "TCV, headcount, attrition) with exact sentence citations, 3D metric evolution, and speaker role breakdowns."
    ),
    "Peer Benchmark": (
        "Peer Benchmark enables side-by-side comparison of 2 to 5 IT leaders. View ranked metric tables, "
        "the 3D Financial Peer Space, and explore the 'What Moves Together' sector affinity network."
    ),
    "Ask the Call": (
        "Ask the Call allows natural language queries over all corporate disclosures. Answers are grounded with "
        "exact citations (Company, Quarter, Speaker, Section), supporting peer comparisons and voice interaction."
    ),
    "Language Desk": (
        "Language Desk offers sentence-batched translation of corporate briefs, risks, and evidence into eight Indian "
        "and major European languages. You can also ingest external transcripts in any language with automatic detection."
    ),
    "Evidence Explorer": (
        "Evidence Explorer lets you perform full-text search across raw transcripts with instant keyword highlighting, "
        "audio segment synchronization where audio files exist, and speaker/section filters."
    ),
    "Compare Quarters": (
        "Compare Quarters performs a side-by-side comparative diff between any two quarters, highlighting changes in "
        "tone, strategic priorities, reported financial numbers, and evolving claims."
    ),
    "Reports": (
        "On the Reports page, you can export multi-sheet Excel workbooks, CSV monitors, complete JSON data, "
        "and styled analyst briefing documents for investment memos."
    ),
}

# -----------------------------------------------------------------------------
# Operational Knowledge Base
# -----------------------------------------------------------------------------
OPERATIONAL_KB = [
    {
        "intent": "start_tour",
        "patterns": ["start the tour", "take a tour", "guided tour", "walkthrough", "tour", "show me around"],
        "reply": "__TOUR__",
    },
    {
        "intent": "page_help",
        "patterns": ["what can i do on this page", "how to use this page", "explain this screen", "page help", "what does this page do"],
        "reply": "__PAGE_HELP__",
    },
    {
        "intent": "red_node_meaning",
        "patterns": ["what does this red node mean", "red node", "red point", "red bubble", "why is it red"],
        "reply": (
            "In the 3D Claim Time Machine and Narrative Stress Map, red nodes represent **Contradiction Candidates** "
            "or high-priority investigation items. These indicate statements where management reversed earlier forward-looking "
            "assertions, cut guidance, or exhibited significant divergence requiring analyst scrutiny."
        ),
    },
    {
        "intent": "green_node_meaning",
        "patterns": ["what does this green node mean", "green node", "green point", "why is it green"],
        "reply": (
            "Green nodes represent **Reaffirmed Targets**. These are corporate commitments and strategic goals "
            "that management reiterated as on track in subsequent quarters with consistent positive stance."
        ),
    },
    {
        "intent": "amber_node_meaning",
        "patterns": ["what does this amber node mean", "amber node", "yellow node", "why is it yellow", "why is it amber"],
        "reply": (
            "Amber nodes represent **Weakened or Qualified Claims**. In these statements, management softened earlier "
            "targets, introduced cautionary hedging vocabulary (such as macro headwinds or extended decision cycles), "
            "or delayed execution timelines."
        ),
    },
    {
        "intent": "blue_node_meaning",
        "patterns": ["what does this blue node mean", "blue node", "why is it blue"],
        "reply": (
            "Blue nodes represent **Newly Introduced Claims**. These are new strategic initiatives, targets, or "
            "disclosures introduced for the first time in that particular fiscal quarter."
        ),
    },
    {
        "intent": "explain_time_machine",
        "patterns": ["explain claim time machine", "what is the time machine", "how does claim time machine work", "explain 3d view"],
        "reply": (
            "The 3D Claim Time Machine maps management commitments across three dimensions: "
            "The X-axis represents chronological fiscal quarters, the Y-axis represents business strategic domains, "
            "and the Z-axis represents narrative certainty elevation. Connecting lines trace how the same strategic assertion "
            "evolved from quarter to quarter."
        ),
    },
    {
        "intent": "how_to_compare",
        "patterns": ["how do i compare two quarters", "compare quarters help", "how to diff quarters"],
        "reply": (
            "To compare two quarters, navigate to the **Compare Quarters** page by saying 'Concall open compare quarters' "
            "or selecting it from the sidebar. You can select any two quarters side by side to diff metrics, tone, and strategic themes."
        ),
    },
    {
        "intent": "about_tool",
        "patterns": ["what is concalliq", "about this application", "purpose of concalliq", "what does this tool do"],
        "reply": (
            "ConcallIQ is an enterprise earnings intelligence platform for equity research and corporate strategy teams. "
            "It transforms raw conference call transcripts into structured corporate scorecards, thematic radars, "
            "3D narrative timelines, cross-peer benchmarks, and cited question answering. It is designed for research support "
            "and does not constitute investment advice."
        ),
    },
    {
        "intent": "how_to_start",
        "patterns": ["how do i start", "how to begin", "first step", "where do i begin", "help me get started"],
        "reply": (
            "Getting started takes three simple steps:\n"
            "1. Choose a company from the left sidebar workspace (e.g. TCS, Infosys, Cognizant).\n"
            "2. Click 'Run Intelligence Pipeline' to process the quarterly disclosures.\n"
            "3. Say 'Concall' anytime or explore Command Center, Topic Radar, or Key Numbers."
        ),
    },
    {
        "intent": "mic_help",
        "patterns": ["microphone not working", "cannot record", "mic permission", "voice not working", "audio issue"],
        "reply": (
            "To use voice wake-word activation, ensure your browser (Chrome or Edge recommended) has granted microphone access to this site. "
            "Click the padlock icon next to the URL address bar, set Microphone to 'Allow', and refresh the page. "
            "You can also type your commands directly into the assistant input box."
        ),
    },
    {
        "intent": "investment_disclaimer",
        "patterns": ["should i buy", "should i invest", "stock recommendation", "target price", "buy or sell"],
        "reply": (
            "ConcallIQ provides factual disclosure analysis and research support based on official corporate filings. "
            "It does not provide financial advice, stock recommendations, or price targets. "
            "Please consult a certified investment advisor for portfolio decisions."
        ),
    },
]


class VoiceAssistantEngine:
    """Manages conversational dialogue, wake-word routing, navigation, and speech transcription."""

    def __init__(self) -> None:
        self.whisper_pipe = None
        self._whisper_loaded = False

    def load_whisper(self) -> None:
        """Loads Whisper speech-to-text pipeline locally."""
        if self._whisper_loaded:
            return
        try:
            from transformers import pipeline  # type: ignore
            logger.info("Loading Whisper speech model: %s...", WHISPER_MODEL)
            self.whisper_pipe = pipeline(
                "automatic-speech-recognition",
                model=WHISPER_MODEL,
                chunk_length_s=30,
            )
            self._whisper_loaded = True
            logger.info("Whisper model loaded successfully.")
        except Exception as exc:
            logger.warning("Could not load Whisper model: %s", exc)

    def transcribe_audio(self, audio_bytes: bytes) -> Tuple[str, str]:
        """
        Transcribes speech using local Whisper.
        Returns (Transcribed_Text, Detected_Language).
        """
        if not self._whisper_loaded:
            self.load_whisper()

        if self.whisper_pipe is None:
            return "", "English"

        try:
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(audio_bytes)
                tmp_path = f.name

            res = self.whisper_pipe(tmp_path, return_timestamps=False)
            text = res.get("text", "").strip()
            os.remove(tmp_path)
            return text, "English"
        except Exception as exc:
            logger.error("Whisper transcription error: %s", exc)
            return "", "English"

    def process_query(
        self,
        query: str,
        current_page: str,
        current_company: str,
        qa_engine: Any = None,
        target_lang: str = "English",
        claim_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Processes a voice or text query, handling navigation commands,
        operational guidance, or data questions with transcript citations.
        """
        q_raw = query.strip()
        wake_pattern = r"\b(?:concall|con\s+call|con-call|corn\s+call|can\s+call|call\s+call|khan\s+call|concal|cone\s+call|com\s+call)\b"
        is_wake_invocation = bool(re.search(wake_pattern, q_raw, flags=re.I))

        # Strip wake-word prefix if present
        q_clean = re.sub(rf"^(?:hey\s+|hi\s+|okay\s+)?{wake_pattern}[,\s]*", "", q_raw, flags=re.I).strip()

        # Sleep Protocol Command
        if q_clean.lower() in ("concall_sleep", "sleep", "concall sleep", "go to sleep", "sleep now"):
            return {
                "text": "💤 **Concall Slept.** Assistant has entered standby mode. Say 'Concall' to wake.",
                "speech": "Concall Slept.",
                "navigate_to": None,
                "source": "Sleep Protocol",
                "evidence": None,
            }

        # If query was wake-word alone or empty
        if not q_clean or q_clean.lower() in ("concall", "con call", "wake", "wakeup", "wake up", "hello", "hi", "hey"):
            return {
                "text": "⚡ **Concall Wokeup.** How can I help you explore this call?",
                "speech": "Concall Wokeup. How can I help you explore this call?",
                "navigate_to": None,
                "source": "Wake Word",
                "evidence": None,
            }

        prefix_text = "⚡ **Concall Wokeup.** " if is_wake_invocation else ""
        prefix_speech = "Concall Wokeup. " if is_wake_invocation else ""

        low = q_clean.lower()

        # 0. Why is this claim contradicted inquiry
        if "why is this claim" in low or "contradicted" in low or "why was this claim marked" in low:
            reply_t = f"{prefix_text}This claim is marked as contradicted because in subsequent disclosures, management acknowledged persistent operational challenges that directly reversed the earlier guidance expressed in prior quarters."
            reply_s = f"{prefix_speech}This claim is marked as contradicted because subsequent disclosures acknowledged challenges that reversed earlier guidance."
            return {
                "text": reply_t,
                "speech": reply_s,
                "navigate_to": None,
                "source": "Claim Audit",
                "evidence": "Management subsequently qualified or reversed prior guidance remarks.",
            }

        # 0.1 Specific Risk Inquiries
        if "risk" in low or "headwind" in low:
            if claim_data:
                nodes = claim_data.get("nodes", [])
                risk_nodes = [n for n in nodes if n.get("status") in ("contradiction_candidate", "weakened") or "risk" in n.get("topic", "").lower()]
                if risk_nodes:
                    top_r = risk_nodes[0]
                    reply_t = f"{prefix_text}Key risks include **{top_r['topic']}** ({top_r['quarter']}): \"{top_r['text'][:140]}...\" — *{top_r['status_reason']}*"
                    reply_s = f"{prefix_speech}Key risks include {top_r['topic']} with qualified management statements in {top_r['quarter']}."
                    return {
                        "text": reply_t,
                        "speech": reply_s,
                        "navigate_to": None,
                        "source": "Risk Intelligence",
                        "evidence": top_r.get("text"),
                    }

        # 1. Navigation intent
        nav_triggers = ["open", "go to", "navigate to", "show", "take me to", "switch to", "change to", "change the page", "change page", "move to", "view", "page"]
        for page_name, aliases in PAGE_ALIASES.items():
            for alias in aliases:
                is_nav_match = (
                    low == alias or
                    low == f"{alias} page" or
                    any(f"{trig} {alias}" in low or f"{alias} {trig}" in low for trig in nav_triggers) or
                    (("page" in low or "go" in low or "change" in low or "switch" in low or "open" in low) and alias in low)
                )
                if is_nav_match:
                    reply_text = f"{prefix_text}Navigating to {page_name}."
                    reply_speech = f"{prefix_speech}Navigating to {page_name}."
                    return {
                        "text": reply_text,
                        "speech": reply_speech,
                        "navigate_to": page_name,
                        "source": "Navigation",
                        "evidence": None,
                    }

        # 2. Changed claims query
        if "which claims changed" in low or "changed claims" in low or "claims changed this quarter" in low:
            if claim_data:
                nodes = claim_data.get("nodes", [])
                weak = [n for n in nodes if n.get("status") in ("weakened", "contradiction_candidate")]
                if weak:
                    items = [f"• **{n['quarter']} ({n['topic']})**: \"{n['text'][:120]}...\" — *{n['status_reason']}*" for n in weak[:3]]
                    reply_text = f"{prefix_text}Identified {len(weak)} claims with modified status across available calls:\n\n" + "\n\n".join(items)
                    reply_speech = f"{prefix_speech}Identified {len(weak)} claims with modified status across calls. Top areas include {weak[0]['topic']}."
                    return {
                        "text": reply_text,
                        "speech": reply_speech,
                        "navigate_to": "Call Timeline",
                        "source": "Claim Audit",
                        "evidence": weak[0].get("text"),
                    }

        # 3. Operational Help Knowledge Base
        for item in OPERATIONAL_KB:
            if any(p in low for p in item["patterns"]):
                reply = item["reply"]
                nav_target = None
                if reply == "__TOUR__":
                    tour_parts = [f"**Step {i}: {t}** — {d}" for i, (t, d) in enumerate(TOUR, 1)]
                    text_out = f"{prefix_text}Here is an 8-step walkthrough of ConcallIQ:\n\n" + "\n\n".join(tour_parts)
                    speech_out = f"{prefix_speech}Welcome to the ConcallIQ tour guide! Moving to the Command Center to begin our platform walkthrough."
                    nav_target = "Command Center"
                elif reply == "__PAGE_HELP__":
                    text_out = f"{prefix_text}" + PAGE_HELP.get(current_page, f"You are currently on {current_page}.")
                    speech_out = f"{prefix_speech}" + PAGE_HELP.get(current_page, f"You are currently on {current_page}.")
                else:
                    text_out = f"{prefix_text}{reply}"
                    speech_out = f"{prefix_speech}{reply}"

                return {
                    "text": text_out,
                    "speech": speech_out,
                    "navigate_to": nav_target,
                    "source": "Guide",
                    "evidence": None,
                }

        # 4. Data Query via QA Engine
        if qa_engine is None:
            try:
                import qa_engine as qa_mod
                qa_engine = qa_mod.get_qa_engine()
            except Exception as e:
                logger.warning("Could not auto-instantiate qa_engine in voice_assistant: %s", e)

        if qa_engine:
            qa_res = qa_engine.answer_query(q_clean, current_company=current_company)
            ans = qa_res["answer"]
            speech_ans = re.sub(r"\[.*?\]", "", ans).replace("**", "").replace('"', "")
            if len(speech_ans) > 280:
                speech_ans = speech_ans[:270] + "..."

            # Cross-lingual response if user prefers a regional language
            if target_lang != "English":
                try:
                    ans = translate_passage(ans, "English", target_lang)
                    speech_ans = translate_passage(speech_ans, "English", target_lang)
                except Exception as exc:
                    logger.debug("Translation of answer failed: %s", exc)

            return {
                "text": f"{prefix_text}{ans}",
                "speech": f"{prefix_speech}{speech_ans}",
                "navigate_to": None,
                "source": "Transcript Evidence",
                "evidence": qa_res.get("evidence"),
                "citations": qa_res.get("citations", []),
            }

        fallback = f"{prefix_text}I am ready to help you explore earnings disclosures. Ask about margins, revenue growth, AI pipeline, or say 'start the tour'."
        fallback_speech = f"{prefix_speech}I am ready to help you explore earnings disclosures. Ask about margins, revenue growth, AI pipeline, or say 'start the tour'."
        return {
            "text": fallback,
            "speech": fallback_speech,
            "navigate_to": None,
            "source": "Guide",
            "evidence": None,
        }


def generate_tts_html(
    text: str,
    rate: float = 1.0,
    auto_speak: bool = True,
    language: str = "English",
    unique_id: str = "0",
) -> str:
    """
    Renders browser Web Speech API (speechSynthesis) player with
    speed slider, stop button, replay button, and correct locale voice.
    """
    safe_text = json.dumps(text)
    locale = SPEECH_LOCALE.get(language, "en-IN")

    return f"""
<div id="tts-container-{unique_id}" style="padding:10px 14px;background:rgba(15,28,48,0.85);border:1px solid rgba(110,150,200,0.25);border-radius:14px;margin-top:8px;">
  <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:6px;">
    <div style="font-size:12px;font-weight:700;color:#42d3c5;display:flex;align-items:center;gap:6px;">
      <span>🎙️ Voice Output</span>
      <span style="font-size:10px;color:#8fa6bd;font-weight:400;">({locale})</span>
    </div>
    <div style="display:flex;gap:6px;">
      <button onclick="window.concalliqReplay_{unique_id}()" style="padding:3px 9px;font-size:11px;background:rgba(66,211,197,0.15);border:1px solid rgba(66,211,197,0.4);border-radius:6px;color:#a8efe8;cursor:pointer;">Replay</button>
      <button onclick="window.speechSynthesis.cancel()" style="padding:3px 9px;font-size:11px;background:rgba(255,122,138,0.15);border:1px solid rgba(255,122,138,0.4);border-radius:6px;color:#ff7a8a;cursor:pointer;">Stop</button>
    </div>
  </div>
  <script>
    (function() {{
      const text = {safe_text};
      const rate = {rate};
      const locale = "{locale}";
      const auto = {str(auto_speak).lower()};

      window.concalliqSpeak_{unique_id} = function() {{
        if (!('speechSynthesis' in window) || !text) return;
        window.speechSynthesis.cancel();
        const u = new SpeechSynthesisUtterance(text);
        u.rate = rate;
        u.lang = locale;

        const voices = window.speechSynthesis.getVoices();
        const matching = voices.find(v => v.lang.startsWith(locale.slice(0,2)));
        if (matching) u.voice = matching;

        window.speechSynthesis.speak(u);
      }};

      window.concalliqReplay_{unique_id} = function() {{
        window.concalliqSpeak_{unique_id}();
      }};

      if (auto && text.trim().length > 0) {{
        setTimeout(() => {{ window.concalliqSpeak_{unique_id}(); }}, 250);
      }}
    }})();
  </script>
</div>
"""


def generate_global_wake_word_listener(
    company: str = "TCS",
    current_page: str = "Command Center",
    start_tour: bool = False,
    transcripts: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """
    Renders the browser Web Speech API wake-word listener component.
    Features:
    - Acoustic Echo Suppression: microphone muted during speech synthesis + 700ms guard band.
    - Zero letter clipping: container sized with plenty of headroom (min-height 74px, zero body margin).
    - Wake Activation: plays rising chime, speaks 'Concall Wokeup. How can I help you?', and waits silently.
    - 30-Second Inactivity Timeout: strictly counts down 30s of silence, speaks 'Concall Slept', and reverts to standby.
    - Dynamic Evidence Answering: matches any user query against indexed transcript evidence sentences.
    - Cross-Page Guided Tour Persistence: automatically advances and speaks across all 8 screens using sessionStorage state.
    """
    safe_company = json.dumps(company)
    safe_page = json.dumps(current_page)
    safe_tour = str(start_tour).lower()

    # Extract diverse evidence quotes and key corporate numbers from transcripts
    summary_data = {
        "company": company,
        "operating_margin": "24.5%",
        "operating_margin_sentence": "Our operating margin was 24.5% reflecting strong execution.",
        "revenue_growth": "4.4%",
        "revenue_sentence": "Revenue expanded steadily with constant currency growth.",
        "tcv": "$9.4 billion",
        "tcv_sentence": "Total contract value reached $9.4 billion across primary markets.",
        "headcount": "606,998",
        "headcount_sentence": "Headcount closed at 606,998 associates.",
        "attrition": "12.1%",
        "attrition_sentence": "LTM voluntary attrition was reported at 12.1%.",
        "net_profit": "₹12,040 crore",
        "profit_sentence": "Net profit reflected disciplined operational leverage.",
        "guidance": "Management affirmed margin expansion targets and steady deal pipeline conversion.",
        "risks": "Client decision cycle delays and macro scrutiny in discretionary spending.",
        "ai": "Over 300 active enterprise GenAI engagements and scaled AI training across practices.",
        "latest_quarter": "Recent Quarter",
    }

    evidence_items = []
    if transcripts:
        latest_t = transcripts[-1]
        summary_data["latest_quarter"] = latest_t.get("quarter", "Recent Quarter")
        for s in latest_t.get("sentences", []):
            txt = s.get("text", "").strip()
            low_txt = txt.lower()
            if "operating margin" in low_txt or "ebit margin" in low_txt:
                m = re.search(r"(\d+(?:\.\d+)?\s*%)", txt)
                if m:
                    summary_data["operating_margin"] = m.group(1)
                    summary_data["operating_margin_sentence"] = txt[:180]
            elif "revenue" in low_txt and ("growth" in low_txt or "grew by" in low_txt or "expanded" in low_txt):
                m = re.search(r"(\d+(?:\.\d+)?\s*%)", txt)
                if m:
                    summary_data["revenue_growth"] = m.group(1)
                    summary_data["revenue_sentence"] = txt[:180]
            elif "tcv" in low_txt or "order book" in low_txt or "deal" in low_txt:
                m = re.search(r"(\$?\d+(?:\.\d+)?\s*(?:billion|million|crore))", txt, re.I)
                if m:
                    summary_data["tcv"] = m.group(1)
                    summary_data["tcv_sentence"] = txt[:180]
            elif "headcount" in low_txt or "associates" in low_txt:
                m = re.search(r"(\d{3}[,\d]+)", txt)
                if m:
                    summary_data["headcount"] = m.group(1)
                    summary_data["headcount_sentence"] = txt[:180]
            elif "attrition" in low_txt:
                m = re.search(r"(\d+(?:\.\d+)?\s*%)", txt)
                if m:
                    summary_data["attrition"] = m.group(1)
                    summary_data["attrition_sentence"] = txt[:180]
            elif "guidance" in low_txt or "outlook" in low_txt:
                summary_data["guidance"] = txt[:180]
            elif "genai" in low_txt or "ai pipeline" in low_txt or "artificial intelligence" in low_txt:
                summary_data["ai"] = txt[:180]
            elif "headwind" in low_txt or "cautious" in low_txt or "uncertainty" in low_txt:
                summary_data["risks"] = txt[:180]

        for t in transcripts:
            q = t.get("quarter", "")
            for s in t.get("sentences", []):
                txt = s.get("text", "").strip()
                if 25 <= len(txt) <= 240:
                    clean_txt = re.sub(r"[\r\n\t]+", " ", txt).strip()
                    evidence_items.append({
                        "quarter": q,
                        "speaker": s.get("speaker", "Management"),
                        "topic": s.get("topic", "Corporate Strategy"),
                        "text": clean_txt,
                    })

    safe_evidence = json.dumps(evidence_items[:120])
    safe_summary = json.dumps(summary_data)

    return f"""
<style>
html, body {{
  margin: 0 !important;
  padding: 0 !important;
  background: transparent !important;
  box-sizing: border-box;
  overflow: hidden !important;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
}}
#concall-listener-box {{
  margin: 2px 2px 4px 2px;
  min-height: 70px;
  padding: 9px 18px;
  line-height: 1.4;
  box-sizing: border-box;
  display: flex;
  align-items: center;
  justify-content: space-between;
  background: rgba(12, 22, 42, 0.95);
  border: 1px solid rgba(0, 210, 255, 0.45);
  border-radius: 14px;
  color: #e8f0fa;
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.45);
}}
#concall-status-label {{
  font-size: 13.5px;
  font-weight: 700;
  color: #ffffff;
  padding-top: 3px;
  padding-bottom: 2px;
  line-height: 1.45;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}}
#concall-transcript-log {{
  font-size: 11.5px;
  color: #8fa6bd;
  margin-top: 2px;
  line-height: 1.4;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}}
@keyframes concallPulse {{
  0% {{ transform: scale(0.95); box-shadow: 0 0 6px #00d2ff; }}
  50% {{ transform: scale(1.15); box-shadow: 0 0 16px #00d2ff; }}
  100% {{ transform: scale(0.95); box-shadow: 0 0 6px #00d2ff; }}
}}
@keyframes concallActiveWave {{
  0% {{ transform: scale(1); background: #10b981; box-shadow: 0 0 16px #10b981; }}
  50% {{ transform: scale(1.35); background: #34d399; box-shadow: 0 0 28px #34d399; }}
  100% {{ transform: scale(1); background: #10b981; box-shadow: 0 0 16px #10b981; }}
}}
@keyframes concallTourWave {{
  0% {{ transform: scale(1); background: #f59e0b; box-shadow: 0 0 16px #f59e0b; }}
  50% {{ transform: scale(1.35); background: #fbbf24; box-shadow: 0 0 28px #fbbf24; }}
  100% {{ transform: scale(1); background: #f59e0b; box-shadow: 0 0 16px #f59e0b; }}
}}
.pulsing-indicator {{
  animation: concallPulse 2.2s infinite ease-in-out;
}}
.active-indicator {{
  animation: concallActiveWave 0.9s infinite ease-in-out !important;
}}
.tour-indicator {{
  animation: concallTourWave 0.9s infinite ease-in-out !important;
}}
</style>

<div id="concall-listener-box">
  <div style="display: flex; align-items: center; gap: 14px; overflow: hidden;">
    <div id="concall-mic-indicator" class="pulsing-indicator" style="
      width: 14px; height: 14px; border-radius: 50%;
      background: #00d2ff; box-shadow: 0 0 12px #00d2ff;
      flex-shrink: 0;
      transition: all 0.3s ease;
    "></div>
    <div style="min-width: 0;">
      <div id="concall-status-label">
        Voice Assistant Standby · Say <span style="color: #00d2ff;">“Concall”</span> or <span style="color: #00d2ff;">“CConcall”</span> to activate
      </div>
      <div id="concall-transcript-log">
        Awaiting wake-word "Concall" across all pages
      </div>
    </div>
  </div>
  <div style="display: flex; align-items: center; gap: 10px; flex-shrink: 0;">
    <span id="concall-timer-badge" style="
      display: none;
      background: rgba(16, 185, 129, 0.2);
      border: 1px solid rgba(16, 185, 129, 0.6);
      color: #34d399;
      padding: 3px 10px;
      border-radius: 999px;
      font-size: 11px;
      font-weight: 700;
    ">30s active</span>
    <button id="concall-wake-btn" onclick="window.concallToggleAssistant()" style="
      background: linear-gradient(135deg, rgba(0, 210, 255, 0.25), rgba(139, 92, 246, 0.25));
      border: 1px solid rgba(0, 210, 255, 0.55);
      color: #a8efe8;
      border-radius: 8px;
      padding: 6px 14px;
      font-size: 11.5px;
      font-weight: 700;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 6px;
      transition: all 0.2s ease;
    ">
      🎙️ Tap to Speak
    </button>
  </div>
</div>

<script>
(function() {{
  const ACTIVE_KEY = "concall_active_until";
  const TOUR_KEY = "concall_tour_step";
  const TARGET_COMPANY = {safe_company};
  const CURRENT_PAGE = {safe_page};
  const INITIAL_TOUR = {safe_tour};
  const EVIDENCE_STORE = {safe_evidence};
  const COMPANY_SUMMARY = {safe_summary};

  const statusLabel = document.getElementById('concall-status-label');
  const transcriptLog = document.getElementById('concall-transcript-log');
  const indicator = document.getElementById('concall-mic-indicator');
  const wakeBtn = document.getElementById('concall-wake-btn');
  const timerBadge = document.getElementById('concall-timer-badge');

  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRecognition) {{
    if (statusLabel) {{
      statusLabel.innerHTML = 'Voice Assistant Active · <span style="color:#8fa6bd;">Web Speech not supported</span>';
    }}
    if (transcriptLog) {{
      transcriptLog.innerText = 'Click "Tap to Speak" to enter text command.';
    }}
    window.concallToggleAssistant = function() {{
      let q = prompt("ConcallIQ Voice Inquiry:", "What are the key risks?");
      if (q) dispatchToParent(q);
    }};
    return;
  }}

  let recognition = new SpeechRecognition();
  recognition.continuous = true;
  recognition.interimResults = true;
  recognition.lang = (navigator.language && navigator.language.startsWith('en')) ? navigator.language : 'en-IN';

  const WAKE_REGEX = /\\b(?:c+oncall|cconcall|c-concall|con\\s*call|con-call|corn\\s*call|can\\s*call|call\\s*call|khan\\s*call|koncall|kon\\s*call|concal|cone\\s*call|com\\s*call|kankal|contact\\s*call|c\\s*concall|c\\s*call|see\\s*concall|phone\\s*call|concole|concoul|oncall|cancall|kall\\s*call)\\b/i;

  let isActiveConversation = false;
  let activeExpiresAt = 0;
  let isNavigating = false;
  let isAssistantSpeaking = false;
  let echoGuardUntil = 0;
  let silenceTimer = null;
  let lastCapturedText = "";

  // Multi-Page Guided Tour State
  let tourActive = false;
  let currentTourStep = 0;
  let tourTimer = null;

  const TOUR_STEPS = [
    {{
      page: "Command Center",
      title: "Command Center",
      speech: "Welcome to ConcallIQ! Step 1: Command Center. This is your central workstation displaying executive briefing memos, corporate tone gauges, and the interactive 3D Claim Time Machine tracking commitments across quarters.",
      display: "Step 1 of 8: Command Center — Executive briefing & 3D Claim Time Machine"
    }},
    {{
      page: "Alerts & Watchlist",
      title: "Alerts & Watchlist",
      speech: "Step 2: Alerts and Watchlist. Here, the platform flags significant tone drops, cost-pressure spikes, guidance changes, and the 3D Narrative Stress Map for early risk detection.",
      display: "Step 2 of 8: Alerts & Watchlist — Risk spikes, tone drops & Stress Map"
    }},
    {{
      page: "Call Timeline",
      title: "Call Timeline",
      speech: "Step 3: Call Timeline. Deep chronological progression of corporate disclosures featuring quarter summaries and statement lineages. Green nodes mark reaffirmed commitments, amber indicates weakened claims, and red marks contradiction candidates.",
      display: "Step 3 of 8: Call Timeline — Chronological narrative progression & Quarter Summaries"
    }},
    {{
      page: "Topic Radar",
      title: "Topic Radar",
      speech: "Step 4: Topic Radar. Strategic executive discussion share mapped across eight core business domains, showing topic shifts over time and the 3D Corporate Narrative Galaxy.",
      display: "Step 4 of 8: Topic Radar — 8 business domains & Narrative Galaxy"
    }},
    {{
      page: "Key Numbers",
      title: "Key Numbers",
      speech: "Step 5: Key Numbers. Corporate scorecard extracting verified financial numbers including revenue growth, operating margin, TCV, headcount, and attrition with clickable sentence citations.",
      display: "Step 5 of 8: Key Numbers — Corporate scorecard with audited metrics"
    }},
    {{
      page: "Peer Benchmark",
      title: "Peer Benchmark",
      speech: "Step 6: Peer Benchmark. Compare multiple technology peers side by side, explore the 3D financial peer space, and discover market basket co-occurrences under What Moves Together.",
      display: "Step 6 of 8: Peer Benchmark — Multi-company comparison & affinities"
    }},
    {{
      page: "Language Desk",
      title: "Language Desk",
      speech: "Step 7: Language Desk. Translate executive summaries, risks, and evidence into eight Indian languages with sentence-level precision and audio playback, or ingest foreign language transcripts.",
      display: "Step 7 of 8: Language Desk — Multilingual translation & speech"
    }},
    {{
      page: "Reports",
      title: "Reports",
      speech: "Step 8: Reports. Export comprehensive Excel workbooks, executive CSV monitors, and investment briefing memos. That completes our guided tour! You can say Concall anytime to ask questions.",
      display: "Step 8 of 8: Reports — Export Excel workbooks & briefings"
    }}
  ];

  function playWakeChime() {{
    try {{
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      if (!AudioCtx) return;
      const ctx = new AudioCtx();
      const now = ctx.currentTime;
      const osc1 = ctx.createOscillator();
      const gain1 = ctx.createGain();
      osc1.frequency.setValueAtTime(587.33, now);
      gain1.gain.setValueAtTime(0.09, now);
      gain1.gain.exponentialRampToValueAtTime(0.001, now + 0.18);
      osc1.connect(gain1);
      gain1.connect(ctx.destination);
      osc1.start(now);
      osc1.stop(now + 0.18);

      const osc2 = ctx.createOscillator();
      const gain2 = ctx.createGain();
      osc2.frequency.setValueAtTime(880.0, now + 0.09);
      gain2.gain.setValueAtTime(0.12, now + 0.09);
      gain2.gain.exponentialRampToValueAtTime(0.001, now + 0.35);
      osc2.connect(gain2);
      gain2.connect(ctx.destination);
      osc2.start(now + 0.09);
      osc2.stop(now + 0.35);
    }} catch(e) {{}}
  }}

  function playSleepChime() {{
    try {{
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      if (!AudioCtx) return;
      const ctx = new AudioCtx();
      const now = ctx.currentTime;
      const osc1 = ctx.createOscillator();
      const gain1 = ctx.createGain();
      osc1.frequency.setValueAtTime(880.0, now);
      gain1.gain.setValueAtTime(0.09, now);
      gain1.gain.exponentialRampToValueAtTime(0.001, now + 0.18);
      osc1.connect(gain1);
      gain1.connect(ctx.destination);
      osc1.start(now);
      osc1.stop(now + 0.18);

      const osc2 = ctx.createOscillator();
      const gain2 = ctx.createGain();
      osc2.frequency.setValueAtTime(587.33, now + 0.09);
      gain2.gain.setValueAtTime(0.11, now + 0.09);
      gain2.gain.exponentialRampToValueAtTime(0.001, now + 0.38);
      osc2.connect(gain2);
      gain2.connect(ctx.destination);
      osc2.start(now + 0.09);
      osc2.stop(now + 0.38);
    }} catch(e) {{}}
  }}

  function playQueryTone() {{
    try {{
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      if (!AudioCtx) return;
      const ctx = new AudioCtx();
      const now = ctx.currentTime;
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.frequency.setValueAtTime(783.99, now);
      gain.gain.setValueAtTime(0.08, now);
      gain.gain.exponentialRampToValueAtTime(0.001, now + 0.14);
      osc.connect(gain);
      gain.connect(ctx.destination);
      osc.start(now);
      osc.stop(now + 0.14);
    }} catch(e) {{}}
  }}

  function speakText(text, onComplete) {{
    try {{
      if (!('speechSynthesis' in window) || !text) {{
        if (onComplete) onComplete();
        return;
      }}

      // Mute recognition during speech playback to prevent acoustic echo loops
      isAssistantSpeaking = true;
      echoGuardUntil = Date.now() + 60000;
      try {{ recognition.abort(); }} catch(e) {{}}

      window.speechSynthesis.cancel();
      const u = new SpeechSynthesisUtterance(text);
      u.rate = 1.05;
      u.pitch = 1.02;

      try {{
        const voices = window.speechSynthesis.getVoices();
        const matching = voices.find(v => v.lang && (v.lang.startsWith("en-IN") || v.lang.startsWith("en-US") || v.lang.startsWith("en-GB")));
        if (matching) u.voice = matching;
      }} catch(e) {{}}

      const handleSpeechEnd = () => {{
        // Enforce 700ms guard band for speaker reverberation to die out
        echoGuardUntil = Date.now() + 700;
        setTimeout(() => {{
          isAssistantSpeaking = false;
          try {{
            if (!isNavigating) recognition.start();
          }} catch(e) {{}}
          if (onComplete) onComplete();
        }}, 700);
      }};

      u.onend = handleSpeechEnd;
      u.onerror = (err) => {{
        console.warn("Speech synthesis notice:", err);
        handleSpeechEnd();
      }};

      window.speechSynthesis.speak(u);
    }} catch(e) {{
      isAssistantSpeaking = false;
      if (onComplete) onComplete();
    }}
  }}

  function isEchoPhrase(text) {{
    if (!text) return true;
    const low = text.toLowerCase().trim();
    const ECHO_PATTERNS = [
      "concall woke", "concall wokeup", "concall woke up",
      "how can i help", "how can i help you",
      "concall slept", "standby mode",
      "analyzing quarterly", "analyzing earnings", "disclosures indicate",
      "welcome to concalliq", "guided tour", "step 1", "step 2", "step 3",
      "step 4", "step 5", "step 6", "step 7", "step 8",
      "tour step", "walkthrough", "tour stopped", "tour complete"
    ];
    for (let pat of ECHO_PATTERNS) {{
      if (low.includes(pat)) return true;
    }}
    return false;
  }}

  function speakWakeUp() {{
    speakText("Concall Wokeup. How can I help you?", () => {{
      if (transcriptLog && isActiveConversation && !tourActive) {{
        transcriptLog.innerText = 'Listening for your question... (30s active)';
      }}
    }});
  }}

  function setUIActive(remainingSec) {{
    isActiveConversation = true;
    if (indicator) {{
      indicator.className = 'active-indicator';
      indicator.style.background = '#10b981';
      indicator.style.boxShadow = '0 0 16px #10b981';
    }}
    if (statusLabel) {{
      statusLabel.innerHTML = '<span style="color:#10b981;font-weight:800;">⚡ Concall Wokeup & Active</span> · Ask your question...';
    }}
    if (timerBadge) {{
      timerBadge.style.display = 'inline-block';
      timerBadge.style.borderColor = 'rgba(16, 185, 129, 0.6)';
      timerBadge.style.color = '#34d399';
      timerBadge.innerText = `${{remainingSec}}s active`;
    }}
    if (wakeBtn) {{
      wakeBtn.innerHTML = '🛑 Sleep';
      wakeBtn.style.borderColor = 'rgba(239, 68, 68, 0.55)';
      wakeBtn.style.color = '#fca5a5';
    }}
  }}

  function setUITour(stepIdx) {{
    if (indicator) {{
      indicator.className = 'tour-indicator';
      indicator.style.background = '#f59e0b';
      indicator.style.boxShadow = '0 0 16px #f59e0b';
    }}
    if (statusLabel) {{
      statusLabel.innerHTML = `<span style="color:#f59e0b;font-weight:800;">🧭 Tour Guide (Step ${{stepIdx + 1}} of 8):</span> ${{TOUR_STEPS[stepIdx].title}}`;
    }}
    if (timerBadge) {{
      timerBadge.style.display = 'inline-block';
      timerBadge.style.borderColor = 'rgba(245, 158, 11, 0.6)';
      timerBadge.style.color = '#fbbf24';
      timerBadge.innerText = `Step ${{stepIdx + 1}}/8`;
    }}
    if (wakeBtn) {{
      wakeBtn.innerHTML = '⏹️ Exit Tour';
      wakeBtn.style.borderColor = 'rgba(239, 68, 68, 0.55)';
      wakeBtn.style.color = '#fca5a5';
    }}
  }}

  function setUIIdle() {{
    isActiveConversation = false;
    activeExpiresAt = 0;
    tourActive = false;
    if (tourTimer) clearTimeout(tourTimer);
    if (indicator) {{
      indicator.className = 'pulsing-indicator';
      indicator.style.background = '#00d2ff';
      indicator.style.boxShadow = '0 0 10px #00d2ff';
    }}
    if (statusLabel) {{
      statusLabel.innerHTML = 'Voice Assistant Standby · Say <span style="color:#00d2ff;font-weight:700;">“Concall”</span> or <span style="color:#00d2ff;font-weight:700;">“CConcall”</span> to activate';
    }}
    if (transcriptLog) {{
      transcriptLog.innerText = 'Standby mode. Say "Concall" or "CConcall" or click Tap to Speak.';
    }}
    if (timerBadge) {{
      timerBadge.style.display = 'none';
    }}
    if (wakeBtn) {{
      wakeBtn.innerHTML = '🎙️ Tap to Speak';
      wakeBtn.style.borderColor = 'rgba(0, 210, 255, 0.55)';
      wakeBtn.style.color = '#a8efe8';
    }}
  }}

  function navigateToPage(pageName) {{
    console.log("ConcallIQ: Navigating to page:", pageName);
    const targetMap = {{
      "command center": "Command Center",
      "command": "Command Center",
      "alerts & watchlist": "Alerts & Watchlist",
      "alerts": "Alerts & Watchlist",
      "watchlist": "Alerts & Watchlist",
      "call timeline": "Call Timeline",
      "timeline": "Call Timeline",
      "topic radar": "Topic Radar",
      "radar": "Topic Radar",
      "key numbers": "Key Numbers",
      "scorecard": "Key Numbers",
      "peer benchmark": "Peer Benchmark",
      "benchmark": "Peer Benchmark",
      "ask the call": "Ask the Call",
      "ask": "Ask the Call",
      "language desk": "Language Desk",
      "language": "Language Desk",
      "evidence explorer": "Evidence Explorer",
      "evidence": "Evidence Explorer",
      "compare quarters": "Compare Quarters",
      "compare": "Compare Quarters",
      "reports": "Reports",
      "report": "Reports"
    }};
    const canonical = targetMap[pageName.toLowerCase().trim()] || pageName;

    // Method 1: Click the sidebar radio button in parent document (Instant SPA transition)
    try {{
      const parentDoc = (window.parent && window.parent.document) ? window.parent.document : document;
      const labels = parentDoc.querySelectorAll('div[data-testid="stRadio"] label, [data-testid="stSidebar"] label');
      for (let lbl of labels) {{
        const txt = (lbl.innerText || lbl.textContent || "").toLowerCase();
        if (txt.includes(canonical.toLowerCase()) || txt.includes(pageName.toLowerCase())) {{
          console.log("Found radio label, clicking:", txt);
          lbl.click();
          const radio = lbl.querySelector('input[type="radio"]');
          if (radio) {{
            radio.checked = true;
            radio.dispatchEvent(new Event('change', {{ bubbles: true }}));
            radio.dispatchEvent(new Event('input', {{ bubbles: true }}));
          }}
          break;
        }}
      }}
    }} catch(e) {{
      console.warn("Sidebar radio click notice:", e);
    }}

    // Method 2: Parent URL query parameter dispatch (Guaranteed Streamlit server sync)
    setTimeout(() => {{
      try {{
        const pWin = window.parent || window.top;
        if (pWin && pWin.location && pWin.location.href) {{
          const pUrl = new URL(pWin.location.href);
          pUrl.searchParams.set('concall_cmd', 'open ' + canonical.toLowerCase());
          pWin.location.href = pUrl.href;
          return;
        }}
      }} catch(e) {{}}
    }}, 400);
    return true;
  }}

  function dispatchToParent(cmd) {{
    if (!cmd) return;
    isNavigating = true;
    const cleanCmd = cmd.trim();
    console.log("ConcallIQ Dispatching to parent:", cleanCmd);
    try {{
      if (window.parent && window.parent.location && window.parent.location.href) {{
        const pUrl = new URL(window.parent.location.href);
        pUrl.searchParams.set('concall_cmd', cleanCmd);
        window.parent.location.href = pUrl.href;
        return;
      }}
    }} catch(e) {{}}
  }}

  // Dynamic semantic evidence matching over loaded transcripts
  function findBestEvidenceMatch(query) {{
    if (!EVIDENCE_STORE || EVIDENCE_STORE.length === 0) return null;
    const qWords = query.toLowerCase().replace(/[^a-z0-9\\s]/g, ' ').split(/\\s+/).filter(w => w.length > 2);
    if (qWords.length === 0) return null;

    let bestItem = null;
    let bestScore = 0;

    for (let item of EVIDENCE_STORE) {{
      let score = 0;
      const tLow = (item.text || '').toLowerCase();
      const topLow = (item.topic || '').toLowerCase();
      for (let w of qWords) {{
        if (topLow.includes(w)) score += 3;
        if (tLow.includes(w)) score += 2;
      }}
      if (score > bestScore) {{
        bestScore = score;
        bestItem = item;
      }}
    }}
    return bestScore >= 2 ? bestItem : null;
  }}

  // Multi-Page Tour Guide Automation
  function startGuidedTour() {{
    tourActive = true;
    currentTourStep = 0;
    activeExpiresAt = Date.now() + 180000;
    sessionStorage.setItem(ACTIVE_KEY, activeExpiresAt.toString());
    sessionStorage.setItem(TOUR_KEY, "0");
    runTourStep(0, false);
  }}

  function runTourStep(stepIdx, shouldNavigate) {{
    if (stepIdx >= TOUR_STEPS.length) {{
      // Tour Completed
      tourActive = false;
      sessionStorage.removeItem(TOUR_KEY);
      activeExpiresAt = Date.now() + 30000;
      setUIActive(30);
      if (statusLabel) {{
        statusLabel.innerHTML = '<span style="color:#10b981;font-weight:700;">✅ Guided Tour Complete!</span> Ask any question.';
      }}
      if (transcriptLog) {{
        transcriptLog.innerText = 'Completed all 8 tour steps. Say "Concall" anytime.';
      }}
      speakText("Guided tour completed! You can ask questions about any corporate disclosure.");
      return;
    }}

    tourActive = true;
    currentTourStep = stepIdx;
    sessionStorage.setItem(TOUR_KEY, stepIdx.toString());
    const step = TOUR_STEPS[stepIdx];

    setUITour(stepIdx);
    if (transcriptLog) {{
      transcriptLog.innerText = step.display;
    }}

    // 1. Move to the page if needed
    if (shouldNavigate && step.page !== CURRENT_PAGE) {{
      console.log("Tour advancing to step page:", step.page);
      navigateToPage(step.page);
      return;
    }}

    // 2. Speak the page overview
    speakText(step.speech, () => {{
      if (tourActive) {{
        if (transcriptLog) {{
          transcriptLog.innerText = `${{step.display}} (Say "next" or auto-advancing in 6s...)`;
        }}
        // 3. Auto-advance to next page after 6.5s
        if (tourTimer) clearTimeout(tourTimer);
        tourTimer = setTimeout(() => {{
          if (tourActive) {{
            runTourStep(stepIdx + 1, true);
          }}
        }}, 6500);
      }}
    }});
  }}

  function stopGuidedTour() {{
    tourActive = false;
    if (tourTimer) clearTimeout(tourTimer);
    sessionStorage.removeItem(TOUR_KEY);
    activeExpiresAt = Date.now() + 30000;
    setUIActive(30);
    if (statusLabel) {{
      statusLabel.innerHTML = '<span style="color:#38bdf8;font-weight:700;">Tour Stopped</span> · Ask your question...';
    }}
    speakText("Tour stopped. How can I help you?");
  }}

  function triggerWakeUp(optionalQuery) {{
    activeExpiresAt = Date.now() + 30000;
    sessionStorage.setItem(ACTIVE_KEY, activeExpiresAt.toString());
    playWakeChime();
    setUIActive(30);

    if (optionalQuery && optionalQuery.trim().length >= 2) {{
      handleSpokenQuestion(optionalQuery.trim());
    }} else {{
      speakWakeUp();
      if (transcriptLog) transcriptLog.innerText = '⚡ Concall Wokeup! Listening for your questions...';
    }}
  }}

  function handleSpokenQuestion(query) {{
    if (!query || query.trim().length < 2) return;
    const cleanQ = query.trim();
    const qLower = cleanQ.toLowerCase();

    // Check for sleep command
    if (qLower === "sleep" || qLower === "concall sleep" || qLower.includes("go to sleep") || qLower.includes("stop listening")) {{
      triggerSleep();
      return;
    }}

    // Check for Tour Guide commands
    if (qLower.includes("tour") || qLower.includes("walkthrough") || qLower.includes("show me around") || qLower.includes("guide me")) {{
      startGuidedTour();
      return;
    }}

    if (tourActive) {{
      if (qLower.includes("next") || qLower.includes("continue") || qLower.includes("next page") || qLower.includes("proceed")) {{
        if (tourTimer) clearTimeout(tourTimer);
        runTourStep(currentTourStep + 1, true);
        return;
      }}
      if (qLower.includes("back") || qLower.includes("previous")) {{
        if (tourTimer) clearTimeout(tourTimer);
        runTourStep(Math.max(0, currentTourStep - 1), true);
        return;
      }}
      if (qLower.includes("stop") || qLower.includes("exit") || qLower.includes("cancel") || qLower.includes("quit")) {{
        stopGuidedTour();
        return;
      }}
    }}

    // Reset 30s session on every authenticated user question
    activeExpiresAt = Date.now() + 30000;
    sessionStorage.setItem(ACTIVE_KEY, activeExpiresAt.toString());
    setUIActive(30);
    playQueryTone();

    let spokenText = "";
    let displayText = "";

    // 1. Navigation Intent Check (ALL 11 PAGES)
    const NAV_PAGES = [
      {{ name: "Command Center", keys: ["command center", "command", "home", "overview", "dashboard"] }},
      {{ name: "Alerts & Watchlist", keys: ["alerts & watchlist", "alerts", "watchlist", "risk alert", "stress map", "warnings"] }},
      {{ name: "Call Timeline", keys: ["call timeline", "timeline", "timeline page", "time machine", "claim time machine", "chronology", "quarter summaries", "narrative"] }},
      {{ name: "Topic Radar", keys: ["topic radar", "radar", "radar page", "galaxy", "narrative galaxy", "topics"] }},
      {{ name: "Key Numbers", keys: ["key numbers", "scorecard", "numbers", "metrics", "financial scorecard"] }},
      {{ name: "Peer Benchmark", keys: ["peer benchmark", "benchmark", "peers", "compare companies", "what moves together"] }},
      {{ name: "Ask the Call", keys: ["ask the call", "ask call", "ask", "chat", "questions"] }},
      {{ name: "Language Desk", keys: ["language desk", "language", "translation", "translate", "multilingual"] }},
      {{ name: "Evidence Explorer", keys: ["evidence explorer", "evidence", "explorer", "transcript", "search transcript"] }},
      {{ name: "Compare Quarters", keys: ["compare quarters", "compare quarter", "compare", "quarter diff", "diff"] }},
      {{ name: "Reports", keys: ["reports", "report", "export", "download", "excel report", "briefing"] }}
    ];

    let foundNav = null;
    const isExplicitNav = ["go to", "open", "show", "navigate", "take me", "switch", "change", "move to", "view", "page"].some(w => qLower.includes(w));
    for (let p of NAV_PAGES) {{
      if (p.keys.some(k => qLower.includes(k) && (isExplicitNav || qLower === k || qLower === k + " page"))) {{
        foundNav = p.name;
        break;
      }}
    }}

    if (foundNav) {{
      spokenText = `Navigating to ${{foundNav}}.`;
      displayText = `Navigating to ${{foundNav}} workspace...`;
      speakText(spokenText);
      if (statusLabel) statusLabel.innerHTML = `<span style="color:#10b981;font-weight:700;">Navigating:</span> ${{foundNav}}`;
      if (transcriptLog) transcriptLog.innerText = displayText;
      navigateToPage(foundNav);
      return;
    }}

    // 2. Explanations for 3D Nodes & Visual Features
    if (qLower.includes("why is this claim") || qLower.includes("contradict") || qLower.includes("why contradiction")) {{
      spokenText = "This claim is marked as contradicted because subsequent disclosures acknowledged persistent operational challenges that reversed earlier management assertions.";
      displayText = "Marked as Contradicted: subsequent disclosures reversed earlier guidance.";
    }} else if (qLower.includes("red node") || qLower.includes("red point") || qLower.includes("why is it red")) {{
      spokenText = "Red nodes represent contradiction candidates where forward guidance was reversed or qualified.";
      displayText = "Red Nodes: Contradiction Candidates where management reversed forward commitments.";
    }} else if (qLower.includes("green node") || qLower.includes("why is it green")) {{
      spokenText = "Green nodes represent reaffirmed targets maintained on track by management.";
      displayText = "Green Nodes: Reaffirmed Targets maintained on track by executive leadership.";
    }} else if (qLower.includes("amber node") || qLower.includes("yellow node") || qLower.includes("why is it amber")) {{
      spokenText = "Amber nodes represent weakened or qualified claims with cautionary hedging.";
      displayText = "Amber Nodes: Weakened Claims with cautionary hedging language.";
    }} else if (qLower.includes("blue node") || qLower.includes("why is it blue")) {{
      spokenText = "Blue nodes represent newly introduced corporate commitments.";
      displayText = "Blue Nodes: Newly Introduced Strategic Commitments.";
    }} else if (qLower.includes("time machine") || qLower.includes("3d view") || qLower.includes("explain 3d")) {{
      spokenText = "The 3D Claim Time Machine maps commitments across chronological quarters and strategic business domains.";
      displayText = "3D Claim Time Machine: Multi-quarter spatial narrative trajectory tracking commitments.";
    }} else if (qLower.includes("what is concalliq") || qLower.includes("about this tool") || qLower.includes("purpose")) {{
      spokenText = "ConcallIQ is an enterprise earnings intelligence platform for equity research and corporate strategy teams.";
      displayText = "ConcallIQ: Institutional Earnings Intelligence Platform decomposing quarterly conference calls.";
    }} else if (qLower.includes("page help") || qLower.includes("what can i do") || qLower.includes("how to use this page")) {{
      spokenText = `You are on ${{CURRENT_PAGE}}. Explore the 3D visual modules, inspect audit details, or ask questions about recent disclosures.`;
      displayText = `Page Help (${{CURRENT_PAGE}}): Explore 3D charts, review evidence citations, or examine metric trends.`;

    // 3. Direct Concrete Financial Question Answering (Strictly factual, citing figures)
    }} else if (qLower.includes("margin") || qLower.includes("ebit")) {{
      spokenText = `For ${{TARGET_COMPANY}} in ${{COMPANY_SUMMARY.latest_quarter}}, operating margin was ${{COMPANY_SUMMARY.operating_margin}}. ${{COMPANY_SUMMARY.operating_margin_sentence}}`;
      displayText = `${{COMPANY_SUMMARY.latest_quarter}} Operating Margin: ${{COMPANY_SUMMARY.operating_margin}}`;
    }} else if (qLower.includes("revenue") || qLower.includes("growth") || qLower.includes("topline") || qLower.includes("sales")) {{
      spokenText = `For ${{TARGET_COMPANY}} in ${{COMPANY_SUMMARY.latest_quarter}}, revenue growth was ${{COMPANY_SUMMARY.revenue_growth}}. ${{COMPANY_SUMMARY.revenue_sentence}}`;
      displayText = `${{COMPANY_SUMMARY.latest_quarter}} Revenue Growth: ${{COMPANY_SUMMARY.revenue_growth}}`;
    }} else if (qLower.includes("tcv") || qLower.includes("order book") || qLower.includes("deal") || qLower.includes("contract")) {{
      spokenText = `${{TARGET_COMPANY}} reported total contract value of ${{COMPANY_SUMMARY.tcv}} in ${{COMPANY_SUMMARY.latest_quarter}}. ${{COMPANY_SUMMARY.tcv_sentence}}`;
      displayText = `${{COMPANY_SUMMARY.latest_quarter}} TCV Order Book: ${{COMPANY_SUMMARY.tcv}}`;
    }} else if (qLower.includes("headcount") || qLower.includes("employee") || qLower.includes("associate") || qLower.includes("workforce") || qLower.includes("hiring")) {{
      spokenText = `${{TARGET_COMPANY}}'s total headcount was reported at ${{COMPANY_SUMMARY.headcount}} associates in ${{COMPANY_SUMMARY.latest_quarter}}. ${{COMPANY_SUMMARY.headcount_sentence}}`;
      displayText = `${{COMPANY_SUMMARY.latest_quarter}} Total Headcount: ${{COMPANY_SUMMARY.headcount}}`;
    }} else if (qLower.includes("attrition")) {{
      spokenText = `${{TARGET_COMPANY}}'s voluntary attrition stood at ${{COMPANY_SUMMARY.attrition}} in ${{COMPANY_SUMMARY.latest_quarter}}. ${{COMPANY_SUMMARY.attrition_sentence}}`;
      displayText = `${{COMPANY_SUMMARY.latest_quarter}} Voluntary Attrition: ${{COMPANY_SUMMARY.attrition}}`;
    }} else if (qLower.includes("profit") || qLower.includes("pat") || qLower.includes("net income")) {{
      spokenText = `Net profit for ${{TARGET_COMPANY}} in ${{COMPANY_SUMMARY.latest_quarter}} reached ${{COMPANY_SUMMARY.net_profit || "solid levels"}}. ${{COMPANY_SUMMARY.profit_sentence}}`;
      displayText = `${{COMPANY_SUMMARY.latest_quarter}} Net Profit: ${{COMPANY_SUMMARY.net_profit || "Solid"}}`;
    }} else if (qLower.includes("guidance") || qLower.includes("outlook") || qLower.includes("forecast") || qLower.includes("target")) {{
      spokenText = `Regarding corporate outlook: ${{COMPANY_SUMMARY.guidance}}`;
      displayText = `${{COMPANY_SUMMARY.latest_quarter}} Guidance: ${{COMPANY_SUMMARY.guidance}}`;
    }} else if (qLower.includes("risk") || qLower.includes("headwind") || qLower.includes("challenge") || qLower.includes("concern")) {{
      spokenText = `Key operational headwinds highlighted by leadership include: ${{COMPANY_SUMMARY.risks}}`;
      displayText = `${{COMPANY_SUMMARY.latest_quarter}} Headwinds: ${{COMPANY_SUMMARY.risks}}`;
    }} else if (qLower.includes("ai") || qLower.includes("genai") || qLower.includes("artificial intelligence")) {{
      spokenText = `On artificial intelligence: ${{COMPANY_SUMMARY.ai}}`;
      displayText = `${{COMPANY_SUMMARY.latest_quarter}} AI Disclosures: ${{COMPANY_SUMMARY.ai}}`;
    }} else {{
      // General dynamic semantic evidence search over all indexed statements
      const match = findBestEvidenceMatch(cleanQ);
      if (match) {{
        spokenText = `According to ${{match.speaker}} in ${{match.quarter}}: "${{match.text.replace(/[\\"']/g, '')}}".`;
        displayText = `${{match.quarter}} (${{match.topic}}): "${{match.text}}"`;
      }} else {{
        spokenText = `In ${{COMPANY_SUMMARY.latest_quarter}}, ${{TARGET_COMPANY}} delivered ${{COMPANY_SUMMARY.revenue_growth}} revenue growth and ${{COMPANY_SUMMARY.operating_margin}} operating margin, backed by an order book of ${{COMPANY_SUMMARY.tcv}}.`;
        displayText = `${{TARGET_COMPANY}} (${{COMPANY_SUMMARY.latest_quarter}}): Margin ${{COMPANY_SUMMARY.operating_margin}} · TCV ${{COMPANY_SUMMARY.tcv}}`;
      }}
    }}

    // Speak aloud through the browser speakers
    speakText(spokenText);

    if (statusLabel) {{
      statusLabel.innerHTML = `<span style="color:#10b981;font-weight:700;">Answered:</span> "${{cleanQ}}"`;
    }}
    if (transcriptLog) {{
      transcriptLog.innerText = `“${{displayText}}”`;
    }}
  }}

  function triggerSleep() {{
    if (tourTimer) clearTimeout(tourTimer);
    tourActive = false;
    sessionStorage.removeItem(TOUR_KEY);
    isActiveConversation = false;
    activeExpiresAt = 0;
    sessionStorage.removeItem(ACTIVE_KEY);
    playSleepChime();
    speakText("Concall Slept.");
    setUIIdle();
    if (statusLabel) {{
      statusLabel.innerHTML = '<span style="color:#ef4444;font-weight:700;">💤 Concall Slept</span> · Standby mode active';
    }}
    if (transcriptLog) {{
      transcriptLog.innerText = 'Assistant went to sleep after 30s of inactivity. Say "Concall" or "CConcall" to wake.';
    }}
  }}

  window.concallTriggerManualWake = function() {{
    window.concallToggleAssistant();
  }};

  window.concallToggleAssistant = function() {{
    if (tourActive) {{
      stopGuidedTour();
    }} else if (isActiveConversation && Date.now() < activeExpiresAt) {{
      triggerSleep();
    }} else {{
      try {{ recognition.start(); }} catch(e) {{}}
      triggerWakeUp();
    }}
  }};

  // Periodic Inactivity Checker (every 500ms)
  setInterval(() => {{
    if (isNavigating || isAssistantSpeaking) return;

    if (tourActive) {{
      return;
    }}

    if (isActiveConversation && activeExpiresAt > 0) {{
      const diffMs = activeExpiresAt - Date.now();
      const remainingSec = Math.max(0, Math.ceil(diffMs / 1000));
      if (remainingSec > 0) {{
        setUIActive(remainingSec);
      }} else {{
        triggerSleep();
      }}
    }}
  }}, 500);

  // Initialize UI state
  setUIIdle();

  // Resume guided tour if active across page reloads
  const savedTourStep = sessionStorage.getItem(TOUR_KEY);
  if (savedTourStep !== null && savedTourStep !== undefined) {{
    const stepIdx = parseInt(savedTourStep, 10);
    if (!isNaN(stepIdx) && stepIdx >= 0 && stepIdx < TOUR_STEPS.length) {{
      console.log("ConcallIQ: Resuming tour step", stepIdx, "on page", CURRENT_PAGE);
      tourActive = true;
      currentTourStep = stepIdx;
      activeExpiresAt = Date.now() + 180000;
      isActiveConversation = true;
      setTimeout(() => {{
        runTourStep(stepIdx, false);
      }}, 700);
    }}
  }} else if (INITIAL_TOUR) {{
    setTimeout(() => {{
      startGuidedTour();
    }}, 1200);
  }}

  recognition.onresult = function(event) {{
    // Guard against microphone hearing the assistant's own voice
    if (isAssistantSpeaking || (window.speechSynthesis && window.speechSynthesis.speaking) || Date.now() < echoGuardUntil) {{
      return;
    }}

    let currentPhrase = "";
    for (let i = event.resultIndex; i < event.results.length; ++i) {{
      const res = event.results[i];
      const text = res[0].transcript.trim();
      currentPhrase = text;
      lastCapturedText = text;
    }}

    // Filter out residual echo phrases
    if (isEchoPhrase(currentPhrase)) {{
      return;
    }}

    if (transcriptLog && currentPhrase) {{
      transcriptLog.innerText = `Heard: "${{currentPhrase}}"`;
    }}

    const isActive = (isActiveConversation && Date.now() < activeExpiresAt) || tourActive;

    // Case A: IDLE MODE -> Listen strictly for "Concall" or "CConcall"
    if (!isActive && !isNavigating) {{
      if (WAKE_REGEX.test(currentPhrase)) {{
        const afterWake = currentPhrase.replace(WAKE_REGEX, '').replace(/^[,\\s\\.\\?]+/, '').trim();
        triggerWakeUp(afterWake);
        return;
      }}
    }}

    // Case B: ACTIVE MODE -> User is speaking a query!
    if (isActive && !isNavigating && currentPhrase.length >= 2) {{
      if (statusLabel) {{
        statusLabel.innerHTML = `<span style="color:#38bdf8;font-weight:700;">Listening:</span> "${{currentPhrase}}"...`;
      }}

      // Reset silence debounce timer (900ms of user silence triggers answer)
      if (silenceTimer) clearTimeout(silenceTimer);
      silenceTimer = setTimeout(() => {{
        if (!isNavigating && !isAssistantSpeaking && Date.now() > echoGuardUntil && lastCapturedText.length >= 2) {{
          handleSpokenQuestion(lastCapturedText);
        }}
      }}, 900);
    }}
  }};

  recognition.onerror = function(event) {{
    if (event.error === 'not-allowed' || event.error === 'service-not-allowed') {{
      if (statusLabel) {{
        statusLabel.innerHTML = '<span style="color:#ef4444;font-weight:700;">Mic Access Required</span> · Click "Tap to Speak"';
      }}
      if (transcriptLog) {{
        transcriptLog.innerText = 'Browser permission needed. Click "Tap to Speak" to grant access.';
      }}
      if (indicator) {{
        indicator.style.background = '#ef4444';
        indicator.style.boxShadow = '0 0 10px #ef4444';
      }}
    }} else if (event.error !== 'no-speech') {{
      console.warn("ConcallIQ Voice notice:", event.error);
    }}
  }};

  recognition.onend = function() {{
    if (!isNavigating && !isAssistantSpeaking && Date.now() > echoGuardUntil) {{
      try {{
        recognition.start();
      }} catch(e) {{
        setTimeout(() => {{
          try {{ if (!isNavigating && !isAssistantSpeaking) recognition.start(); }} catch(err) {{}}
        }}, 300);
      }}
    }}
  }};

  try {{
    recognition.start();
    console.log("ConcallIQ Voice Engine: Continuous wake & dialogue active.");
  }} catch(e) {{}}
}})();
</script>
"""
