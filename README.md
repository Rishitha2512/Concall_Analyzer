# ConcallIQ — Immersive 3D Financial Intelligence Platform

> **Institutional-Grade 3D Earnings-Call Intelligence Workstation & Global Voice System**  
> *Engineered for Equity Research Analysts, Investor Relations (IR) Executives, and Corporate Strategy Teams.*

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40%20--%201.63-FF4B4B.svg)](https://streamlit.io/)
[![PyTest](https://img.shields.io/badge/Tests-42%20Passed%20(100%25)-success.svg)](https://docs.pytest.org/)
[![License](https://img.shields.io/badge/License-Proprietary%20%2F%20Academic-informational.svg)]()

---

## Executive Summary & Product Vision

**ConcallIQ** transforms unstructured corporate earnings-call disclosures across major Indian and global IT giants into an evidence-grounded, spatial financial intelligence workstation.

Rather than presenting isolated charts or black-box summaries, ConcallIQ is organized around the core experience of **investigating a company's evolving narrative**:
1. **The 3D Claim Time Machine (Signature Centerpiece)**: Spatially maps and tracks management commitments across quarters. Distinguishes reaffirmed targets (green), weakened/hedged assertions (amber), newly introduced initiatives (blue), and potential contradiction candidates (red) requiring investigation.
2. **Corporate Narrative Galaxy**: A 3D spherical knowledge network linking companies, strategic business domains, technology platforms, and geographic markets.
3. **Narrative Stress Map**: A 3D priority landscape quantifying operational stress indices based on claim modifications, guidance variations, and risk frequency.
4. **3D Financial Peer Space**: Interactive 3-dimensional positioning of peers (Revenue Growth vs. Operating Margin vs. Financial Scale) with 2D tabular fallbacks.
5. **Global Voice Assistant with Wake-Word “Concall”**: Persistent across all pages. Users simply say **“Concall”** to navigate, ask data questions, or investigate flagged statements without clicking a microphone button.
6. **Zero-Jargon Institutional UI**: The interface uses strictly institutional financial terminology (*Language Desk*, *Key Numbers Scorecard*, *Topic Radar*, *What Moves Together*), with technical ML details reserved for evaluation documentation.
7. **Complete Data Sanitization**: Resolves duplicate filings, corrupted recording stubs, and chronologically mislabelled documents using cryptographic hashing and textual calendar induction.

---

## Quickstart Guide

### 1. System Requirements
- **OS**: Windows 10/11, Linux, or macOS (Optimized for Windows with `.bat` launcher).
- **Python**: Python 3.10 or 3.11.
- **Hardware**: 8 GB RAM minimum (16 GB recommended for local transformers).
- **Browser**: Google Chrome or Microsoft Edge (recommended for Web Speech API continuous wake-word listening and speech synthesis).

### 2. Instant Launch (Windows)
Double-click `run_concalliq.bat` or execute in PowerShell:
```powershell
.\run_concalliq.bat
```
*The batch runner automatically provisions the virtual environment, verifies dependencies, tests the runtime, and opens the application on `http://localhost:8501`.*

### 3. Manual Installation
```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

### 4. Running the Complete Verification Test Suite
```powershell
python -m pytest tests/ -v
```
*Current test suite status: **46 / 46 Passed (100%)**.*

---

## Global Voice Assistant — Wake-Word “Concall”

The global voice assistant is accessible from **every single page** of the platform:
- **Wake-Word Activation**: Say **“Concall”** into your microphone. The assistant immediately responds with:
  - **Instant Visual Flash**: `⚡ Concall Wokeup!` on the HUD status banner and a floating toast notification.
  - **Melodic Activation Chime**: High-tech two-tone Web Audio chime (`D5 -> A5`).
  - **Spoken Voice Acknowledgment**: Speaks **“Concall Wokeup. How can I help you explore this call?”** aloud via speech synthesis.
- **Hands-Free Navigation**:
  - *“Concall, open timeline.”* -> Responds: *“Concall Wokeup. Navigating to Call Timeline.”*
  - *“Concall, show peer comparison.”* -> Navigates to Peer Benchmark.
  - *“Concall, go to key numbers.”* -> Navigates to Key Numbers.
  - *“Concall, take me to evidence.”* -> Navigates to Evidence Explorer.
- **Interface Explanation**:
  - *“Concall, what can I do on this page?”*
  - *“Concall, what does this red node mean?”* (Explains contradiction candidates requiring investigation)
  - *“Concall, what does this green node mean?”* (Explains reaffirmed targets on track)
- **Investigation & Data Inquiries**:
  - *“Concall, which claims changed this quarter?”*
  - *“Concall, compare TCS and Infosys.”*
  - *“Concall, why was this claim flagged?”*
- **Accessibility & Browser Permissions**: An interactive listener HUD bar is docked at the top of every page showing real-time microphone status and a 1-click **🎙️ Wake Concall** button to grant browser permissions or wake without a microphone.
- **Mandatory Footer**: At the bottom of every page:  
  **To activate voice assistant, say “Concall”.**

---

## 3D Spatial Intelligence Capabilities

| 3D Visualization Module | Placement in Application | Analytical Value | Accessible 2D Fallback |
|---|---|---|---|
| **The 3D Claim Time Machine** | *Command Center* & *Call Timeline* | Traces how specific management commitments evolve across quarters. Nodes are color-coded by status (Reaffirmed, Weakened, Contradiction Candidate, New) with connecting lines indicating lineage. | Chronological 2D scatter timeline with status filters and click-to-evidence cards. |
| **Corporate Narrative Galaxy** | *Topic Radar* | 3D spherical knowledge network linking corporate focus areas to emerging technologies (GenAI, Cloud, S/4HANA) and geographic markets. | Multi-quarter Radar chart and per-topic statement breakdown table. |
| **Narrative Stress Map** | *Alerts & Watchlist* | 3D investigation priority landscape across quarters and topics (Z-axis = Narrative Stress Index). | Categorized alert cards with cited executive quotes and severity filter. |
| **3D Financial Peer Space** | *Peer Benchmark* | Spatial benchmarking comparing peers on Revenue Growth (%) vs. Operating Margin (%) vs. Financial Scale. | Ranked tabular comparison and sector affinity network graph. |
| **Audio & Narrative View** | *Evidence Explorer* | Acoustic discussion density tracking speaker turns, speech segment durations, and discussion pacing without subjective intent inference. | Transcript search with highlighted search phrases and speaker filters. |

---

## Academic Techniques to Institutional Feature Mapping

Per academic assessment requirements, all six foundational techniques power multiple concrete features while remaining insulated behind business terminology:

| # | Academic NLP / ML Technique | Institutional UI Module | Direct Capabilities Powered in ConcallIQ |
|---|---|---|---|
| **1** | **Machine Translation** | *Language Desk* & *Executive Briefings* | • Sentence-batched NLLB-200 translation with zero silent truncation.<br>• Covers 8 Indian languages (Hindi, Tamil, Telugu, Kannada, Malayalam, Bengali, Marathi, Gujarati) + 3 European languages.<br>• Side-by-side view with export and speech synthesis. |
| **2** | **Multilingual NLP** | *Document Ingestion* & *Voice Input* | • Custom transcript uploader (PDF/TXT) in any script.<br>• Automatic Unicode script heuristics and language detection.<br>• Quality note and confidence score badge per uploaded call.<br>• Indian-accented and vernacular speech recognition via local Whisper STT.<br>• English translation pivot for automated financial scoring. |
| **3** | **Text Classification** | *Topic Radar* & *Analysis Reliability* | • Supervised Logistic Regression trained on hand-curated non-circular disclosures.<br>• Classifies statements into 8 business domains.<br>• Tri-state Stance scoring (Confident / Cautious / Neutral) and Risk Severity rating (High / Medium / Low).<br>• Analysis Reliability subpanel showing Precision, Recall, and Confusion Matrix. |
| **4** | **Named Entity Recognition (NER)** | *Key Numbers Scorecard* & *Entity Explorer* | • High-precision extraction of money, percentages, dates, periods, people, organizations, technologies, and geographies.<br>• Automated 8-metric Corporate Scorecard.<br>• QoQ delta tracking and click-to-cite drill-down into source sentences.<br>• Speaker-wise segmentation (Management vs. Analyst) and section breakdown (Prepared Remarks vs. Q&A). |
| **5** | **Retrieval-Augmented QA & Chatbot** | *Ask the Call* & *Voice Assistant ("Concie")* | • Multi-vector lexical-semantic retrieval QA engine over 40+ call transcripts.<br>• Verified citations with company, quarter, speaker, and sentence index.<br>• Multi-company comparative query parsing.<br>• Integrated voice tour (8 steps) and persistent wake-word ("Concall"). |
| **6** | **Market Basket Analysis (MBA)** | *What Moves Together* & *Peer Benchmark* | • Cross-company transaction mining over statement-level and call-level disclosures.<br>• Support, Confidence, and Lift metrics across topics, tech stacks, and geographic markets.<br>• Plain-language strategic rules.<br>• Interactive network relationship graph and predictive recommendations. |

---

## Complete 11-Page Demonstration Walkthrough

Follow this workflow for a comprehensive live viva demonstration:

1. **Command Center**:
   - Greeted by the unified cyber-navy workstation matching institutional equity research terminals.
   - Say **“Concall”** anywhere: hears ascending chime, speaks **“Concall Wokeup”**, and stays active for 30 seconds.
   - Ask any question while active (automatically resets the 30-second timer); if silent for 30 seconds, plays descending chime and announces **“Concall Slept”**.
   - Inspect top KPI pills (`Overall Sentiment`, `Key Risks`, `Claim Changes`), the central **3D Claim Evolution Timeline**, the **Selected Claim Details** card with classification rationale, and the **Ask ConcallIQ** conversation box.
   - Explore middle-row 3D charts: **Management Tone & Sentiment (3D)** surface and **Key Business Topics (3D)** orbital network, alongside the **Top Risks** progress breakdown.
   - Review bottom **Recent Key Insights** 5-card metric strip.
   - Expand **Extended Financial Analysis & Sentiment Diagnostics** for 2D diagnostic fallbacks.
2. **Alerts & Watchlist**:
   - Inspect the **3D Narrative Stress Map** to identify where contradiction candidates and weakened statements concentrate.
   - Adjust the **Alert Sensitivity Threshold** slider to filter minor tone fluctuations.
   - Review the **Modified Management Claims** audit cards.
3. **Call Timeline**:
   - Use the **Claim Time Machine** to trace commitments across quarters.
   - Use the **Side-by-Side Statement Diff** dropdown to compare the original baseline statement against the subsequent qualified statement.
4. **Topic Radar**:
   - Explore the **3D Corporate Narrative Galaxy** knowledge network.
   - Compare multi-quarter priority shifts on the Radar chart.
   - Expand the **Analysis Reliability** panel to review precision, recall, and confusion matrix benchmarks.
5. **Key Numbers**:
   - Review the 8-metric Corporate Scorecard table with QoQ deltas.
   - Select a metric (e.g., *Operating Margin*) to view the historical bar trend.
   - Click the citation inspector to reveal the exact supporting executive sentence.
6. **Peer Benchmark**:
   - Select 2 to 5 peers (e.g., TCS, Infosys, Cognizant).
   - Explore the **3D Financial Peer Space** plotting Revenue Growth vs. Operating Margin vs. Scale.
   - Review the **What Moves Together** sector linkage network and plain-English findings.
7. **Ask the Call**:
   - Ask a question: *"What did management say about margin headwinds in Q1 FY26?"*
   - Review the synthesized response accompanied by verified citation cards.
   - Test voice input and speech synthesis playback.
8. **Language Desk**:
   - Translate an executive summary into Hindi, Tamil, Telugu, or German.
   - Review side-by-side output and click **Read Translation Aloud**.
   - Switch to **Ingest Custom Transcript** to upload an external document in any script.
9. **Evidence Explorer**:
   - Search keywords (e.g., *"margin expansion"* or *"cloud"*) with automatic highlight badges.
   - Open the **Audio & Narrative Exploration View** to view acoustic turn durations and listen to call recordings.
10. **Compare Quarters**:
    - Select two quarters (e.g., Infosys Q4 FY25 vs. Q1 FY26) to view net tone variance, summaries, and commitment deltas.
11. **Reports**:
    - Download the 5-sheet institutional Excel workbook, CSV monitor, JSON payload, or styled 1-page HTML analyst briefing.

---

## Codebase Architecture

```
ideas2026/
├── app.py                     # Main application router with global wake-word listener
├── claim_tracker.py           # Claim extraction, cross-quarter matching & stress indicators
├── visualization_3d.py        # 3D Claim Time Machine, Narrative Galaxy, Stress Map & Peer Space
├── data_loader.py             # PDF extraction, period detection, SHA-256 deduplication, stub filter
├── topic_model.py             # Supervised Logistic Regression classifier & reliability evaluator
├── entity_extractor.py        # Financial entity extraction, Scorecard generator, QoQ deltas
├── qa_engine.py               # Lexical-semantic vector retrieval QA engine with citations
├── peer_analysis.py           # Cross-company thematic co-occurrence engine & network graphs
├── translator.py              # Sentence-batched NLLB-200 neural machine translation
├── multilingual.py            # Unicode script heuristics, language detector, ingestion handler
├── voice_assistant.py         # Global wake-word engine ("Concall"), Whisper STT & browser TTS
├── reports.py                 # Multi-sheet Excel (openpyxl) & styled HTML executive briefing
├── ui_components.py           # Dark Aurora styling, Plotly charts with insights, global footer
├── run_concalliq.bat          # One-click Windows batch launcher
├── requirements.txt           # Python dependencies
├── README.md                  # Comprehensive platform overview and demo script
├── PROJECT_DOCUMENTATION.md   # IIT-grade architectural report & 10 viva Q&As
├── cache/transcripts/         # Pre-warmed disk cache for instant startup
├── pdfs/                      # Complete earnings-call PDF corpus across 10 IT giants
├── audip/                     # Audio call recordings for acoustic narrative exploration
└── tests/                     # 46-test automated verification suite
    ├── test_claim_tracker.py
    ├── test_visualization_3d.py
    ├── test_period_detection.py
    ├── test_topic_model.py
    ├── test_entity_extractor.py
    ├── test_peer_analysis.py
    ├── test_qa_engine.py
    ├── test_smoke_pages.py
    ├── test_ui_audit.py
    └── test_voice_assistant.py
```

---

## Research Disclaimer

> **IMPORTANT NOTICE**: ConcallIQ is an academic and financial research decision-support workstation developed for educational and evaluation purposes. The automated metrics, 3D visualizations, classifications, risk indicators, and synthesized responses generated by this software do NOT constitute financial, investment, legal, or tax advice. Past corporate disclosures do not guarantee future market results.
