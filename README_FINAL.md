# ConcallIQ — Earnings Intelligence + Voice Assistant

Turns quarterly earnings-call transcripts into an interactive dashboard (tone, risks, commitments,
themes) built from six NLP concepts, and adds a **speech-processing voice assistant ("Concie")**
that guides new users and answers questions — by voice or text.

## Run
```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python -m streamlit run app.py          # or double-click run_concalliq.bat
```
Use **Chrome or Edge** (voice output uses the browser's speech engine). Allow the microphone when asked.

## The six NLP lab concepts  (Advanced Insights page)
| Tab | NLP concept | Technique |
|---|---|---|
| Translate | Machine translation | Meta NLLB-200 transformer (local) |
| Language coverage | Multilingual text processing | Unicode script analysis |
| Statement themes | Text classification | TF-IDF + Logistic Regression, confidence scores |
| Financial references | Named entity recognition / information extraction | Regex patterns: %, money, fiscal periods, dates, organisations |
| Ask the call | Question answering / chatbot | TF-IDF cosine-similarity retrieval |
| Theme relationships | Market-basket / association rules | Support, confidence, lift over management statements |

Supporting analysis: VADER sentiment (tone), TF-IDF extractive summary, key-phrase extraction.

## Speech processing (new) — Voice Assistant page
| Stage | Technology |
|---|---|
| Speech-to-text | OpenAI **Whisper** (`openai/whisper-base.en`, local via transformers; ~290 MB first-time download) |
| Understanding | TF-IDF intent matching over a built-in help knowledge base + data handlers + transcript retrieval |
| Text-to-speech | Browser `speechSynthesis` (no install; speed / stop / replay controls in the sidebar) |

What it does
- **Guides beginners**: "How do I start?", "Start the tour" (narrates 8 steps), "What can I do on this page?", "How do I translate?", mic troubleshooting.
- **Answers questions about the calls**: tone, tone trend, risks, commitments, themes, quarter summaries, or any free question (answered with transcript evidence).
- **Voice navigation**: "Open reports", "Go to timeline".
- Optional: `set CONCALLIQ_WHISPER_MODEL=openai/whisper-tiny.en` for a faster, smaller model.

## Changes in this version
- Complete visual redesign: aurora theme, KPI cards, tone gauge, theme heatmap + radar, timeline view, network graph, donut charts, highlighted evidence search.
- Company picker (tiles + dropdown) instead of typing; works from any working directory.
- Fixes: tone was +1.00 for every quarter (now sentence-level average); percentages were never extracted; page markers / participant lists / legal disclaimers no longer pollute summaries, risks and commitments; "ai" is no longer counted inside words like "said"; market basket now mines statement-level baskets.

## Data notes
- `pdfs/TCS/Q4.pdf` and `Q5.pdf` are byte-identical files.
- `pdfs/Infosys/Q2.pdf` is the Q4 FY25 call (April 2025), i.e. earlier than `Q1.pdf` (Q1 FY26); quarter labels come from filenames.

ConcallIQ is a research tool; it does not provide investment advice.
