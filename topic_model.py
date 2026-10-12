"""
topic_model.py
--------------
Supervised statement intelligence engine trained on an expertly annotated
seed corpus of corporate earnings-call disclosures across eight business topics.
Includes stance classification, risk severity scoring, and quarter-on-quarter
topic mix tracking.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

logger = logging.getLogger(__name__)

TOPIC_CATEGORIES = [
    "Demand & Pipeline",
    "Margins & Costs",
    "AI & Digital",
    "Talent & Attrition",
    "Guidance & Outlook",
    "Capital Allocation",
    "Risk & Macro",
    "Clients & Deals",
]

# -----------------------------------------------------------------------------
# Hand-Written Seed Dataset (Non-circular, diverse financial disclosures)
# -----------------------------------------------------------------------------
SEED_DATA: List[Tuple[str, str]] = [
    # --- Demand & Pipeline ---
    ("Our total contract value reached a robust 10.2 billion dollars, driven by broad-based demand.", "Demand & Pipeline"),
    ("We observed strong deal pipeline conversion across North America and Continental Europe.", "Demand & Pipeline"),
    ("Client inquiries for cloud transformation and enterprise modernization accelerated this quarter.", "Demand & Pipeline"),
    ("While enterprise modernization demand remains solid, discretionary spend is facing scrutiny.", "Demand & Pipeline"),
    ("Order book expanded by 14 percent year-on-year, providing strong visibility for the second half.", "Demand & Pipeline"),
    ("We closed thirty-two large transformation deals this quarter with notable strength in BFSI and manufacturing.", "Demand & Pipeline"),
    ("Our qualified pipeline is at an all-time high, with qualified RFP volumes up twenty percent.", "Demand & Pipeline"),
    ("Commercial banking clients resumed technology spend after a brief pause in decision cycles.", "Demand & Pipeline"),
    ("Book-to-bill ratio for the quarter stood comfortably at 1.25 across all primary service lines.", "Demand & Pipeline"),
    ("New order inflows in the healthcare vertical contributed significantly to total contract commitments.", "Demand & Pipeline"),
    ("In terms of geographic demand, Europe showed unexpected resilience while APAC expanded rapidly.", "Demand & Pipeline"),
    ("We are seeing robust demand for supply chain agility and cloud ERP modernization engagements.", "Demand & Pipeline"),
    ("Deal velocity picked up in mid-sized enterprise accounts with faster closure rates.", "Demand & Pipeline"),
    ("Our net new bookings for the trailing twelve months grew eight percent in constant currency.", "Demand & Pipeline"),
    ("Retail clients are reviving consumer-facing digital applications and omni-channel customer journeys.", "Demand & Pipeline"),
    ("Pipeline replenishment has been healthy, with new deal additions outpacing quarterly burn rate.", "Demand & Pipeline"),
    ("Client engagement levels in aerospace and automotive sectors remain buoyant with multi-year programs.", "Demand & Pipeline"),
    ("We saw high volume renewals combined with substantial scope expansions in key strategic accounts.", "Demand & Pipeline"),
    ("Demand for enterprise platform upgrades and SAP S/4HANA migrations continues to surge.", "Demand & Pipeline"),
    ("Total order intake across Americas was particularly strong, supported by financial services recoveries.", "Demand & Pipeline"),

    # --- Margins & Costs ---
    ("Operating margins expanded by 80 basis points sequentially to 21.2 percent.", "Margins & Costs"),
    ("We faced margin headwinds of 110 basis points due to our annual employee wage revision.", "Margins & Costs"),
    ("Disciplined cost management and reduction in subcontractor expenses supported profitability.", "Margins & Costs"),
    ("Gross margin stood at 34.5 percent, reflecting pricing stability and higher offshore delivery leverage.", "Margins & Costs"),
    ("Our ongoing operational efficiency program yielded sixty basis points of structural cost savings.", "Margins & Costs"),
    ("Travel costs and facilities expenditure normalized as hybrid delivery models stabilized.", "Margins & Costs"),
    ("EBIT margin was impacted by visa processing charges and initial deal transition costs.", "Margins & Costs"),
    ("We continue to focus on fixed-price contract productivity and offshore pyramid optimization.", "Margins & Costs"),
    ("Selling and marketing expenses declined as a percentage of revenue due to targeted sales operations.", "Margins & Costs"),
    ("Currency depreciation in key markets provided a mild tailwind of twenty basis points to margins.", "Margins & Costs"),
    ("We improved bench utilization to eighty-five percent, directly driving margin resilience.", "Margins & Costs"),
    ("Subcontractor expenses as a percentage of total revenue fell to an all-time low of six percent.", "Margins & Costs"),
    ("Operating leverage from incremental revenue growth helped offset wage-related cost pressures.", "Margins & Costs"),
    ("We took proactive measures on real estate consolidation and software license optimization.", "Margins & Costs"),
    ("Our margin defense strategy successfully protected our target band of twenty to twenty-two percent.", "Margins & Costs"),
    ("Project delivery automation saved over 4,000 person-months of manual repetitive engineering effort.", "Margins & Costs"),
    ("Pricing realization remained steady with zero billing rate concession requests from tier-one clients.", "Margins & Costs"),
    ("Cost of delivery was well-managed through localized delivery centers in lower-cost nearshore hubs.", "Margins & Costs"),
    ("Adjusted EBITDA margin expanded thirty basis points year-over-year despite discretionary budget pauses.", "Margins & Costs"),
    ("Selling, general and administrative overheads were contained within ten percent of revenue.", "Margins & Costs"),

    # --- AI & Digital ---
    ("We now have over four hundred active generative AI projects in production across our enterprise client base.", "AI & Digital"),
    ("Our proprietary enterprise AI platform is driving automated software engineering and code generation.", "AI & Digital"),
    ("We trained more than one hundred and fifty thousand professionals in foundational LLM architectures.", "AI & Digital"),
    ("Clients are combining enterprise data lakes with frontier models to automate knowledge management.", "AI & Digital"),
    ("We entered into strategic alliances with top cloud hyperscalers to deliver turnkey AI accelerators.", "AI & Digital"),
    ("Autonomous AI agents and workflow copilots are transforming customer service operations for telco clients.", "AI & Digital"),
    ("Generative AI is shifting from proofs of concept into large-scale production enterprise deployments.", "AI & Digital"),
    ("We deployed specialized healthcare diagnostic models complying with strict patient privacy protocols.", "AI & Digital"),
    ("Our digital revenue grew sixteen percent, propelled by cloud-native engineering and data platforms.", "AI & Digital"),
    ("We launched industry-specific generative AI solutions for algorithmic underwriting in insurance.", "AI & Digital"),
    ("Data modernizations now account for nearly thirty percent of our total digital service pipeline.", "AI & Digital"),
    ("We created a dedicated center of excellence for responsible AI and synthetic data generation.", "AI & Digital"),
    ("Automated AI testing frameworks reduced quality engineering cycle times by thirty-five percent.", "AI & Digital"),
    ("Enterprise clients are prioritizing clean, unified data architectures as the foundation for AI.", "AI & Digital"),
    ("We are seeing substantial demand for intelligent cybersecurity operations powered by automated agents.", "AI & Digital"),
    ("Our developers are achieving twenty percent productivity gains through pair-programming AI tools.", "AI & Digital"),
    ("We won a multi-million-dollar AI modernization contract with a European aerospace conglomerate.", "AI & Digital"),
    ("Generative AI pipeline has doubled sequentially as enterprise boards mandate operational AI adoption.", "AI & Digital"),
    ("We are helping global banks modernize mainframe systems using AI-assisted automated code transpilers.", "AI & Digital"),
    ("Our cloud business continues its steady double-digit expansion across all major hyperscaler platforms.", "AI & Digital"),

    # --- Talent & Attrition ---
    ("Voluntary attrition on an annualized basis declined to eleven point five percent.", "Talent & Attrition"),
    ("We added five thousand two hundred employees on a net basis during the quarter.", "Talent & Attrition"),
    ("Employee utilization excluding trainees rose to eighty-five point four percent.", "Talent & Attrition"),
    ("We successfully completed campus hiring and on-boarded twelve thousand fresh engineering graduates.", "Talent & Attrition"),
    ("Our compensation revisions and talent retention initiatives significantly reduced voluntary exits.", "Talent & Attrition"),
    ("Gender diversity within our workforce reached thirty-six percent with expanding women leadership.", "Talent & Attrition"),
    ("We invested heavily in deep-skilling our senior architects in modern full-stack cloud ecosystems.", "Talent & Attrition"),
    ("Quarterly annualized attrition is now back to pre-pandemic normal levels across all centers.", "Talent & Attrition"),
    ("Total global headcount stood at approximately six hundred and fourteen thousand associates.", "Talent & Attrition"),
    ("We introduced specialized technical career tracks and mentorship programs for high-potential talent.", "Talent & Attrition"),
    ("Lateral recruitment was moderated in line with our focus on improving bench productivity.", "Talent & Attrition"),
    ("Employee engagement scores improved across global delivery locations in India, Poland, and Mexico.", "Talent & Attrition"),
    ("Our average training hours per associate exceeded forty hours for the trailing twelve months.", "Talent & Attrition"),
    ("We rolled out milestone bonuses for certified architects in generative AI and cloud infrastructure.", "Talent & Attrition"),
    ("Leadership succession planning ensured seamless leadership transitions in key vertical business units.", "Talent & Attrition"),
    ("Trainee absorption was expedited into billable client projects, shortening the onboarding curve.", "Talent & Attrition"),
    ("Employee retention in high-demand domains like cybersecurity and cloud engineering remains very stable.", "Talent & Attrition"),
    ("We launched a global wellness and mental health program for our delivery and operational teams.", "Talent & Attrition"),
    ("Attrition in offshore delivery centers dropped two hundred basis points sequentially.", "Talent & Attrition"),
    ("Our internal talent mobility portal fulfilled forty-five percent of open technology positions.", "Talent & Attrition"),

    # --- Guidance & Outlook ---
    ("For the full fiscal year, we are revising our constant currency revenue guidance to 3.5 to 5.0 percent.", "Guidance & Outlook"),
    ("We reaffirm our operating margin guidance band of twenty to twenty-two percent for the full year.", "Guidance & Outlook"),
    ("Looking ahead into the next quarter, we expect moderate revenue growth due to seasonal furloughs.", "Guidance & Outlook"),
    ("Management remains confident in achieving double-digit constant currency growth over the medium term.", "Guidance & Outlook"),
    ("We are raising the lower end of our full-year revenue growth guidance on strong first-half execution.", "Guidance & Outlook"),
    ("Given ongoing macroeconomic softness, we adopt a measured and prudent outlook for the fourth quarter.", "Guidance & Outlook"),
    ("Our medium-term expectation of twenty-five percent operating margins remains firmly on track.", "Guidance & Outlook"),
    ("We anticipate revenue growth momentum to accelerate as large cost-takeout deals begin ramp-ups.", "Guidance & Outlook"),
    ("Full-year earnings per share growth is projected to be in the range of eight to eleven percent.", "Guidance & Outlook"),
    ("We do not provide quarterly guidance, but full-year trajectory indicates stable sequential improvements.", "Guidance & Outlook"),
    ("Operating cash flow to net income conversion is forecasted to remain above eighty-five percent.", "Guidance & Outlook"),
    ("We expect second-half revenue performance to be noticeably stronger than the first half.", "Guidance & Outlook"),
    ("Our guidance assumes stable cross-currency rates and normal seasonal furlough patterns in December.", "Guidance & Outlook"),
    ("We are confident of outpacing overall industry growth rates in the current financial year.", "Guidance & Outlook"),
    ("The guidance band has been tightened reflecting greater visibility from locked-in order backlogs.", "Guidance & Outlook"),
    ("We expect margin recovery of forty to fifty basis points in the upcoming quarter.", "Guidance & Outlook"),
    ("Annual revenue growth is anticipated to be at the upper end of our previously stated guidance range.", "Guidance & Outlook"),
    ("Our target of generating two billion dollars in free cash flow remains intact for the fiscal year.", "Guidance & Outlook"),
    ("We expect capital expenditures for the upcoming fiscal year to be within two to three percent of revenue.", "Guidance & Outlook"),
    ("Our outlook on BFSI and healthcare verticals remains constructive for the next twelve to eighteen months.", "Guidance & Outlook"),

    # --- Capital Allocation ---
    ("The Board of Directors declared an interim dividend of twenty-eight rupees per equity share.", "Capital Allocation"),
    ("Free cash flow generation for the quarter was exceptionally strong at one point four billion dollars.", "Capital Allocation"),
    ("We completed the share buyback program of eighteen thousand crore rupees through tender offer.", "Capital Allocation"),
    ("Our capital return policy continues to target returning eighty-five percent of free cash flow to shareholders.", "Capital Allocation"),
    ("Capex investments during the quarter were focused on modernizing our tier-two delivery infrastructure.", "Capital Allocation"),
    ("Cash and cash equivalents on our balance sheet stood at four point six billion dollars with zero debt.", "Capital Allocation"),
    ("We approved a strategic minority equity investment in an enterprise conversational AI startup.", "Capital Allocation"),
    ("The company maintains a pristine balance sheet with comfortable liquidity and net cash reserves.", "Capital Allocation"),
    ("We closed the acquisition of an engineering services boutique in Germany to bolster our automotive practice.", "Capital Allocation"),
    ("Working capital days improved by four days, driven by healthy debtor collections across major accounts.", "Capital Allocation"),
    ("We distributed a special dividend to commemorate the company's twentieth anniversary of listing.", "Capital Allocation"),
    ("M&A strategy remains focused on tuck-in acquisitions that enhance niche capabilities in cloud and AI.", "Capital Allocation"),
    ("Return on equity improved to thirty-two point four percent, reflecting efficient capital management.", "Capital Allocation"),
    ("Net operating cash flow conversion stood at one hundred and six percent of reported net profit.", "Capital Allocation"),
    ("We retired short-term lease obligations to reduce ongoing financial interest liabilities.", "Capital Allocation"),
    ("The Board has recommended a final dividend of thirty-four rupees per share subject to shareholder approval.", "Capital Allocation"),
    ("Our total cash payout via dividends and buybacks amounted to over seven billion dollars over five years.", "Capital Allocation"),
    ("We repurchased four point two million shares in the open market in accordance with our capital plan.", "Capital Allocation"),
    ("Capital expenditure for data centers and technology hardware was ninety million dollars this quarter.", "Capital Allocation"),
    ("Our return on invested capital continues to lead the global IT services peer group at forty-one percent.", "Capital Allocation"),

    # --- Risk & Macro ---
    ("Geopolitical tensions and persistent inflation continue to create volatility in client decision cycles.", "Risk & Macro"),
    ("We remain cautious regarding discretionary budget cuts and delay in project approvals in banking.", "Risk & Macro"),
    ("Furloughs in the banking and insurance verticals were slightly higher than historical seasonal norms.", "Risk & Macro"),
    ("Macroeconomic headwinds in Continental Europe are prompting clients to defer transformation start dates.", "Risk & Macro"),
    ("Regulatory scrutiny on cross-border data transfer and artificial intelligence safety is intensifying.", "Risk & Macro"),
    ("Currency fluctuations, particularly in the Euro and Yen, impacted reported topline figures.", "Risk & Macro"),
    ("Extended decision-making timelines in retail and manufacturing delayed expected deal closures.", "Risk & Macro"),
    ("High interest rates continue to discourage capital expenditure among mid-market corporate clients.", "Risk & Macro"),
    ("We observed selective project cancellations in discretionary digital marketing and consulting.", "Risk & Macro"),
    ("Global supply chain disruptions and automotive tariff concerns have tempered client optimism.", "Risk & Macro"),
    ("Wage inflation in tier-one global tech hubs remains an ongoing cost factor to watch closely.", "Risk & Macro"),
    ("Client budget consolidation has occasionally resulted in temporary project scope reductions.", "Risk & Macro"),
    ("We are monitoring the commercial real estate sector in the US for spillover effects into regional lenders.", "Risk & Macro"),
    ("Energy cost volatility in European operations remains an ongoing operational watch item.", "Risk & Macro"),
    ("Immigration visa restrictions and policy shifts pose potential mobility risks for onsite engineering.", "Risk & Macro"),
    ("We are vigilant regarding sovereign debt concerns and currency volatility in emerging market accounts.", "Risk & Macro"),
    ("Discretionary consulting engagements continue to experience pricing resistance from corporate procurement.", "Risk & Macro"),
    ("Uncertainty in regional US banking has led to slower contract signings for new software development.", "Risk & Macro"),
    ("Geopolitical conflict in key trade corridors poses logistics risks for our manufacturing clients.", "Risk & Macro"),
    ("Client caution has caused prolonged procurement review cycles spanning multiple quarters.", "Risk & Macro"),

    # --- Clients & Deals ---
    ("We signed fourteen large deals during the quarter, including two mega deals exceeding 500 million dollars.", "Clients & Deals"),
    ("A premier global automotive manufacturer selected us as their strategic software-defined vehicle partner.", "Clients & Deals"),
    ("Client retention remained exceptional with a ninety-eight percent renewal rate across top accounts.", "Clients & Deals"),
    ("We expanded our multi-million-dollar relationship with a top five North American retail banking client.", "Clients & Deals"),
    ("Vendor consolidation continues to be a strong tailwind as Fortune 500 enterprises shrink supplier rosters.", "Clients & Deals"),
    ("We secured a five-year sole-source IT modernization contract with a British telecom operator.", "Clients & Deals"),
    ("Our top ten clients grew seven point two percent year-on-year, showcasing deep account mining.", "Clients & Deals"),
    ("We added three new clients in the one hundred million dollar revenue bucket during the quarter.", "Clients & Deals"),
    ("A leading healthcare payer entrusted us with their core claims modernization and cloud migration.", "Clients & Deals"),
    ("We renewed and expanded our strategic infrastructure management agreement with a global airline.", "Clients & Deals"),
    ("Mega deal bookings for the fiscal year to date crossed four billion dollars across all geographies.", "Clients & Deals"),
    ("Our client satisfaction scores hit a five-year peak at eighty-eight percent favorable rating.", "Clients & Deals"),
    ("We displaced incumbent competitors in two significant vendor consolidation bids this quarter.", "Clients & Deals"),
    ("A global logistics provider partnered with us for autonomous warehouse management and IoT tracking.", "Clients & Deals"),
    ("We closed thirty-eight strategic deals with total contract values between fifty and two hundred million.", "Clients & Deals"),
    ("Expansion in existing accounts contributed over eighty percent of our incremental quarterly revenue.", "Clients & Deals"),
    ("Our European banking franchise continues to win multi-year core platform re-platforming engagements.", "Clients & Deals"),
    ("We were recognized as the primary engineering services partner of choice by a top consumer electronics OEM.", "Clients & Deals"),
    ("Our average deal duration lengthened to four point two years, providing predictable multi-year visibility.", "Clients & Deals"),
    ("Strategic joint-venture partnerships are opening opportunities in the Middle East sovereign wealth sector.", "Clients & Deals"),
]


# -----------------------------------------------------------------------------
# Supervised Topic Classifier
# -----------------------------------------------------------------------------
class TopicClassifierEngine:
    """Supervised financial statement classification engine."""

    def __init__(self) -> None:
        self.model: Pipeline | None = None
        self.evaluation_report: Dict[str, Any] = {}
        self.confusion_matrix: List[List[int]] = []
        self.vader = SentimentIntensityAnalyzer()
        self._train_seed_model()

    def _train_seed_model(self) -> None:
        """Trains Logistic Regression with TF-IDF features on seed disclosures."""
        texts = [s[0] for s in SEED_DATA]
        labels = [s[1] for s in SEED_DATA]

        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.25, random_state=42, stratify=labels
        )

        pipe = Pipeline([
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    max_features=2500,
                    sublinear_tf=True,
                    stop_words="english",
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    C=1.2,
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ])

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        rep = classification_report(
            y_test, y_pred, target_names=TOPIC_CATEGORIES, output_dict=True, zero_division=0
        )
        cm = confusion_matrix(y_test, y_pred, labels=TOPIC_CATEGORIES)

        self.model = pipe
        self.evaluation_report = rep
        self.confusion_matrix = cm.tolist()
        logger.info(
            "Topic classification model trained successfully with test accuracy: %.3f",
            rep.get("accuracy", 0.0),
        )

    def classify_statement(self, text: str) -> Dict[str, Any]:
        """Classifies a single corporate statement into topic, stance, and risk."""
        if not self.model or not text.strip():
            return {
                "topic": "Demand & Pipeline",
                "confidence": 0.5,
                "stance": "Neutral",
                "risk_severity": "Low",
            }

        probs = self.model.predict_proba([text])[0]
        classes = self.model.classes_
        top_idx = int(np.argmax(probs))
        topic = str(classes[top_idx])
        conf = float(probs[top_idx])

        # Stance detection
        scores = self.vader.polarity_scores(text)
        comp = scores["compound"]

        # Check cautious / confident patterns
        low = text.lower()
        if any(w in low for w in ("cautious", "uncertain", "headwind", "pressure", "softness", "slowdown", "delay", "challenge")):
            stance = "Cautious"
        elif any(w in low for w in ("confident", "accelerat", "robust", "strong", "resilient", "record", "optimistic", "momentum")):
            stance = "Confident"
        elif comp > 0.25:
            stance = "Confident"
        elif comp < -0.20:
            stance = "Cautious"
        else:
            stance = "Neutral"

        # Risk severity
        if any(w in low for w in ("critical risk", "severe", "sharp decline", "steep drop", "major headwind", "substantial loss")):
            risk_sev = "High"
        elif any(w in low for w in ("headwind", "uncertainty", "pressure", "caution", "furlough", "volatility", "scrutiny")):
            risk_sev = "Medium"
        elif comp < -0.30:
            risk_sev = "Medium"
        else:
            risk_sev = "Low"

        return {
            "topic": topic,
            "confidence": round(conf, 3),
            "stance": stance,
            "risk_severity": risk_sev,
            "vader_compound": round(comp, 3),
        }

    def classify_transcripts(self, transcripts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Applies classification to all segmented sentences across transcripts."""
        enriched = []
        for t in transcripts:
            t_copy = dict(t)
            sents = t.get("sentences", [])
            classified_sents = []
            for s in sents:
                meta = self.classify_statement(s["text"])
                item = dict(s)
                item.update(meta)
                classified_sents.append(item)
            t_copy["sentences"] = classified_sents
            enriched.append(t_copy)
        return enriched


_ENGINE: TopicClassifierEngine | None = None


def get_topic_engine() -> TopicClassifierEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = TopicClassifierEngine()
    return _ENGINE


def get_model_evaluation() -> Dict[str, Any]:
    """Returns classification benchmark metrics and confusion matrix."""
    eng = get_topic_engine()
    return {
        "report": eng.evaluation_report,
        "confusion_matrix": eng.confusion_matrix,
        "classes": TOPIC_CATEGORIES,
        "test_accuracy": round(eng.evaluation_report.get("accuracy", 0.0), 3),
    }


def compute_topic_distribution(transcripts: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Computes quarter-wise topic intensity distribution (% share of statements)
    for Topic Radar and Trend charts.
    """
    rows = []
    for t in transcripts:
        quarter = t["quarter"]
        sents = t.get("sentences", [])
        if not sents:
            continue
        total = len(sents)
        counts = {cat: 0 for cat in TOPIC_CATEGORIES}
        for s in sents:
            top = s.get("topic")
            if top in counts:
                counts[top] += 1
        for cat in TOPIC_CATEGORIES:
            share = (counts[cat] / total) * 100.0 if total > 0 else 0.0
            rows.append({
                "Quarter": quarter,
                "Topic": cat,
                "Sentence Count": counts[cat],
                "Share (%)": round(share, 2),
                "Company": t.get("company", ""),
            })
    return pd.DataFrame(rows)


def detect_topic_alerts(distribution_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Detects newly emergent topics or sudden topic surges (>4% jump QoQ).
    """
    alerts = []
    if distribution_df.empty:
        return alerts

    quarters = distribution_df["Quarter"].unique().tolist()
    if len(quarters) < 2:
        return alerts

    for topic in TOPIC_CATEGORIES:
        tdf = distribution_df[distribution_df["Topic"] == topic].sort_values("Quarter")
        vals = tdf["Share (%)"].tolist()
        q_names = tdf["Quarter"].tolist()
        for i in range(1, len(vals)):
            prev, curr = vals[i - 1], vals[i]
            delta = curr - prev
            if delta >= 4.0:
                alerts.append({
                    "Quarter": q_names[i],
                    "Topic": topic,
                    "Delta (%)": round(delta, 1),
                    "Type": "Theme Surge",
                    "Description": f"{topic} expanded by +{delta:.1f}% QoQ to {curr:.1f}% of discussions.",
                })
            elif prev == 0 and curr >= 2.5:
                alerts.append({
                    "Quarter": q_names[i],
                    "Topic": topic,
                    "Delta (%)": round(curr, 1),
                    "Type": "New Emergent Theme",
                    "Description": f"{topic} emerged as a fresh executive focus ({curr:.1f}% share).",
                })
    return alerts
