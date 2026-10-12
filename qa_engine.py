"""
qa_engine.py
------------
Evidence-grounded conversational search and comparative question-answering
engine across corporate earnings calls.
Provides cited responses (Company, Quarter, Speaker, Section), handles
follow-up queries and explicit filters, supports peer comparisons, and
supplies dynamic questions with feedback tracking.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from data_loader import list_available_companies, load_company_data

logger = logging.getLogger(__name__)

_AI_MODEL = None
_AI_TOKENIZER = None
_AI_INITIALIZED = False


def get_ai_generator():
    """Lazily loads local google/flan-t5-small model from cache without downloading."""
    global _AI_MODEL, _AI_TOKENIZER, _AI_INITIALIZED
    if _AI_INITIALIZED:
        return _AI_MODEL, _AI_TOKENIZER
    _AI_INITIALIZED = True
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        model_name = "google/flan-t5-small"
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, local_files_only=True)
        _AI_TOKENIZER = tokenizer
        _AI_MODEL = model
        logger.info("Local AI model (google/flan-t5-small) loaded successfully.")
    except Exception as exc:
        logger.warning("Could not load local flan-t5 model: %s", exc)
        _AI_MODEL = None
        _AI_TOKENIZER = None
    return _AI_MODEL, _AI_TOKENIZER


QA_STOP_WORDS = [
    "what", "did", "management", "say", "about", "tell", "me", "how", "is", "the",
    "are", "there", "in", "for", "of", "and", "a", "an", "on", "to", "with", "at",
    "by", "from", "up", "about", "into", "over", "after", "company", "call", "calls",
    "quarter", "quarterly", "please", "can", "you", "give", "details",
]


class RetrievalQAEngine:
    """Enterprise retrieval question answering engine with AI generation."""

    def __init__(self) -> None:
        self.corpus_sentences: List[Dict[str, Any]] = []
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None
        self._is_indexed = False

    def build_index(self, company_transcripts: List[Dict[str, Any]]) -> None:
        """Indexes sentences from loaded company transcripts."""
        all_sents = []
        for t in company_transcripts:
            comp = t.get("company", "")
            q = t.get("quarter", "")
            for s in t.get("sentences", []):
                item = dict(s)
                item["company"] = comp
                item["quarter"] = q
                all_sents.append(item)

        self.corpus_sentences = all_sents
        if all_sents:
            texts = [s["text"] for s in all_sents]
            self.vectorizer = TfidfVectorizer(
                stop_words=QA_STOP_WORDS,
                ngram_range=(1, 2),
                sublinear_tf=True,
                max_features=8000,
            )
            self.tfidf_matrix = self.vectorizer.fit_transform(texts)
            self._is_indexed = True

    def _parse_query_filters(self, query: str) -> Dict[str, Any]:
        """Detects explicit quarter, company, or speaker constraints in query."""
        low = query.lower()
        filters: Dict[str, Any] = {
            "quarter": None,
            "company": None,
            "speaker_role": None,
            "is_comparison": False,
            "compare_companies": [],
        }

        # Quarter filter
        m_q = re.search(r"\b(Q[1-4]\s*(?:FY\s*\d{2,4})?)\b", query, re.I)
        if m_q:
            filters["quarter"] = m_q.group(1).upper()

        # Company alias map
        alias_map = {
            "tcs": "TCS",
            "tata consultancy": "TCS",
            "infosys": "Infosys",
            "infy": "Infosys",
            "hcl": "HCL",
            "hcltech": "HCL",
            "hcl tech": "HCL",
            "wipro": "Wipro",
            "ltimindtree": "LTIMindtree",
            "ltim": "LTIMindtree",
            "lti": "LTIMindtree",
            "mindtree": "LTIMindtree",
            "mphasis": "Mphasis",
            "tech mahindra": "Tech_Mahindra",
            "techm": "Tech_Mahindra",
            "coforge": "Coforge",
            "cognizant": "Cognizant",
            "cts": "Cognizant",
            "infoedge": "infoedge",
            "naukri": "infoedge",
        }

        all_comps = list_available_companies()
        matched_comps = []
        for alias, cname in alias_map.items():
            if re.search(rf"\b{re.escape(alias)}\b", low):
                if cname in all_comps and cname not in matched_comps:
                    matched_comps.append(cname)

        for c in all_comps:
            clean_c = c.replace("_", " ").lower()
            if clean_c in low and c not in matched_comps:
                matched_comps.append(c)

        if len(matched_comps) >= 2 or "compare" in low or " vs " in low:
            filters["is_comparison"] = True
            filters["compare_companies"] = matched_comps
        elif len(matched_comps) == 1:
            filters["company"] = matched_comps[0]

        # Speaker role filter
        if "analyst" in low:
            filters["speaker_role"] = "Analyst"
        elif "management" in low or "ceo" in low or "cfo" in low:
            filters["speaker_role"] = "Management"

        return filters

    def answer_query(
        self,
        query: str,
        current_company: str = "",
        top_k: int = 4,
    ) -> Dict[str, Any]:
        """
        Executes evidence retrieval and formats a cited response with
        Company, Quarter, Speaker, Section, and Relevance score.
        """
        filters = self._parse_query_filters(query)

        # Check multi-company comparison query
        if filters["is_comparison"] and len(filters["compare_companies"]) >= 2:
            return self._handle_comparison_query(query, filters["compare_companies"], top_k=2)

        target_company = filters.get("company") or current_company or "TCS"

        # Auto-index if currently unindexed
        if not self._is_indexed or not self.corpus_sentences or not self.vectorizer:
            try:
                data = load_company_data(target_company, use_cache=True)
                vts = data.get("valid_transcripts", [])
                if vts:
                    self.build_index(vts)
            except Exception as exc:
                logger.warning("Could not auto-index for %s: %s", target_company, exc)

        # If a specific company was requested in query and differs from current index, load on-the-fly
        if filters.get("company"):
            req_c = filters["company"].replace("_", " ").upper()
            has_company = any(
                s.get("company", "").replace("_", " ").upper() == req_c
                for s in self.corpus_sentences
            )
            if not has_company:
                try:
                    data = load_company_data(filters["company"], use_cache=True)
                    vts = data.get("valid_transcripts", [])
                    if vts:
                        self.build_index(vts)
                except Exception as exc:
                    logger.warning("Could not switch index to %s: %s", filters["company"], exc)

        if not self._is_indexed or not self.corpus_sentences or not self.vectorizer:
            return {
                "answer": "No transcript evidence is currently indexed. Please select a company from the workspace selector to run the intelligence pipeline.",
                "evidence": pd.DataFrame(),
                "citations": [],
                "is_comparison": False,
            }

        # Vector retrieval
        q_vec = self.vectorizer.transform([query])
        sims = (self.tfidf_matrix @ q_vec.T).toarray().ravel()

        ranked_indices = np.argsort(sims)[::-1]

        def _gather_candidates(strict_quarter: bool = True) -> List[Tuple[Dict[str, Any], float]]:
            matched = []
            for idx in ranked_indices:
                score = float(sims[idx])
                if score <= 0.03:
                    break
                cand = self.corpus_sentences[idx]

                # Apply company filter if specified
                if filters.get("company"):
                    req_c = filters["company"].replace("_", " ").upper()
                    cand_c = cand.get("company", "").replace("_", " ").upper()
                    if cand_c and req_c and cand_c != req_c:
                        continue

                # Apply quarter filter if specified
                if strict_quarter and filters["quarter"]:
                    q_req = filters["quarter"].replace(" ", "").upper()
                    q_cand = cand.get("quarter", "").replace(" ", "").upper()
                    if q_req not in q_cand:
                        continue

                # Apply speaker role filter if specified
                if filters["speaker_role"] and cand.get("speaker_role") != filters["speaker_role"]:
                    continue

                matched.append((cand, score))
                if len(matched) >= top_k:
                    break
            return matched

        matched_sents = _gather_candidates(strict_quarter=True)
        # If strict quarter produced no matches, relax quarter filter gracefully
        if not matched_sents and filters["quarter"]:
            matched_sents = _gather_candidates(strict_quarter=False)

        if not matched_sents:
            return {
                "answer": f"I could not find direct transcript evidence matching '{query}'. Try exploring with key business terms such as 'margins', 'AI pipeline', or 'attrition'.",
                "evidence": pd.DataFrame(),
                "citations": [],
                "is_comparison": False,
            }

        citations = []
        evidence_rows = []

        for cand, score in matched_sents:
            comp = cand.get("company", current_company)
            q = cand.get("quarter", "")
            spk = cand.get("speaker", "Management")
            sec = cand.get("section", "Prepared Remarks")
            txt = cand["text"]

            citations.append({
                "company": comp,
                "quarter": q,
                "speaker": spk,
                "section": sec,
                "text": txt,
                "score": round(score, 3),
            })
            evidence_rows.append({
                "Quarter": q,
                "Speaker": spk,
                "Section": sec,
                "Evidence": txt,
                "Relevance": round(score, 3),
            })

        lead = citations[0]
        # Synthesize with local AI model if available
        ai_synth = ""
        try:
            model, tokenizer = get_ai_generator()
            if model is not None and tokenizer is not None:
                context_sents = " ".join([c["text"] for c in citations[:3]])
                prompt = f"Answer the financial question using the context accurately.\nContext: {context_sents[:450]}\nQuestion: {query}\nAnswer:"
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
                outputs = model.generate(**inputs, max_new_tokens=50, num_beams=2, early_stopping=True)
                gen_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
                if len(gen_text) >= 4 and gen_text.lower() not in query.lower():
                    ai_synth = f"**Executive Synthesis:** {gen_text}\n\n"
        except Exception as exc:
            logger.debug("AI generation fallback: %s", exc)

        answer_text = (
            f"{ai_synth}"
            f"According to {lead['speaker']} in {lead['quarter']} ({lead['section']}):\n\n"
            f"\"{lead['text']}\"\n\n"
        )
        if len(citations) > 1:
            second = citations[1]
            answer_text += f"Further, in {second['quarter']}, {second['speaker']} noted: \"{second['text']}\""

        return {
            "answer": answer_text,
            "evidence": pd.DataFrame(evidence_rows),
            "citations": citations,
            "is_comparison": False,
        }

    def _handle_comparison_query(
        self,
        query: str,
        companies: List[str],
        top_k: int = 2,
    ) -> Dict[str, Any]:
        """Retrieves and contrasts evidence across multiple peer companies."""
        comp_answers = {}
        citations = []
        evidence_rows = []

        for comp in companies:
            try:
                data = load_company_data(comp, use_cache=True)
                vts = data.get("valid_transcripts", [])
                sents = []
                for t in vts:
                    for s in t.get("sentences", []):
                        item = dict(s)
                        item["company"] = comp
                        item["quarter"] = t.get("quarter", "")
                        sents.append(item)
                if not sents:
                    continue

                vec = TfidfVectorizer(stop_words=QA_STOP_WORDS, ngram_range=(1, 2), sublinear_tf=True)
                mat = vec.fit_transform([s["text"] for s in sents])
                q_vec = vec.transform([query])
                sims = (mat @ q_vec.T).toarray().ravel()
                top_idx = np.argsort(sims)[::-1][:top_k]

                top_for_comp = []
                for idx in top_idx:
                    if sims[idx] > 0.05:
                        cand = sents[idx]
                        top_for_comp.append(cand)
                        citations.append({
                            "company": comp,
                            "quarter": cand.get("quarter", ""),
                            "speaker": cand.get("speaker", "Management"),
                            "section": cand.get("section", "Prepared Remarks"),
                            "text": cand["text"],
                            "score": round(float(sims[idx]), 3),
                        })
                        evidence_rows.append({
                            "Company": comp,
                            "Quarter": cand.get("quarter", ""),
                            "Speaker": cand.get("speaker", "Management"),
                            "Evidence": cand["text"],
                            "Relevance": round(float(sims[idx]), 3),
                        })
                comp_answers[comp] = top_for_comp
            except Exception as exc:
                logger.warning("Error searching peer %s: %s", comp, exc)

        summary_parts = [f"**Comparative Analysis for {', '.join(companies)}:**\n"]
        for comp, items in comp_answers.items():
            if items:
                best = items[0]
                summary_parts.append(
                    f"• **{comp} ({best.get('quarter', '')})** — *{best.get('speaker', 'Management')}*: \"{best['text']}\""
                )
            else:
                summary_parts.append(f"• **{comp}**: No direct disclosure found for this specific query.")

        return {
            "answer": "\n\n".join(summary_parts),
            "evidence": pd.DataFrame(evidence_rows),
            "citations": citations,
            "is_comparison": True,
        }


_GLOBAL_QA_ENGINE: RetrievalQAEngine | None = None


def get_qa_engine(transcripts: Optional[List[Dict[str, Any]]] = None) -> RetrievalQAEngine:
    global _GLOBAL_QA_ENGINE
    if _GLOBAL_QA_ENGINE is None:
        _GLOBAL_QA_ENGINE = RetrievalQAEngine()
    if transcripts:
        _GLOBAL_QA_ENGINE.build_index(transcripts)
    elif not _GLOBAL_QA_ENGINE._is_indexed:
        try:
            d = load_company_data("TCS", use_cache=True)
            vts = d.get("valid_transcripts", [])
            if vts:
                _GLOBAL_QA_ENGINE.build_index(vts)
        except Exception as exc:
            logger.warning("Default auto-indexing error in get_qa_engine: %s", exc)
    return _GLOBAL_QA_ENGINE


def get_suggested_questions(company: str) -> List[str]:
    """Generates contextually relevant queries for the user."""
    return [
        f"What did {company} management say about Generative AI deal pipelines?",
        f"How did operating margins move sequentially and what were the primary cost pressures?",
        f"What is the full-year constant currency revenue growth guidance?",
        f"What were the trends in voluntary employee attrition and net headcount additions?",
        f"What deal terms and large TCV bookings were highlighted by {company}?",
        f"Compare {company} and Infosys on margin outlook and pricing stability.",
    ]
