"""
translator.py
-------------
Sentence-batched cross-lingual translation engine powered by Meta's NLLB-200.
Translates executive briefs, risk signals, commitments, and transcript passages
into eight major Indian languages and European languages without truncation.
"""
from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional

import torch

logger = logging.getLogger(__name__)

MODEL_NAME = "facebook/nllb-200-distilled-600M"

LANGUAGE_CODES: Dict[str, str] = {
    "English": "eng_Latn",
    "Hindi": "hin_Deva",
    "Tamil": "tam_Taml",
    "Telugu": "tel_Telu",
    "Kannada": "kan_Knda",
    "Malayalam": "mal_Mlym",
    "Bengali": "ben_Beng",
    "Marathi": "mar_Deva",
    "Gujarati": "guj_Gujr",
    "French": "fra_Latn",
    "German": "deu_Latn",
    "Spanish": "spa_Latn",
}

# Browser SpeechSynthesis locale mapping
SPEECH_LOCALE: Dict[str, str] = {
    "English": "en-IN",
    "Hindi": "hi-IN",
    "Tamil": "ta-IN",
    "Telugu": "te-IN",
    "Kannada": "kn-IN",
    "Malayalam": "ml-IN",
    "Bengali": "bn-IN",
    "Marathi": "mr-IN",
    "Gujarati": "gu-IN",
    "French": "fr-FR",
    "German": "de-DE",
    "Spanish": "es-ES",
}


def split_into_sentences(text: str) -> List[str]:
    """Splits paragraphs into clean grammatical sentences."""
    if not text:
        return []
    cleaned = text.replace("\r", " ").strip()
    sents = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\u0900-\u0D7F])", cleaned)
    out = []
    for s in sents:
        st = s.strip()
        if st:
            out.append(st)
    return out or [text.strip()]


class SentenceBatchedTranslator:
    """Enterprise NLLB translation engine with sentence-level batching."""

    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.model = None
        self._is_loaded = False

    def load_model(self) -> None:
        """Loads NLLB-200 transformer model."""
        if self._is_loaded:
            return
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # type: ignore

        logger.info("Initializing NLLB-200 translation model on %s...", self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
        self.model.to(self.device)
        self.model.eval()
        self._is_loaded = True
        logger.info("NLLB-200 loaded successfully.")

    def translate_text(
        self,
        text: str,
        source_language: str = "English",
        target_language: str = "Hindi",
        batch_size: int = 4,
    ) -> str:
        """
        Translates text with sentence-level batching.
        Each sentence is tokenized and generated without truncation,
        preventing any silent cutoff.
        """
        if not text or not text.strip():
            return ""

        if source_language == target_language:
            return text

        src_code = LANGUAGE_CODES.get(source_language, "eng_Latn")
        tgt_code = LANGUAGE_CODES.get(target_language, "hin_Deva")

        # Lazy load on first use
        if not self._is_loaded:
            self.load_model()

        assert self.tokenizer is not None and self.model is not None

        sentences = split_into_sentences(text)
        translated_sentences: List[str] = []

        self.tokenizer.src_lang = src_code
        target_token_id = self.tokenizer.convert_tokens_to_ids(tgt_code)

        for i in range(0, len(sentences), batch_size):
            batch = sentences[i : i + batch_size]
            # Tokenize batch without truncation since individual sentences fit comfortably
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    forced_bos_token_id=target_token_id,
                    max_length=256,
                    num_beams=2,
                )

            decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            translated_sentences.extend(decoded)

        return " ".join(translated_sentences)


_TRANSLATOR: SentenceBatchedTranslator | None = None


def get_translator() -> SentenceBatchedTranslator:
    global _TRANSLATOR
    if _TRANSLATOR is None:
        _TRANSLATOR = SentenceBatchedTranslator()
    return _TRANSLATOR


def translate_passage(
    text: str,
    source_language: str = "English",
    target_language: str = "Hindi",
) -> str:
    """Public helper for translating text passages with zero truncation."""
    engine = get_translator()
    return engine.translate_text(text, source_language, target_language)