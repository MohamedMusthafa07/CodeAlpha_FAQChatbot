"""CodeAlpha FAQ Chatbot - local NLP engine (TF-IDF + cosine similarity)."""
from __future__ import annotations

from .data_loader import (
    DEFAULT_DATA_PATH,
    FAQ,
    DatasetError,
    list_categories,
    load_faqs,
    validate_faqs,
)
from .matcher import (
    DEFAULT_THRESHOLD,
    FALLBACK_MESSAGE,
    FAQMatcher,
    MatchResult,
)
from .processor import (
    MAX_INPUT_CHARS,
    InvalidQuestionError,
    normalize_text,
    preprocess,
    validate_question,
)

__all__ = [
    "DEFAULT_DATA_PATH",
    "DEFAULT_THRESHOLD",
    "FALLBACK_MESSAGE",
    "FAQ",
    "FAQMatcher",
    "MAX_INPUT_CHARS",
    "DatasetError",
    "InvalidQuestionError",
    "MatchResult",
    "list_categories",
    "load_faqs",
    "normalize_text",
    "preprocess",
    "validate_faqs",
    "validate_question",
]
