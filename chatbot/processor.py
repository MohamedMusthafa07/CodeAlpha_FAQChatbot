"""Input validation and text preprocessing for the FAQ chatbot."""
from __future__ import annotations

import re

MAX_INPUT_CHARS = 500

_TOKEN_PATTERN = re.compile(r"[^\W_]+")

STOPWORDS = frozenset(
    {
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "am",
        "do", "does", "did", "i", "me", "my", "we", "our", "you", "your",
        "it", "its", "to", "of", "in", "on", "at", "for", "with", "and",
        "or", "can", "could", "would", "should", "will", "how", "what",
        "when", "where", "which", "who", "why", "there", "this", "that",
        "these", "those", "if", "as", "by", "from", "about", "please",
        "any", "have", "has", "had", "get", "so", "tell", "want", "need",
        "know", "us", "am", "im", "ive", "just", "some", "much", "many",
    }
)


class InvalidQuestionError(ValueError):
    """Raised when the user's input cannot be processed."""


def validate_question(question: object, max_chars: int = MAX_INPUT_CHARS) -> str:
    """Validate raw user input and return a cleaned version.

    Raises InvalidQuestionError with a user-friendly message on bad input.
    """
    if not isinstance(question, str):
        raise InvalidQuestionError("Your question must be text.")

    printable = "".join(ch for ch in question if ch.isprintable() or ch.isspace())
    cleaned = " ".join(printable.split())

    if not cleaned:
        raise InvalidQuestionError("Please type a question before sending.")
    if len(cleaned) > max_chars:
        raise InvalidQuestionError(
            f"Your question is too long ({len(cleaned)} characters). "
            f"Please keep it under {max_chars} characters."
        )
    if not any(ch.isalnum() for ch in cleaned):
        raise InvalidQuestionError("Your question needs to contain letters or numbers.")
    return cleaned


def normalize_text(text: str) -> str:
    """Lowercase, drop apostrophes and punctuation, collapse whitespace."""
    lowered = text.lower().replace("'", "").replace("\u2019", "")
    return " ".join(_TOKEN_PATTERN.findall(lowered))


def stem(token: str) -> str:
    """Very light suffix stripping so that e.g. 'certificates' == 'certificate'."""
    if len(token) <= 3:
        return token
    if token.endswith("ies") and len(token) > 4:
        token = token[:-3] + "y"
    elif token.endswith("s") and not token.endswith("ss"):
        token = token[:-1]
    for suffix in ("ing", "ed"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            token = token[: -len(suffix)]
            break
    if token.endswith("e") and len(token) > 3:
        token = token[:-1]
    return token


def preprocess(text: str) -> str:
    """Full pipeline: normalize -> remove stop words -> light stemming."""
    tokens = normalize_text(text).split()
    return " ".join(stem(tok) for tok in tokens if tok not in STOPWORDS)
