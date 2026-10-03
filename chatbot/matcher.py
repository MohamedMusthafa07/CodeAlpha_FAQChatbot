"""TF-IDF + cosine-similarity FAQ matching engine."""
from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import asdict, dataclass
from typing import Optional, Sequence

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .data_loader import FAQ
from .processor import MAX_INPUT_CHARS, preprocess, validate_question

DEFAULT_THRESHOLD = 0.25

FALLBACK_MESSAGE = (
    "I couldn't find a confident answer to that in my FAQ knowledge base. "
    "Could you rephrase your question or add a little more detail? "
    "You can also contact technical support for further help."
)

_TOP_K = 3


@dataclass(frozen=True)
class MatchResult:
    """Structured response returned for every question."""

    question: str
    matched: bool
    answer: str
    score: float
    threshold: float
    category: Optional[str] = None
    faq_id: Optional[str] = None
    matched_question: Optional[str] = None
    suggestions: tuple[tuple[str, float], ...] = ()

    def to_dict(self) -> dict:
        return asdict(self)


def _check_threshold(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Threshold must be a number between 0 and 1.")
    if not 0.0 <= float(value) <= 1.0:
        raise ValueError("Threshold must be between 0 and 1.")
    return float(value)


class FAQMatcher:
    """Builds the TF-IDF index once and answers questions quickly.

    The match score is a cosine similarity between TF-IDF vectors
    (0 = nothing in common, 1 = identical wording). It is NOT a probability.
    """

    def __init__(
        self,
        faqs: Sequence[FAQ],
        threshold: float = DEFAULT_THRESHOLD,
        max_input_chars: int = MAX_INPUT_CHARS,
        cache_size: int = 256,
    ) -> None:
        if not faqs:
            raise ValueError("At least one FAQ is required to build the matcher.")
        self._faqs: tuple[FAQ, ...] = tuple(faqs)
        self.threshold = _check_threshold(threshold)
        self._max_input_chars = max_input_chars
        self._cache_size = max(1, cache_size)
        self._cache: OrderedDict[str, tuple[tuple[int, float], ...]] = OrderedDict()
        self._lock = threading.Lock()

        self._vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            token_pattern=r"(?u)\b\w+\b",
        )
        documents = [preprocess(faq.index_text) for faq in self._faqs]
        try:
            self._matrix = self._vectorizer.fit_transform(documents)
        except ValueError as exc:
            raise ValueError("The FAQ dataset does not contain usable text.") from exc

    @property
    def faqs(self) -> tuple[FAQ, ...]:
        return self._faqs

    def _rank(self, processed: str) -> tuple[tuple[int, float], ...]:
        """Return (faq_index, score) pairs, best first. Results are cached."""
        if not processed:
            return ()
        with self._lock:
            cached = self._cache.get(processed)
            if cached is not None:
                self._cache.move_to_end(processed)
                return cached

        vector = self._vectorizer.transform([processed])
        if vector.nnz == 0:
            ranked: tuple[tuple[int, float], ...] = ()
        else:
            sims = cosine_similarity(vector, self._matrix).ravel()
            order = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:_TOP_K]
            ranked = tuple((i, float(sims[i])) for i in order if sims[i] > 0)

        with self._lock:
            self._cache[processed] = ranked
            self._cache.move_to_end(processed)
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        return ranked

    def match(self, question: str, threshold: Optional[float] = None) -> MatchResult:
        """Find the best FAQ for a question, or return a fallback response.

        Raises InvalidQuestionError for empty, invalid, or overly long input.
        """
        cleaned = validate_question(question, self._max_input_chars)
        limit = self.threshold if threshold is None else _check_threshold(threshold)
        ranked = self._rank(preprocess(cleaned))

        if ranked and ranked[0][1] >= limit:
            best_index, best_score = ranked[0]
            best = self._faqs[best_index]
            others = tuple(
                (self._faqs[i].question, round(s, 4)) for i, s in ranked[1:] if s > 0
            )
            return MatchResult(
                question=cleaned,
                matched=True,
                answer=best.answer,
                score=round(best_score, 4),
                threshold=limit,
                category=best.category,
                faq_id=best.id,
                matched_question=best.question,
                suggestions=others,
            )

        best_score = ranked[0][1] if ranked else 0.0
        close = tuple(
            (self._faqs[i].question, round(s, 4))
            for i, s in ranked
            if s > 0 and s >= limit * 0.5
        )
        return MatchResult(
            question=cleaned,
            matched=False,
            answer=FALLBACK_MESSAGE,
            score=round(best_score, 4),
            threshold=limit,
            suggestions=close,
        )
