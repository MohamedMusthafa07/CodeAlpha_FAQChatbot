"""Load and validate the local FAQ knowledge base (JSON)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Union

DEFAULT_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "faqs.json"

_REQUIRED_TEXT_FIELDS = ("category", "question", "answer")


class DatasetError(Exception):
    """Raised when the FAQ dataset is missing, unreadable, or invalid."""


@dataclass(frozen=True)
class FAQ:
    """A single validated FAQ record."""

    id: str
    category: str
    question: str
    answer: str
    keywords: tuple[str, ...] = ()

    @property
    def index_text(self) -> str:
        """Text used to build the search index (question + keywords)."""
        return " ".join([self.question, *self.keywords])


def _parse_record(record: Any, position: int) -> FAQ:
    """Validate one raw record and convert it to an FAQ."""
    label = f"FAQ #{position}"
    if not isinstance(record, dict):
        raise DatasetError(f"{label}: expected an object, got {type(record).__name__}.")

    raw_id = record.get("id")
    if isinstance(raw_id, bool) or not isinstance(raw_id, (int, str)) or not str(raw_id).strip():
        raise DatasetError(f"{label}: 'id' must be a non-empty string or integer.")

    for field in _REQUIRED_TEXT_FIELDS:
        value = record.get(field)
        if not isinstance(value, str) or not value.strip():
            raise DatasetError(f"{label} (id={raw_id}): '{field}' must be a non-empty string.")

    keywords = record.get("keywords", [])
    if not isinstance(keywords, list) or not all(isinstance(k, str) for k in keywords):
        raise DatasetError(f"{label} (id={raw_id}): 'keywords' must be a list of strings.")

    return FAQ(
        id=str(raw_id).strip(),
        category=record["category"].strip(),
        question=record["question"].strip(),
        answer=record["answer"].strip(),
        keywords=tuple(k.strip() for k in keywords if k.strip()),
    )


def validate_faqs(raw: Any) -> list[FAQ]:
    """Validate parsed JSON and return a list of FAQ objects.

    Accepts either a list of records or an object with a "faqs" list.
    """
    records = raw.get("faqs") if isinstance(raw, dict) else raw
    if not isinstance(records, list):
        raise DatasetError('The dataset must be a list of FAQs or an object with a "faqs" list.')
    if not records:
        raise DatasetError("The FAQ dataset is empty.")

    faqs: list[FAQ] = []
    seen_ids: set[str] = set()
    for position, record in enumerate(records, start=1):
        faq = _parse_record(record, position)
        if faq.id in seen_ids:
            raise DatasetError(f"FAQ #{position}: duplicate id '{faq.id}'.")
        seen_ids.add(faq.id)
        faqs.append(faq)
    return faqs


def load_faqs(path: Union[str, Path, None] = None) -> list[FAQ]:
    """Read, parse and validate the FAQ JSON file."""
    file_path = Path(path) if path is not None else DEFAULT_DATA_PATH
    if not file_path.is_file():
        raise DatasetError(f"FAQ dataset not found: {file_path.name}. Expected at: {file_path}")

    try:
        text = file_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise DatasetError(f"Could not read the FAQ dataset: {exc}") from exc

    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise DatasetError(
            f"The FAQ dataset is not valid JSON (line {exc.lineno}, column {exc.colno}): {exc.msg}."
        ) from exc

    return validate_faqs(raw)


def list_categories(faqs: Iterable[FAQ]) -> list[str]:
    """Return the sorted unique categories of the given FAQs."""
    return sorted({faq.category for faq in faqs})
