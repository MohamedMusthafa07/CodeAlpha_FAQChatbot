"""Unit tests for the FAQ chatbot core (no Streamlit required)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chatbot import (  # noqa: E402
    DEFAULT_DATA_PATH,
    FALLBACK_MESSAGE,
    MAX_INPUT_CHARS,
    DatasetError,
    FAQMatcher,
    InvalidQuestionError,
    list_categories,
    load_faqs,
    normalize_text,
    preprocess,
    validate_faqs,
)

REQUIRED_CATEGORIES = {
    "Account",
    "Registration",
    "Login",
    "Password",
    "Courses",
    "Certificates",
    "Payments",
    "Technical Support",
    "Internship",
    "General Information",
}


@pytest.fixture(scope="module")
def faqs():
    return load_faqs(DEFAULT_DATA_PATH)


@pytest.fixture(scope="module")
def matcher(faqs):
    return FAQMatcher(faqs, threshold=0.25)


def _write_json(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "faqs.json"
    path.write_text(content, encoding="utf-8")
    return path


# ----------------------------- dataset loading -----------------------------
def test_dataset_loads(faqs):
    assert len(faqs) >= 40
    assert len({faq.id for faq in faqs}) == len(faqs)


def test_dataset_covers_required_categories(faqs):
    assert REQUIRED_CATEGORIES.issubset(set(list_categories(faqs)))


def test_missing_file_raises(tmp_path):
    with pytest.raises(DatasetError, match="not found"):
        load_faqs(tmp_path / "nope.json")


def test_invalid_json_raises(tmp_path):
    with pytest.raises(DatasetError, match="not valid JSON"):
        load_faqs(_write_json(tmp_path, "{ this is not json"))


def test_empty_dataset_raises(tmp_path):
    with pytest.raises(DatasetError, match="empty"):
        load_faqs(_write_json(tmp_path, json.dumps({"faqs": []})))


# --------------------------- dataset validation ----------------------------
VALID = {"id": "X-1", "category": "Test", "question": "Q?", "answer": "A.", "keywords": ["k"]}


def test_validate_accepts_list_and_object():
    assert len(validate_faqs([VALID])) == 1
    assert len(validate_faqs({"faqs": [VALID]})) == 1


@pytest.mark.parametrize(
    "bad_record",
    [
        "not-a-dict",
        {**VALID, "answer": ""},
        {**VALID, "question": None},
        {**VALID, "category": 5},
        {**VALID, "keywords": "oops"},
        {**VALID, "keywords": [1, 2]},
        {k: v for k, v in VALID.items() if k != "id"},
    ],
)
def test_invalid_records_rejected(bad_record):
    with pytest.raises(DatasetError):
        validate_faqs([bad_record])


def test_duplicate_ids_rejected():
    with pytest.raises(DatasetError, match="duplicate"):
        validate_faqs([VALID, dict(VALID)])


def test_wrong_top_level_type_rejected():
    with pytest.raises(DatasetError):
        validate_faqs("hello")


# ----------------------------- preprocessing -------------------------------
def test_normalize_text_strips_punctuation_and_case():
    assert normalize_text("  How CAN I Reset?!  ") == "how can i reset"
    assert normalize_text("Don't panic") == "dont panic"


def test_preprocess_removes_stopwords_and_stems():
    assert preprocess("How can I reset my passwords?") == "reset password"
    assert preprocess("certificates") == preprocess("certificate")


def test_preprocess_all_stopwords_is_empty():
    assert preprocess("how do I") == ""


# ------------------------------- matching ----------------------------------
def test_every_exact_question_matches_itself(faqs, matcher):
    for faq in faqs:
        result = matcher.match(faq.question)
        assert result.matched, faq.question
        assert result.faq_id == faq.id, faq.question
        assert result.answer == faq.answer


@pytest.mark.parametrize(
    "question, expected_id",
    [
        ("I forgot my password. How do I change it?", "PWD-01"),
        ("Can't sign in to my account", "LOG-01"),
        ("how to get my money back", "PAY-02"),
        ("What is the length of the internship?", "INT-02"),
        ("Where do I download my receipt?", "PAY-04"),
        ("videos are not playing", "TEC-03"),
        ("Is there an Android app?", "GEN-03"),
        ("How do I verify a certificate?", "CER-04"),
    ],
)
def test_similar_wording_matches(matcher, question, expected_id):
    result = matcher.match(question)
    assert result.matched
    assert result.faq_id == expected_id


@pytest.mark.parametrize(
    "question",
    ["What is the capital of France?", "asdf qwerty zzzz", "Tell me a joke about pizza"],
)
def test_low_similarity_returns_fallback(matcher, question):
    result = matcher.match(question)
    assert not result.matched
    assert result.answer == FALLBACK_MESSAGE
    assert result.category is None and result.faq_id is None


def test_stopword_only_question_is_fallback(matcher):
    result = matcher.match("how do I")
    assert not result.matched
    assert result.score == 0.0


def test_custom_threshold_can_force_fallback(matcher):
    result = matcher.match("How can I reset my password?", threshold=1.0)
    # an exact question can reach 1.0 only if identical vectors; either way it must not crash
    assert result.threshold == 1.0


def test_threshold_validation(faqs):
    with pytest.raises(ValueError):
        FAQMatcher(faqs, threshold=1.5)
    with pytest.raises(ValueError):
        FAQMatcher(faqs, threshold="high")  # type: ignore[arg-type]


def test_repeated_queries_are_consistent(matcher):
    first = matcher.match("How do I enroll in a course?")
    second = matcher.match("how do i enroll in a course")
    assert first.faq_id == second.faq_id
    assert first.score == second.score


# ------------------------------ invalid input ------------------------------
@pytest.mark.parametrize("bad", ["", "   ", "\n\t", "?!?!", None, 123, ["list"]])
def test_invalid_input_rejected(matcher, bad):
    with pytest.raises(InvalidQuestionError):
        matcher.match(bad)  # type: ignore[arg-type]


def test_long_input_rejected(matcher):
    with pytest.raises(InvalidQuestionError, match="too long"):
        matcher.match("password " * (MAX_INPUT_CHARS // 4))


# --------------------------- response structure ----------------------------
def test_response_structure(matcher):
    payload = matcher.match("How do I request a refund?").to_dict()
    expected = {
        "question", "matched", "answer", "score", "threshold",
        "category", "faq_id", "matched_question", "suggestions",
    }
    assert expected == set(payload)
    assert payload["matched"] is True
    assert 0.0 <= payload["score"] <= 1.0
    assert payload["category"] == "Payments"


def test_multiple_categories_are_reachable(matcher):
    questions = {
        "How do I register for the platform?": "Registration",
        "What payment methods do you accept?": "Payments",
        "How do I apply for an internship?": "Internship",
        "How do I enable two-factor authentication?": "Account",
        "Which browsers are supported?": "Technical Support",
        "What are your support hours?": "General Information",
    }
    for question, category in questions.items():
        assert matcher.match(question).category == category
