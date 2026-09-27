import pytest

from src.grounding import parse_answerability_json, validate_numeric_date_facts


def test_parse_answerability_json_accepts_structured_result():
    answerable, reason = parse_answerability_json(
        '{"answerable": false, "reason": "운영시간 정보가 없음"}'
    )

    assert answerable is False
    assert reason == "운영시간 정보가 없음"


def test_parse_answerability_json_rejects_unstructured_result():
    with pytest.raises(ValueError, match="valid JSON"):
        parse_answerability_json("ANSWERABLE")


def test_numeric_date_facts_are_supported_by_cited_context():
    valid, unsupported = validate_numeric_date_facts(
        answer="2025년 총수입은 386,585,434,203원입니다. [S1]\n출처: [S1]",
        cited_source_texts=["2025년 수입 합계 386,585,434,203 원"],
    )

    assert valid is True
    assert unsupported == []


def test_numeric_date_facts_reject_unsupported_value():
    valid, unsupported = validate_numeric_date_facts(
        answer="총수입은 999원입니다. [S1]\n출처: [S1]",
        cited_source_texts=["수입 합계 100원"],
    )

    assert valid is False
    assert unsupported == ["999원"]
