import logging
from pprint import pprint
import pytest

from llm.recommendation import SkinRecommendationService


class FakeRetriever:
    def retrieve(self, predicted_label):
        return {
            "matched": True,
            "match_type": "exact",
            "input_label": predicted_label,
            "matched_label": predicted_label,
            "disease": "Acne",
            "context": "Acne may cause pimples and oily skin.",
        }


class FakeLLMClient:
    def generate(self, messages):
        return "This is a safe AI-generated skin health explanation."


@pytest.fixture
def service():
    return SkinRecommendationService(
        retriever=FakeRetriever(),
        llm_client=FakeLLMClient(),
    )





def test_generate_recommendation_returns_expected_structure(service):
    result = service.generate_recommendation(
        predicted_label="Acne",
        confidence=0.461,
    )

    print(result["recommendation"])

    assert isinstance(result, dict)
    assert result["predicted_label"] == "Acne"
    assert result["confidence"] == 0.87
    assert result["retrieval_result"]["disease"] == "Acne"
    assert result["recommendation"] == (
        "This is a safe AI-generated skin health explanation."
    )


def test_generate_recommendation_calls_retriever_and_llm(service):
    result = service.generate_recommendation(
        predicted_label="Acne",
        confidence=0.90,
    )

    assert result["retrieval_result"]["matched"] is True
    assert "AI-generated" in result["recommendation"]


def test_generate_recommendation_logs_process(service, caplog):
    caplog.set_level(logging.INFO, logger="DERMIGHT")

    logging.getLogger("DERMIGHT").propagate = True

    service.generate_recommendation(
        predicted_label="Acne",
        confidence=0.85,
    )

    assert "Generating recommendation" in caplog.text
    assert "predicted_label=Acne" in caplog.text
    assert "confidence=0.85" in caplog.text
    assert "Retrieval result received" in caplog.text
    assert "disease=Acne" in caplog.text
    assert "Prompt messages built successfully" in caplog.text
    assert "LLM recommendation generated successfully" in caplog.text


def test_empty_predicted_label_raises_error(service):
    with pytest.raises(ValueError):
        service.generate_recommendation(
            predicted_label="",
            confidence=0.85,
        )


def test_invalid_confidence_type_raises_error(service):
    with pytest.raises(TypeError):
        service.generate_recommendation(
            predicted_label="Acne",
            confidence="high",
        )


def test_confidence_below_zero_raises_error(service):
    with pytest.raises(ValueError):
        service.generate_recommendation(
            predicted_label="Acne",
            confidence=-0.1,
        )


def test_confidence_above_one_raises_error(service):
    with pytest.raises(ValueError):
        service.generate_recommendation(
            predicted_label="Acne",
            confidence=1.2,
        )