import pytest
import requests

from llm.ollama_client import OllamaClient
from llm.recommendation import SkinRecommendationService


class FakeRetriever:
    def retrieve(self, predicted_label):
        return {
            "matched": True,
            "match_type": "exact",
            "input_label": predicted_label,
            "matched_label": predicted_label,
            "disease": "Acne",
            "context": "Acne may cause pimples, oily skin, and clogged pores.",
        }


def is_ollama_running(base_url: str = "http://localhost:11434") -> bool:
    try:
        response = requests.get(base_url, timeout=3)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


@pytest.mark.integration
def test_generate_recommendation_with_real_ollama():
    if not is_ollama_running():
        pytest.skip("Ollama server is not running")

    service = SkinRecommendationService(
        retriever=FakeRetriever(),
        llm_client=OllamaClient(timeout=120),
    )

    result = service.generate_recommendation(
        predicted_label="Acne",
        confidence=0.85,
    )

    print("\n=== Real Service Recommendation Result ===")
    print("Predicted label:", result["predicted_label"])
    print("Confidence:", result["confidence"])
    print("Disease:", result["retrieval_result"]["disease"])
    print("\nRecommendation:")
    print(result["recommendation"])

    assert result["predicted_label"] == "Acne"
    assert result["retrieval_result"]["matched"] is True
    assert isinstance(result["recommendation"], str)
    assert len(result["recommendation"].strip()) > 0