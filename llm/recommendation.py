import logging
from typing import Any, Dict

from llm.ollama_client import OllamaClient
from llm.prompts import build_skin_disease_prompt
from llm.rag_retriever import SkinDiseaseRetriever


logger = logging.getLogger(__name__)


class SkinRecommendationService:
    def __init__(
        self,
        retriever: SkinDiseaseRetriever,
        llm_client: OllamaClient,
    ):
        self.retriever = retriever
        self.llm_client = llm_client

    def generate_recommendation(
        self,
        predicted_label: str,
        confidence: float,
    ) -> Dict[str, Any]:

        if not predicted_label or not isinstance(predicted_label, str):
            raise ValueError("predicted_label must be a non-empty string")

        if not isinstance(confidence, (float, int)):
            raise TypeError("confidence must be a float or int")

        if confidence < 0 or confidence > 1:
            raise ValueError("confidence must be between 0 and 1")

        logger.info("Generating recommendation for label=%s", predicted_label)

        retrieval_result = self.retriever.retrieve(predicted_label)

        logger.info(
            "Retrieval result: matched=%s, disease=%s",
            retrieval_result.get("matched"),
            retrieval_result.get("disease"),
        )

        messages = build_skin_disease_prompt(
            predicted_label=predicted_label,
            confidence=float(confidence),
            retrieval_result=retrieval_result,
        )

        logger.info("Prompt messages built successfully")

        llm_response = self.llm_client.generate(messages)

        logger.info("LLM recommendation generated successfully")

        return {
            "predicted_label": predicted_label,
            "confidence": round(float(confidence), 4),
            "retrieval_result": retrieval_result,
            "recommendation": llm_response,
        }