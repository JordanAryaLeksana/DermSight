from typing import Any, Dict

from llm.ollama_client import OllamaClient
from llm.prompts import build_skin_disease_prompt
from llm.rag_retriever import SkinDiseaseRetriever
from utils.logger import get_logger


logger = get_logger(__name__)


class SkinRecommendationService:
    def __init__(
        self,
        retriever: SkinDiseaseRetriever,
        llm_client: OllamaClient,
    ):
        logger.info("Initializing SkinRecommendationService")

        self.retriever = retriever
        self.llm_client = llm_client

        logger.info("SkinRecommendationService initialized successfully")

    def generate_recommendation(
        self,
        predicted_label: str,
        confidence: float,
    ) -> Dict[str, Any]:

        logger.info(
            "Generating recommendation | predicted_label=%s | confidence=%s",
            predicted_label,
            confidence,
        )

        if not predicted_label or not isinstance(predicted_label, str):
            logger.warning(
                "Invalid predicted_label received | value=%s | type=%s",
                predicted_label,
                type(predicted_label),
            )
            raise ValueError("predicted_label must be a non-empty string")

        if not isinstance(confidence, (float, int)):
            logger.warning(
                "Invalid confidence type received | value=%s | type=%s",
                confidence,
                type(confidence),
            )
            raise TypeError("confidence must be a float or int")

        if confidence < 0 or confidence > 1:
            logger.warning(
                "Confidence out of range | confidence=%s",
                confidence,
            )
            raise ValueError("confidence must be between 0 and 1")

        confidence = float(confidence)

        logger.info("Starting retrieval step | label=%s", predicted_label)

        retrieval_result = self.retriever.retrieve(predicted_label)

        logger.info(
            "Retrieval result received | matched=%s | match_type=%s | disease=%s",
            retrieval_result.get("matched"),
            retrieval_result.get("match_type"),
            retrieval_result.get("disease"),
        )

        logger.info("Building prompt messages")

        messages = build_skin_disease_prompt(
            predicted_label=predicted_label,
            confidence=confidence,
            retrieval_result=retrieval_result,
        )

        logger.info(
            "Prompt messages built successfully | message_count=%s",
            len(messages),
        )

        logger.info("Starting LLM recommendation generation")

        llm_response = self.llm_client.generate(messages)

        logger.info(
            "LLM recommendation generated successfully | response_length=%s",
            len(llm_response),
        )

        result = {
            "predicted_label": predicted_label,
            "confidence": round(confidence, 4),
            "retrieval_result": retrieval_result,
            "recommendation": llm_response,
            "generation_mode": "llm_from_structured_knowledge",
        }

        logger.info(
            "Recommendation service completed successfully | label=%s | confidence=%.4f",
            predicted_label,
            round(confidence, 4),
        )

        return result