import os

from llm.ollama_client import OllamaClient
from llm.rag_retriever import SkinDiseaseRetriever
from llm.recommendation import SkinRecommendationService

from api.inference.predictor import SkinDiseasePredictor


_skin_predictor = None
_recommendation_service = None


def get_skin_predictor() -> SkinDiseasePredictor:
    global _skin_predictor

    if _skin_predictor is None:
        model_path = os.getenv(
            "SKIN_MODEL_PATH",
            "src/outputs/final_model.weights.h5",
        )

        class_names_path = os.getenv(
            "CLASS_NAMES_PATH",
            "src/outputs/class_names.json",
        )

        config_path = os.getenv(
            "MODEL_CONFIG_PATH",
            "src/outputs/config.json",
        )

        _skin_predictor = SkinDiseasePredictor(
            model_path=model_path,
            class_names_path=class_names_path,
            config_path=config_path,
        )

    return _skin_predictor


def get_recommendation_service() -> SkinRecommendationService:
    global _recommendation_service

    if _recommendation_service is None:
        disease_list_path = os.getenv(
            "DISEASE_LIST_PATH",
            "llm/data/skin_knowledge_serving.csv",
        )

        retriever = SkinDiseaseRetriever(
            disease_list_path=disease_list_path,
        )

        llm_client = OllamaClient()

        _recommendation_service = SkinRecommendationService(
            retriever=retriever,
            llm_client=llm_client,
        )

    return _recommendation_service