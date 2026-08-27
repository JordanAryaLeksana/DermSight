import threading


class LLMService:
    _service = None
    _lock = threading.Lock()

    def __init__(self, disease_list_path, base_url, model_name, timeout):
        self.settings = (disease_list_path, base_url, model_name, timeout)

    def _get_service(self):
        if self.__class__._service is None:
            with self.__class__._lock:
                if self.__class__._service is None:
                    from llm.ollama_client import OllamaClient
                    from llm.rag_retriever import SkinDiseaseRetriever
                    from llm.recommendation import SkinRecommendationService

                    disease_path, base_url, model_name, timeout = self.settings
                    self.__class__._service = SkinRecommendationService(
                        retriever=SkinDiseaseRetriever(disease_path),
                        llm_client=OllamaClient(
                            base_url=base_url, model_name=model_name, timeout=timeout
                        ),




                    )
        return self.__class__._service

    def analyze(self, label, confidence):
        return self._get_service().generate_recommendation(label, confidence)
