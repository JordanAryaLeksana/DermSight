import threading


class LLMService:
    _service = None
    _lock = threading.Lock()

    def __init__(
        self,
        disease_list_path,
        timeout=60,
    ):
        self.disease_list_path = disease_list_path
        self.timeout = timeout

    def _get_service(self):
        if self.__class__._service is None:
            with self.__class__._lock:
                if self.__class__._service is None:
                    from llm.sumopod_client import SumoPodClient
                    from llm.rag_retriever import SkinDiseaseRetriever
                    from llm.recommendation import SkinRecommendationService

                    self.__class__._service = SkinRecommendationService(
                        retriever=SkinDiseaseRetriever(
                            self.disease_list_path
                        ),
                        llm_client=SumoPodClient()
                    )

        return self.__class__._service

    def analyze(
        self,
        label,
        confidence,
    ):
        return (
            self._get_service()
            .generate_recommendation(
                label,
                confidence,
            )
        )