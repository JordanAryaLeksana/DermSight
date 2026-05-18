import os
from typing import Dict, List

import requests


class OllamaClient:
    def __init__(
        self,
        model_name: str | None = None,
        base_url: str | None = None,
        timeout: int = 60,
    ):
        self.model_name = (
            model_name or os.getenv("OLLAMA_MODEL", "gemma4:e2b")
        )

        self.base_url = (
            base_url or os.getenv(
                "OLLAMA_BASE_URL",
                "http://localhost:11434"
            )
        ).rstrip("/")

        self.timeout = timeout

    def generate(self, messages: List[Dict[str, str]]) -> str:
        if not messages or not isinstance(messages, list):
            raise ValueError(
                "messages must be a non-empty list"
            )

        url = f"{self.base_url}/api/chat"

        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
        }

        try:
            response = requests.post(
                url,
                json=payload,
                timeout=self.timeout,
            )

            response.raise_for_status()

        except requests.exceptions.RequestException as exc:
            raise RuntimeError(
                f"Failed to generate response from Ollama: {exc}"
            ) from exc

        data = response.json()

        try:
            content = data["message"]["content"]

        except KeyError as exc:
            raise RuntimeError(
                "Invalid response format from Ollama"
            ) from exc

        return content.strip()