import os
from typing import Dict, List

import requests

from utils.logger import get_logger


logger = get_logger(__name__)


class OllamaClient:
    def __init__(
        self,
        model_name: str | None = None,
        base_url: str | None = None,
        timeout: int = 60,
    ):
        logger.info("Initializing OllamaClient")

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

        logger.info(
            "OllamaClient initialized | model=%s | base_url=%s | timeout=%s",
            self.model_name,
            self.base_url,
            self.timeout,
        )

    def generate(self, messages: List[Dict[str, str]]) -> str:
        logger.info("Generating response with Ollama")

        if not messages or not isinstance(messages, list):
            logger.warning(
                "Invalid messages input for Ollama | value=%s | type=%s",
                messages,
                type(messages),
            )
            raise ValueError(
                "messages must be a non-empty list"
            )

        logger.debug(
            "Ollama messages received | message_count=%s",
            len(messages),
        )

        url = f"{self.base_url}/api/chat"

        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
        }

        logger.info(
            "Sending request to Ollama | url=%s | model=%s | timeout=%s",
            url,
            self.model_name,
            self.timeout,
        )

        try:
            response = requests.post(
                url,
                json=payload,
                timeout=self.timeout,
            )

            logger.info(
                "Ollama response received | status_code=%s",
                response.status_code,
            )

            response.raise_for_status()

        except requests.exceptions.RequestException as exc:
            logger.exception(
                "Failed to generate response from Ollama | url=%s | model=%s",
                url,
                self.model_name,
            )
            raise RuntimeError(
                f"Failed to generate response from Ollama: {exc}"
            ) from exc

        data = response.json()

        logger.debug(
            "Ollama response JSON parsed successfully | keys=%s",
            list(data.keys()),
        )

        try:
            content = data["message"]["content"]

        except KeyError as exc:
            logger.exception(
                "Invalid response format from Ollama | response_keys=%s",
                list(data.keys()),
            )
            raise RuntimeError(
                "Invalid response format from Ollama"
            ) from exc

        logger.info(
            "Ollama content generated successfully | content_length=%s",
            len(content.strip()),
        )

        return content.strip()