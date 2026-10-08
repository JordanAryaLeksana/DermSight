import os
from typing import Any, Dict, List

import requests

from utils.logger import get_logger


logger = get_logger(__name__)


class SumoPodClient:
    def __init__(
        self,
        model_name: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: int = 60,
    ):
        logger.info("Initializing SumoPodClient")

        self.model_name = (
            model_name
            or os.getenv(
                "SUMOPOD_CHAT_MODEL",
                "glm-5.3-flash",
            )
        )

        self.base_url = (
            base_url
            or os.getenv(
                "SUMOPOD_BASE_URL",
                "https://ai.sumopod.com/v1",
            )
        ).rstrip("/")

        self.api_key = (
            api_key
            or os.getenv("SUMOPOD_API_KEY")
        )

        self.timeout = timeout

        self.max_tokens = int(
            os.getenv(
                "SUMOPOD_MAX_TOKENS",
                "900",
            )
        )

        self.max_input_chars = int(
            os.getenv(
                "SUMOPOD_MAX_INPUT_CHARS",
                "12000",
            )
        )

        self.max_messages = int(
            os.getenv(
                "SUMOPOD_MAX_MESSAGES",
                "8",
            )
        )

        if not self.api_key:
            raise RuntimeError(
                "SUMOPOD_API_KEY is not configured"
            )

        if self.max_tokens <= 0:
            raise ValueError(
                "SUMOPOD_MAX_TOKENS must be greater than 0"
            )

        if self.max_input_chars <= 0:
            raise ValueError(
                "SUMOPOD_MAX_INPUT_CHARS must be greater than 0"
            )

        if self.max_messages <= 0:
            raise ValueError(
                "SUMOPOD_MAX_MESSAGES must be greater than 0"
            )

        logger.info(
            "SumoPodClient initialized | "
            "model=%s | "
            "base_url=%s | "
            "api_key_configured=%s | "
            "timeout=%s | "
            "max_tokens=%s | "
            "max_input_chars=%s | "
            "max_messages=%s",
            self.model_name,
            self.base_url,
            bool(self.api_key),
            self.timeout,
            self.max_tokens,
            self.max_input_chars,
            self.max_messages,
        )

    def _validate_messages(
        self,
        messages: List[Dict[str, Any]],
    ) -> int:
        if not isinstance(messages, list):
            raise ValueError(
                "messages must be a list"
            )

        if not messages:
            raise ValueError(
                "messages must not be empty"
            )

        if len(messages) > self.max_messages:
            raise ValueError(
                f"Too many messages: "
                f"{len(messages)} > {self.max_messages}"
            )

        allowed_roles = {
            "system",
            "user",
            "assistant",
        }

        total_chars = 0

        for index, message in enumerate(messages):
            if not isinstance(message, dict):
                raise ValueError(
                    f"Message at index {index} "
                    "must be a dictionary"
                )

            role = message.get("role")
            content = message.get("content")

            if role not in allowed_roles:
                raise ValueError(
                    f"Invalid role at index {index}: "
                    f"{role}"
                )

            if not isinstance(content, str):
                raise ValueError(
                    f"Message content at index {index} "
                    "must be a string"
                )

            if not content.strip():
                raise ValueError(
                    f"Message content at index {index} "
                    "must not be empty"
                )

            total_chars += len(content)

        if total_chars > self.max_input_chars:
            raise ValueError(
                f"Prompt is too large: "
                f"{total_chars} characters > "
                f"{self.max_input_chars}"
            )

        return total_chars

    def generate(
        self,
        messages: List[Dict[str, Any]],
    ) -> str:
        logger.info(
            "Generating response with SumoPod"
        )

        total_chars = self._validate_messages(
            messages
        )

        logger.info(
            "LLM request validated | "
            "message_count=%s | "
            "input_chars=%s | "
            "max_output_tokens=%s",
            len(messages),
            total_chars,
            self.max_tokens,
        )

        url = (
            f"{self.base_url}"
            "/chat/completions"
        )

        headers = {
            "Authorization":
                f"Bearer {self.api_key}",
            "Content-Type":
                "application/json",
        }

        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "temperature": 0.1,
            "top_p": 0.7,
            "max_tokens": self.max_tokens,
        }

        logger.info(
            "Sending request to SumoPod | "
            "url=%s | model=%s",
            url,
            self.model_name,
        )

        try:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )

            logger.info(
                "SumoPod response received | "
                "status_code=%s",
                response.status_code,
            )

            response.raise_for_status()

        except requests.exceptions.RequestException as exc:
            logger.exception(
                "Failed to generate response from SumoPod"
            )

            raise RuntimeError(
                "Failed to generate response "
                f"from SumoPod: {exc}"
            ) from exc

        try:
            data = response.json()

        except ValueError as exc:
            logger.exception(
                "Invalid JSON response from SumoPod"
            )

            raise RuntimeError(
                "Invalid JSON response from SumoPod"
            ) from exc

        try:
            choice = data["choices"][0]

            content = (
                choice["message"]["content"]
            )

            finish_reason = choice.get(
                "finish_reason"
            )

        except (
            KeyError,
            IndexError,
            TypeError,
        ) as exc:
            logger.exception(
                "Invalid SumoPod response format | "
                "response=%s",
                data,
              )

            raise RuntimeError(
                "Invalid response format from SumoPod"
            ) from exc

        if not isinstance(content, str):
            raise RuntimeError(
                "SumoPod returned invalid content"
            )

        if not content.strip():
            raise RuntimeError(
                "SumoPod returned empty content"
            )

        if finish_reason == "length":
            logger.warning(
                "SumoPod response reached "
                "the output token limit | "
                "max_tokens=%s",
                self.max_tokens,
            )

        usage = data.get(
            "usage",
            {},
        )

        logger.info(
            "SumoPod generation completed | "
            "input_tokens=%s | "
            "output_tokens=%s | "
            "total_tokens=%s | "
            "finish_reason=%s",
            usage.get("prompt_tokens"),
            usage.get("completion_tokens"),
            usage.get("total_tokens"),
            finish_reason,
        )

        return content.strip()