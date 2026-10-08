import os
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]


class Config:
    # Flask
    SECRET_KEY = os.getenv(
        "SECRET_KEY",
    )

    if not SECRET_KEY:
        raise RuntimeError(
            "SECRET_KEY is not configured"
        )

    RATELIMIT_STORAGE_URI = os.getenv(
    "RATELIMIT_STORAGE_URI",
    "memory://",
    )

    RATELIMIT_HEADERS_ENABLED = True
    # Upload
    MAX_UPLOAD_MB = int(
        os.getenv("MAX_UPLOAD_MB", "8")
    )
    CACHE_TYPE = os.getenv(
    "CACHE_TYPE",
    "RedisCache",
    )

    CACHE_REDIS_URL = os.getenv(
    "CACHE_REDIS_URL",
    "redis://127.0.0.1:6379/0",
    )

    CACHE_DEFAULT_TIMEOUT = int(
    os.getenv(
        "CACHE_DEFAULT_TIMEOUT",
        "1800",
        )
    )

    ANALYSIS_CACHE_TTL = int(
    os.getenv(
        "ANALYSIS_CACHE_TTL",
        "1800",
    )
    )

    RATELIMIT_STORAGE_URI = os.getenv(
    "RATELIMIT_STORAGE_URI",
    "redis://127.0.0.1:6379/1",
    )

    RATELIMIT_HEADERS_ENABLED = True
    MAX_CONTENT_LENGTH = (
        MAX_UPLOAD_MB * 1024 * 1024
    )

    # ML model
    MODEL_PATH = os.getenv(
        "SKIN_MODEL_PATH",
        str(
            ROOT_DIR
            / "src/outputs/final_model.weights.h5"
        ),
    )

    CLASS_NAMES_PATH = os.getenv(
        "CLASS_NAMES_PATH",
        str(
            ROOT_DIR
            / "src/outputs/class_names.json"
        ),
    )

    MODEL_CONFIG_PATH = os.getenv(
        "MODEL_CONFIG_PATH",
        str(
            ROOT_DIR
            / "src/outputs/config.json"
        ),
    )

    # RAG knowledge base
    DISEASE_LIST_PATH = os.getenv(
        "DISEASE_LIST_PATH",
        str(
            ROOT_DIR
            / "llm/data/skin_knowledge_serving.csv"
        ),
    )

    # LLM
    LLM_TIMEOUT = int(
        os.getenv("LLM_TIMEOUT", "60")
    )

    # Aspiration / email
    ASPIRATION_EMAIL_TO = os.getenv(
        "ASPIRATION_EMAIL_TO",
        "",
    )

    ASPIRATION_WHATSAPP_URL = os.getenv(
        "ASPIRATION_WHATSAPP_URL",
        "",
    )

    SMTP_HOST = os.getenv(
        "SMTP_HOST",
        "",
    )

    SMTP_PORT = int(
        os.getenv("SMTP_PORT", "587")
    )

    SMTP_USERNAME = os.getenv(
        "SMTP_USERNAME",
        "",
    )

    SMTP_PASSWORD = os.getenv(
        "SMTP_PASSWORD",
        "",
    )

    SMTP_FROM = os.getenv(
        "SMTP_FROM",
        "",
    )

    SMTP_USE_TLS = (
        os.getenv(
            "SMTP_USE_TLS",
            "true",
        ).lower()
        in {"1", "true", "yes"}
    )

    SMTP_TIMEOUT = int(
        os.getenv("SMTP_TIMEOUT", "15")
    )

    # Session / security
    SESSION_COOKIE_HTTPONLY = True

    SESSION_COOKIE_SAMESITE = "Lax"

    SESSION_COOKIE_SECURE = (
        os.getenv(
            "SESSION_COOKIE_SECURE",
            "false",
        ).lower()
        in {"1", "true", "yes"}
    )

    TRUST_PROXY_HEADERS = (
        os.getenv(
            "TRUST_PROXY_HEADERS",
            "false",
        ).lower()
        in {"1", "true", "yes"}
    )

    ANALYSIS_TOKEN_MAX_AGE = int(
        os.getenv(
            "ANALYSIS_TOKEN_MAX_AGE",
            "1800",
        )
    )

    # Logging
    LOG_LEVEL = os.getenv(
        "LOG_LEVEL",
        "INFO",
    )