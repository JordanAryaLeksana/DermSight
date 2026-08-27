import os
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]


class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-only-change-this-secret")
    MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "8"))
    MAX_CONTENT_LENGTH = MAX_UPLOAD_MB * 1024 * 1024
    MODEL_PATH = os.getenv("SKIN_MODEL_PATH", str(ROOT_DIR / "src/outputs/final_model.weights.h5"))
    CLASS_NAMES_PATH = os.getenv("CLASS_NAMES_PATH", str(ROOT_DIR / "src/outputs/class_names.json"))
    MODEL_CONFIG_PATH = os.getenv("MODEL_CONFIG_PATH", str(ROOT_DIR / "src/outputs/config.json"))
    DISEASE_LIST_PATH = os.getenv(
        "DISEASE_LIST_PATH", str(ROOT_DIR / "llm/data/skin_knowledge_serving.csv")
    )
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4:e2b")
    LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "60"))
    ASPIRATION_EMAIL_TO = os.getenv("ASPIRATION_EMAIL_TO", "")
    ASPIRATION_WHATSAPP_URL = os.getenv("ASPIRATION_WHATSAPP_URL", "")
    SMTP_HOST = os.getenv("SMTP_HOST", "")
    SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
    SMTP_USERNAME = os.getenv("SMTP_USERNAME", "")
    SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
    SMTP_FROM = os.getenv("SMTP_FROM", "")
    SMTP_USE_TLS = os.getenv("SMTP_USE_TLS", "true").lower() in {"1", "true", "yes"}
    SMTP_TIMEOUT = int(os.getenv("SMTP_TIMEOUT", "15"))
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"
    SESSION_COOKIE_SECURE = os.getenv("SESSION_COOKIE_SECURE", "false").lower() in {"1", "true", "yes"}
    TRUST_PROXY_HEADERS = os.getenv("TRUST_PROXY_HEADERS", "false").lower() in {"1", "true", "yes"}
    ANALYSIS_TOKEN_MAX_AGE = int(os.getenv("ANALYSIS_TOKEN_MAX_AGE", "1800"))
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
