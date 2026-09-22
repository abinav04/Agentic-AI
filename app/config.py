import os
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

# Pydantic BaseSettings class loading configuration parameters from environment variables and .env file.
# Configures API keys, LLM models, vector store paths, LangSmith tracing, and server ports.
class Settings(BaseSettings):
    # LLM Keys
    GROQ_API_KEY: Optional[str] = None
    GEMINI_API_KEY: Optional[str] = None

    # LLM Models (Verified active free-tier models)
    GROQ_MODEL: str = "groq/compound-mini"
    GEMINI_MODEL: str = "gemini-3.6-flash"

    # Provider & Fallback Settings
    LLM_PRIMARY_PROVIDER: str = "groq"
    LLM_FALLBACK_ENABLED: bool = True

    # Embedding & Vector Database
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    CHROMA_PERSIST_DIRECTORY: str = "./chroma_db"
    CORPUS_FILE_PATH: str = "corpus.jsonl"

    # LangSmith Settings
    LANGSMITH_API_KEY: Optional[str] = None
    LANGSMITH_PROJECT: str = "kestrel-research-assistant"
    LANGSMITH_TRACING_V2: bool = True

    # Server Settings
    PORT: int = 8000
    HOST: str = "0.0.0.0"
    FASTAPI_URL: str = "http://localhost:8000"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()

# Set environment variables for LangSmith if present
if settings.LANGSMITH_API_KEY:
    os.environ["LANGSMITH_API_KEY"] = settings.LANGSMITH_API_KEY
    os.environ["LANGSMITH_PROJECT"] = settings.LANGSMITH_PROJECT
    os.environ["LANGSMITH_TRACING"] = "true" if settings.LANGSMITH_TRACING_V2 else "false"
