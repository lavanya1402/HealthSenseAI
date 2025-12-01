"""
config.py
Central configuration for HealthSenseAI (Groq version).

- Loads environment variables
- Exposes Settings dataclass
- Creates a LangChain-compatible LLM (ChatGroq)
"""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.language_models import BaseChatModel

load_dotenv()


@dataclass
class Settings:
    app_env: str
    llm_provider: str
    llm_model_name: str
    groq_api_key: str | None
    data_raw_dir: Path
    index_dir: Path
    embedding_model_name: str

    @classmethod
    def from_env(cls) -> "Settings":
        app_env = os.getenv("APP_ENV", "dev")

        # Default to Groq
        provider = os.getenv("LLM_PROVIDER", "groq").lower()

        if provider == "groq":
            model_name = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
            groq_api_key = os.getenv("GROQ_API_KEY")
            if not groq_api_key:
                raise ValueError("GROQ_API_KEY is missing in .env")
        else:
            raise ValueError(
                f"Unsupported LLM_PROVIDER '{provider}'. "
                "For now we support only 'groq'."
            )

        data_raw_dir = Path(os.getenv("DATA_RAW_DIR", "data/raw"))
        index_dir = Path(os.getenv("INDEX_DIR", "data/processed/faiss_index"))
        embedding_model_name = os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )

        # Ensure dirs exist
        data_raw_dir.mkdir(parents=True, exist_ok=True)
        index_dir.mkdir(parents=True, exist_ok=True)

        return cls(
            app_env=app_env,
            llm_provider=provider,
            llm_model_name=model_name,
            groq_api_key=groq_api_key,
            data_raw_dir=data_raw_dir,
            index_dir=index_dir,
            embedding_model_name=embedding_model_name,
        )


def get_llm(settings: Settings) -> BaseChatModel:
    """
    Return a LangChain-compatible chat model.
    (Currently: Groq ChatGroq)
    """
    if settings.llm_provider == "groq":
        return ChatGroq(
            groq_api_key=settings.groq_api_key,
            model_name=settings.llm_model_name,
            temperature=0.2,
        )

    raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")
