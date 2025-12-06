"""
config.py
Central configuration for HealthSenseAI (Groq + RAG)
"""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# Optional: Streamlit is only available on Streamlit Cloud / when running the app
try:
    import streamlit as st  # type: ignore
except Exception:
    st = None

# Detect project root: .../HealthSenseAI
PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Settings:
    app_env: str
    llm_provider: str
    llm_model_name: str
    groq_api_key: str
    data_raw_dir: Path
    index_dir: Path
    embedding_model_name: str

    @classmethod
    def from_env(cls) -> "Settings":
        app_env = os.getenv("APP_ENV", "dev")

        provider = os.getenv("LLM_PROVIDER", "groq").lower()
        if provider != "groq":
            raise ValueError("Only 'groq' provider is supported at the moment.")

        model_name = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

        # 1️⃣ Try normal environment / .env (local dev)
        groq_api_key = os.getenv("GROQ_API_KEY")

        # 2️⃣ Fallback to Streamlit Cloud secrets
        if not groq_api_key and st is not None:
            groq_api_key = st.secrets.get("GROQ_API_KEY")

        if not groq_api_key:
            raise ValueError(
                "GROQ_API_KEY is missing. Set it in a local .env file "
                "or in Streamlit Cloud Secrets."
            )

        # These now always point to HealthSenseAI/data/... by default
        data_raw_dir = Path(
            os.getenv("DATA_RAW_DIR", str(PROJECT_ROOT / "data" / "raw"))
        )
        index_dir = Path(
            os.getenv(
                "INDEX_DIR",
                str(PROJECT_ROOT / "data" / "processed" / "faiss_index"),
            )
        )

        embedding_model_name = os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )

        # Ensure directories exist
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


def get_llm(settings: Settings) -> Groq:
    """Return a Groq client instance."""
    return Groq(api_key=settings.groq_api_key)
