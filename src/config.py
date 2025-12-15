from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

load_dotenv()


@dataclass(frozen=True)
class Settings:
    # ---- Paths (match your repo: src/data/raw) ----
    base_dir: Path
    data_dir: Path
    data_raw_dir: Path
    index_dir: Path

    # ---- Models ----
    llm_model_name: str
    embedding_model: str

    # ---- RAG knobs ----
    top_k: int = 6
    chunk_size: int = 900
    chunk_overlap: int = 150
    clear_score_threshold: float = 1.8
    partial_score_threshold: float = 2.6

    @staticmethod
    def from_env() -> "Settings":
        base_dir = Path(__file__).resolve().parent  # src/
        data_dir = base_dir / "data"
        raw_dir = data_dir / "raw"
        index_dir = data_dir / "index"

        # ✅ Groq model updated (llama-3.3-70b-versatile is supported)
        llm_model_name = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

        embedding_model = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

        return Settings(
            base_dir=base_dir,
            data_dir=data_dir,
            data_raw_dir=raw_dir,
            index_dir=index_dir,
            llm_model_name=llm_model_name,
            embedding_model=embedding_model,
            top_k=int(os.getenv("TOP_K", "6")),
            chunk_size=int(os.getenv("CHUNK_SIZE", "900")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "150")),
            clear_score_threshold=float(os.getenv("CLEAR_SCORE_THRESHOLD", "1.8")),
            partial_score_threshold=float(os.getenv("PARTIAL_SCORE_THRESHOLD", "2.6")),
        )


def get_llm(settings: Settings) -> Groq:
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Missing GROQ_API_KEY in environment/.env")
    return Groq(api_key=api_key)
