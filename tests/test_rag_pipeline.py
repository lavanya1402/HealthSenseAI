"""
Basic pytest to check that the RAG pipeline can be instantiated.

Run:
    pytest tests/test_rag_pipeline.py
"""

import pytest

from src.config import Settings, get_llm
from src.rag_pipeline import HealthSenseRAG


def test_settings_from_env():
    settings = Settings.from_env()
    assert settings.data_raw_dir.exists()
    assert settings.llm_provider == "groq"


@pytest.mark.skip(
    reason="Requires PDFs in data/raw and a valid GROQ_API_KEY. "
    "Unskip when environment is ready."
)
def test_rag_engine_builds_index_and_answers():
    settings = Settings.from_env()
    llm = get_llm(settings)
    rag = HealthSenseRAG(settings=settings, llm=llm, language="en")

    rag.build_or_load_index()
    answer = rag.answer_query("What are common prevention methods for dengue?")
    assert isinstance(answer, str)
    assert len(answer) > 0
