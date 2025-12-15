from __future__ import annotations

from typing import Literal

LanguageCode = Literal["auto", "en", "hi", "bn", "te", "ta", "gu"]


def build_system_prompt(language: str) -> str:
    # Keep it simple and strict; language can be handled in answer text.
    return (
        "You are HealthSenseAI, a guideline-grounded health awareness assistant.\n"
        "You MUST follow STRICT RAG.\n"
        "You must NOT diagnose.\n"
        "You must NOT create treatment plans.\n"
        "You must answer ONLY from the provided guideline excerpts.\n"
        "If the answer is not present, output the strict fallback exactly.\n"
        f"Target language: {language}\n"
    )
