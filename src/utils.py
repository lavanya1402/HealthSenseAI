"""
utils.py
General helper functions for HealthSenseAI.
"""

from functools import lru_cache
from typing import Literal


LanguageCode = Literal["en", "hi", "mr"]  # English, Hindi, Marathi (extend as needed)


@lru_cache(maxsize=32)
def build_system_prompt(language: LanguageCode = "en") -> str:
    """
    Build a system prompt for the LLM.
    """

    base_prompt_en = (
        "You are HealthSenseAI, an AI assistant for PUBLIC HEALTH AWARENESS.\n"
        "- Your job is to explain symptoms, risk factors, prevention, and screening "
        "in simple language based on trusted public health guidelines "
        "(such as WHO, CDC, and national health authorities).\n"
        "- You are NOT a doctor. You MUST NOT provide diagnosis, prescriptions, or "
        "exact doses.\n"
        "- Encourage users to consult licensed health professionals.\n"
        "- If the user asks for medicines, exact treatment, or diagnosis, gently "
        "refuse and redirect them to a doctor.\n"
        "- Always stay calm, supportive, and non-judgmental.\n"
    )

    if language == "hi":
        base_prompt_en += (
            "\nYou are answering in **simple Hindi**. "
            "Use easy words that a non-technical person in India can understand."
        )
    elif language == "mr":
        base_prompt_en += (
            "\nYou are answering in **simple Marathi**. "
            "Use easy words that a non-technical person can understand."
        )
    else:
        base_prompt_en += "\nYou are answering in **simple international English**."

    return base_prompt_en


def language_label(language: LanguageCode) -> str:
    if language == "en":
        return "English"
    if language == "hi":
        return "Hindi"
    if language == "mr":
        return "Marathi"
    return "English"
