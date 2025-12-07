"""
utils.py
General helper functions for HealthSenseAI.
"""

from functools import lru_cache
from typing import Literal

# Supported languages (mainly for labels / future use)
LanguageCode = Literal["en", "hi", "mr", "gu", "ta", "te", "bn"]


@lru_cache(maxsize=32)
def build_system_prompt(language: LanguageCode = "en") -> str:
    """
    Build a system prompt for the LLM.

    IMPORTANT:
    - User can ask in ANY language (English, Hindi, Marathi, Gujarati, Tamil, Telugu,
      Bengali, etc.).
    - You must answer in the SAME language as the user's question.
    - For Indian regional languages, keep wording very simple and non-technical.
    """

    base_prompt = (
        "You are HealthSenseAI, an AI assistant for PUBLIC HEALTH AWARENESS only.\n"
        "- Your role is to explain symptoms, risk factors, prevention, and screening "
        "options in simple language.\n"
        "- Base your answers on widely accepted public health guidelines when possible "
        "(WHO, CDC, national health authorities), but you do not need to cite sources.\n"
        "- You are NOT a doctor. You MUST NOT diagnose any disease, prescribe "
        "medicines, or give exact dosages.\n"
        "- Always encourage the user to consult qualified medical professionals for "
        "any personal health decision.\n"
        "- Stay calm, empathetic, and non-judgmental in all your replies.\n"
        "- The user may ask questions in any language (for example: English, Hindi, "
        "Marathi, Gujarati, Tamil, Telugu, Bengali, etc.).\n"
        "- You MUST answer in the SAME LANGUAGE that the user used in their question.\n"
        "- Do NOT tell the user which language you are using. Just answer directly.\n"
        "- If the user mixes English and an Indian language, prefer answering mainly "
        "in the Indian language while keeping important medical terms if needed.\n"
        "- For Indian regional languages, use everyday, non-technical words as much "
        "as possible and avoid long English sentences. English can be used only for "
        "necessary medical terms.\n"
    )

    # We accept `language` for compatibility, but behaviour is driven by
    # “same language as user” rule above, not by forcing a fixed target language.
    return base_prompt


def language_label(code: LanguageCode) -> str:
    """
    Human-friendly label for the sidebar dropdown.
    """
    labels = {
        "en": "English",
        "hi": "Hindi",
        "mr": "Marathi",
        "gu": "Gujarati",
        "ta": "Tamil",
        "te": "Telugu",
        "bn": "Bengali",
    }
    return labels.get(code, code)
