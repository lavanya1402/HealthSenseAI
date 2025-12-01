"""
guards.py
Safety and guardrails for HealthSenseAI.

This project is strictly for PUBLIC HEALTH AWARENESS and EDUCATION.
It must NOT:
- give diagnosis
- prescribe medicines or doses
- replace professional medical advice
"""

from dataclasses import dataclass
from enum import Enum
from typing import List


class RiskLevel(str, Enum):
    GENERAL = "general"
    SENSITIVE = "sensitive"
    EMERGENCY = "emergency"


EMERGENCY_KEYWORDS: List[str] = [
    "chest pain",
    "severe chest pain",
    "difficulty breathing",
    "cannot breathe",
    "shortness of breath",
    "unconscious",
    "fainted",
    "stroke",
    "heart attack",
    "suicidal",
    "suicide",
]

SENSITIVE_KEYWORDS: List[str] = [
    "cancer",
    "tumor",
    "pregnant",
    "pregnancy",
    "miscarriage",
    "abortion",
    "mental health",
    "depression",
    "anxiety",
    "self harm",
]


@dataclass
class GuardrailResult:
    risk_level: RiskLevel
    message_prefix: str
    must_include_disclaimer: bool = True


def classify_query(text: str) -> GuardrailResult:
    lowered = text.lower()

    if any(keyword in lowered for keyword in EMERGENCY_KEYWORDS):
        return GuardrailResult(
            risk_level=RiskLevel.EMERGENCY,
            message_prefix=(
                "⚠️ This may represent a potential emergency.\n"
                "Please contact your local emergency services or visit the nearest "
                "hospital / emergency room immediately.\n\n"
            ),
        )

    if any(keyword in lowered for keyword in SENSITIVE_KEYWORDS):
        return GuardrailResult(
            risk_level=RiskLevel.SENSITIVE,
            message_prefix=(
                "⚠️ This is a sensitive health topic. I can share general, public "
                "health information, but this is **not a diagnosis**.\n\n"
            ),
        )

    return GuardrailResult(
        risk_level=RiskLevel.GENERAL,
        message_prefix="",
    )


STANDARD_DISCLAIMER = (
    "\n\n---\n"
    "**Important:** I am an AI assistant for *public health awareness only*.\n"
    "- I do **not** provide diagnosis or treatment.\n"
    "- I cannot prescribe medicines or doses.\n"
    "- I may not reflect the latest local medical guidelines.\n"
    "For any persistent, severe, or unclear symptoms, please consult a registered "
    "medical professional or your local health authority."
)


def apply_guardrails(user_query: str, model_answer: str) -> str:
    guard_result = classify_query(user_query)
    final_text = ""

    if guard_result.message_prefix:
        final_text += guard_result.message_prefix

    final_text += model_answer

    if guard_result.must_include_disclaimer:
        final_text += STANDARD_DISCLAIMER

    return final_text
