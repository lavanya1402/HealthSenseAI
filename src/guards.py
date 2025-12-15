from __future__ import annotations

from typing import Tuple

# Very light safety filter; keep it simple (STRICT RAG already prevents hallucination)
def apply_guardrails(user_query: str, raw_answer: str) -> Tuple[str, str]:
    text = (raw_answer or "").strip()
    if not text:
        return "", "empty"
    return text, "ok"
