from __future__ import annotations

import re
from dataclasses import dataclass

STRICT_FALLBACK = "The guideline does not provide information on this topic."

@dataclass
class EvidenceCheck:
    ok: bool
    reason: str = ""

def normalize_text(s: str) -> str:
    if not s:
        return ""
    s = s.replace("\u00A0", " ")
    s = s.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def extract_section(text: str, header: str) -> str:
    if not text:
        return ""
    pat = re.compile(rf"{re.escape(header)}\s*(.*)", re.IGNORECASE | re.DOTALL)
    m = pat.search(text)
    if not m:
        return ""
    after = m.group(1)

    # Stop at next known header
    stop_headers = ["Sources", "Source", "Direct Answer", "Notes", "Disclaimer"]
    for h in stop_headers:
        hit = re.search(rf"\n\s*{re.escape(h)}\s*:", after, flags=re.IGNORECASE)
        if hit:
            after = after[: hit.start()]
            break

    return after.strip()

def contains_inference_language(evidence: str) -> bool:
    if not evidence:
        return True
    bad_phrases = [
        "अनुमान", "अंदाजा", "यह अनुमान लगाया जा सकता है", "यह माना जा सकता है",
        "likely", "probably", "can be inferred", "suggests that", "it seems",
        "we can assume", "no direct reference",
    ]
    ev_low = evidence.lower()
    return any(p.lower() in ev_low for p in bad_phrases)

def evidence_is_verbatim_in_excerpts(evidence: str, excerpts_blob: str) -> bool:
    ev = normalize_text(evidence)
    ex = normalize_text(excerpts_blob)
    if not ev or not ex:
        return False

    # Extract quoted lines from evidence: we expect blockquotes ">"
    lines = []
    for ln in ev.splitlines():
        ln = ln.strip()
        if ln.startswith(">"):
            lines.append(normalize_text(ln.lstrip(">").strip()))

    # Fallback: if no ">" lines, try split
    if not lines:
        lines = [normalize_text(x) for x in re.split(r"\n|- ", ev) if normalize_text(x)]

    # Require at least 1 strong match
    for ln in lines:
        if len(ln) >= 12 and ln in ex:
            return True

    return False

def validate_answer_against_evidence(raw_answer: str, excerpts_blob: str) -> EvidenceCheck:
    """
    Tuned STRICT RAG validator.

    Rules:
    - If answer == STRICT_FALLBACK → OK
    - Must contain 'Guideline Evidence:' section
    - Evidence must come from excerpts (verbatim)
    - Allows numeric/definition answers, but still requires evidence match
    - Blocks inference language
    """
    if not raw_answer:
        return EvidenceCheck(False, "empty_answer")

    ans = normalize_text(raw_answer)

    if ans.strip() == STRICT_FALLBACK:
        return EvidenceCheck(True, "fallback_ok")

    evidence = normalize_text(extract_section(raw_answer, "Guideline Evidence:"))
    if not evidence:
        return EvidenceCheck(False, "missing_evidence_section")

    if STRICT_FALLBACK.lower() in evidence.lower():
        return EvidenceCheck(False, "evidence_is_fallback")

    if contains_inference_language(evidence):
        return EvidenceCheck(False, "inference_detected")

    # Numeric/definition answers allowed, but evidence must still match excerpts
    if not evidence_is_verbatim_in_excerpts(evidence, excerpts_blob):
        return EvidenceCheck(False, "evidence_not_verbatim")

    return EvidenceCheck(True, "evidence_ok")
