from __future__ import annotations

import re
from pathlib import Path

import streamlit as st

from config import Settings, get_llm
from rag_pipeline import HealthSenseRAG
from utils import LanguageCode

STRICT_FALLBACK = "The guideline does not provide information on this topic."


# ---------------------- Language detection ----------------------
def detect_language_code(text: str) -> LanguageCode:
    t = text or ""
    if re.search(r"[\u0980-\u09FF]", t):  # Bengali
        return "bn"
    if re.search(r"[\u0A80-\u0AFF]", t):  # Gujarati
        return "gu"
    if re.search(r"[\u0B80-\u0BFF]", t):  # Tamil
        return "ta"
    if re.search(r"[\u0C00-\u0C7F]", t):  # Telugu
        return "te"
    if re.search(r"[\u0900-\u097F]", t):  # Hindi / Devanagari
        return "hi"
    return "en"


def coverage_badge(label: str) -> str:
    if label == "CLEAR":
        return "🟢 Guideline coverage: Clear"
    if label == "PARTIAL":
        return "🟡 Guideline coverage: Partial"
    return "🔴 Guideline coverage: Not covered"


def init_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "prefill_question" not in st.session_state:
        st.session_state.prefill_question = ""


# ---------------------- ONLY WORKING EXAMPLES ----------------------
EXAMPLE_QUESTIONS = [
    # EN
    "After pregnancy, how should women with a history of gestational diabetes be followed up?",
    # HI
    "हाई ब्लड प्रेशर रोकने के लिए कौन-से जीवनशैली बदलाव सुझाए गए हैं?",
    # MR
    "वयस्क व्यक्तीत कोणत्या रक्तदाब पातळीवर उच्च रक्तदाब (हायपरटेंशन) निदान केले जाते?",
    # GU
    "માર્ગદર્શિકા મુજબ પુખ્ત લોકોએ બ્લડ પ્રેશર ચેક કેટલા સમયાંતરે કરાવવું જોઈએ?",
    # TA
    "உயர் இரத்த அழுத்தம் உள்ளவர்களுக்கு உடனடி மருத்துவ உதவி தேவைப்படுகிறதா என்பதை காட்டும் எத்தகைய எச்சரிக்கை அறிகுறிகள் உள்ளன?",
    # TE
    "మార్గదర్శక ప్రకారం పెద్దవారు ఎంత కాల వ్యవధిలో ఒకసారి రక్తపోటు చెక్ చేయించుకోవాలి?",
    # BN
    "গাইডলাইন অনুযায়ী কোন কোন নারীকে ডায়াবেটিস হওয়ার উচ্চ ঝুঁকিপূর্ণ হিসেবে ধরা হয়?",
]


def main() -> None:
    st.set_page_config(page_title="HealthSenseAI", page_icon="🩺", layout="wide")
    init_session_state()

    # ---------------------- Styling ----------------------
    st.markdown(
        """
        <style>
        .hs-title { font-size: 2.7rem; font-weight: 900; margin-bottom: 0.2rem; }
        .hs-sub { font-size: 1.05rem; color: #6b7280; margin-bottom: 0.6rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ---------------------- Header ----------------------
    st.markdown('<div class="hs-title">HealthSenseAI</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hs-sub">A guideline-grounded assistant for public health awareness & early risk guidance. Educational use only — not a diagnostic tool.</div>',
        unsafe_allow_html=True,
    )

    # 🌸 BIG COLOURED NAME (GUARANTEED VISIBLE)
    st.markdown(
        """
        <div style="
            font-size: 1.3rem;
            font-weight: 900;
            margin-bottom: 1.4rem;
        ">
            Built by 
            <span style="
                background: linear-gradient(90deg, #2563eb, #9333ea);
                color: white;
                padding: 6px 14px;
                border-radius: 999px;
                font-size: 1.25rem;
                font-weight: 900;
                box-shadow: 0 4px 12px rgba(0,0,0,0.25);
            ">
                🌸 Lavanya Srivastava
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---------------------- Settings + Engine ----------------------
    settings = Settings.from_env()
    llm = get_llm(settings)
    rag_engine = HealthSenseRAG(settings=settings, llm=llm, language="en")

    # ---------------------- Sidebar ----------------------
    with st.sidebar:
        st.header("Settings")
        language_mode = st.selectbox(
            "Response language",
            ["auto", "en", "hi", "bn", "te", "ta", "gu"],
            index=0,
        )

        st.divider()
        st.subheader("✅ Example Questions (working)")
        st.caption("Click to auto-fill the question.")

        for q in EXAMPLE_QUESTIONS:
            if st.button(q, use_container_width=True):
                st.session_state.prefill_question = q

        st.divider()
        st.subheader("Debug")
        show_pairs = st.checkbox("Show retrieved excerpts", value=False)

        st.divider()
        st.subheader("Knowledge Base")
        st.caption("Upload WHO / CDC / MoHFW guideline PDFs (text-based preferred).")

        existing_pdfs = sorted(Path(settings.data_raw_dir).glob("*.pdf"))
        st.caption(f"📄 PDFs in KB: **{len(existing_pdfs)}**")

        uploaded = st.file_uploader(
            "Upload health guideline PDFs",
            type=["pdf"],
            accept_multiple_files=True,
        )

        st.divider()
        st.info(
            "Important:\n"
            "- No diagnosis / no treatment plans.\n"
            "- STRICT RAG: answers ONLY from retrieved guideline excerpts.\n"
            f"- If not covered: returns '{STRICT_FALLBACK}'.\n"
        )

    # ---------------------- Save PDFs + rebuild index ----------------------
    if uploaded:
        settings.data_raw_dir.mkdir(parents=True, exist_ok=True)
        for f in uploaded:
            (settings.data_raw_dir / f.name).write_bytes(f.read())

        st.sidebar.success(f"Uploaded {len(uploaded)} file(s) ✅")
        rag_engine.force_rebuild()
        st.sidebar.info("Index rebuilt ✅")

    st.markdown("### Ask a health awareness question")

    # ---------------------- Chat history ----------------------
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ---------------------- Chat input (with auto-fill) ----------------------
    user_input = st.chat_input(
        "Type a question about prevention, lifestyle, screenings..."
    )

    if st.session_state.prefill_question and not user_input:
        user_input = st.session_state.prefill_question
        st.session_state.prefill_question = ""

    if user_input:
        lang = detect_language_code(user_input) if language_mode == "auto" else language_mode
        rag_engine.language = lang  # type: ignore

        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Checking guideline excerpts..."):
                response, coverage, pairs = rag_engine.answer_query(user_input)

            st.markdown(f"**{coverage_badge(coverage)}**")
            st.markdown(response)

            if show_pairs and pairs:
                st.divider()
                st.caption("Retrieved chunks (score: lower is better):")
                for d, s in pairs:
                    src = d.metadata.get("source", "Guideline")
                    page = d.metadata.get("page", "Unknown")
                    st.write(f"- {src} — page {page} | score: {float(s):.4f}")

            st.session_state.messages.append({"role": "assistant", "content": response})

    st.markdown("---")
    st.caption(
        "Developed as an educational project demonstrating Generative AI + RAG for public health awareness. "
        "Always consult licensed medical professionals for personal medical concerns."
    )


if __name__ == "__main__":
    main()
