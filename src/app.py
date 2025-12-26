from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List

import streamlit as st

from config import Settings, get_llm
from rag_pipeline import HealthSenseRAG

STRICT_FALLBACK = "The guideline does not provide information on this topic."


# ---------------------- Language detection ----------------------
def detect_language_code(text: str) -> str:
    t = (text or "").strip()
    if re.search(r"[\u0980-\u09FF]", t):  # Bengali
        return "bn"
    if re.search(r"[\u0A80-\u0AFF]", t):  # Gujarati
        return "gu"
    if re.search(r"[\u0B80-\u0BFF]", t):  # Tamil
        return "ta"
    if re.search(r"[\u0C00-\u0C7F]", t):  # Telugu
        return "te"
    if re.search(r"[\u0900-\u097F]", t):  # Hindi/Devanagari
        return "hi"
    return "en"


def coverage_badge(label: str) -> str:
    if label == "CLEAR":
        return "🟢 Guideline coverage: Clear"
    if label == "PARTIAL":
        return "🟡 Guideline coverage: Partial"
    return "🔴 Guideline coverage: Not covered"


def save_uploads(files, out_dir: Path) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: List[Path] = []
    for f in files or []:
        p = out_dir / f.name
        p.write_bytes(f.getbuffer())
        saved.append(p)
    return saved


# ---------------------- Streamlit UI ----------------------
st.set_page_config(page_title="HealthSenseAI (Guideline Q&A)", page_icon="🩺", layout="wide")
st.title("🩺 HealthSenseAI — Guideline Q&A (STRICT RAG)")
st.caption("Upload text-based guideline PDFs → Ask questions → Get excerpt-grounded answers with sources.")

# ---------------------- Init session state ----------------------
if "settings" not in st.session_state:
    st.session_state.settings = Settings.from_env()

if "llm" not in st.session_state:
    st.session_state.llm = get_llm(st.session_state.settings)

if "rag" not in st.session_state:
    st.session_state.rag = HealthSenseRAG(
        settings=st.session_state.settings,
        llm=st.session_state.llm,
        language="en",
    )

if "messages" not in st.session_state:
    st.session_state.messages: List[Dict[str, str]] = []

settings: Settings = st.session_state.settings
rag: HealthSenseRAG = st.session_state.rag

# ---------------------- Sidebar ----------------------
with st.sidebar:
    st.header("⚙️ Setup")

    st.caption("✅ Running with Groq (from .env)")
    st.code(f"GROQ_MODEL = {settings.llm_model_name}", language="text")

    st.divider()
    st.subheader("📄 Knowledge Base (Text PDFs only)")

    raw_dir = Path(settings.data_raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(raw_dir.glob("*.pdf"))
    st.caption(f"PDFs in KB: {len(existing)}")
    if existing:
        with st.expander("Show PDF names"):
            for p in existing:
                st.write(f"- {p.name}")

    uploads = st.file_uploader(
        "Upload 1 or more PDF guidelines",
        type=["pdf"],
        accept_multiple_files=True,
    )

    st.divider()
    col1, col2 = st.columns(2)
    build_btn = col1.button("🔨 Build / Rebuild Index", use_container_width=True)
    reset_btn = col2.button("♻️ Reset Chat", use_container_width=True)

    st.divider()
    st.info(
        "Important:\n"
        "- NO OCR: scanned/image PDFs won't work.\n"
        "- STRICT RAG: answers only from retrieved excerpts.\n"
        f"- Not covered → '{STRICT_FALLBACK}'."
    )

if reset_btn:
    st.session_state.messages = []
    st.toast("Chat reset ✅")

# ---------------------- Build index ----------------------
if build_btn:
    if not uploads and not existing:
        st.warning("Please upload at least one text-based PDF first.")
    else:
        if uploads:
            save_uploads(uploads, raw_dir)
            st.sidebar.success(f"Uploaded {len(uploads)} file(s) ✅")

        with st.spinner("Building/Loading index..."):
            try:
                rag.force_rebuild()
                st.success("Index rebuilt successfully ✅")
            except Exception as e:
                st.error(f"Index build failed: {e}")

st.divider()

# ---------------------- Render chat history ----------------------
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# ---------------------- Ask question ----------------------
q = st.chat_input("Ask a question from the uploaded guideline PDFs…")

if q:
    st.session_state.messages.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)

    user_lang = detect_language_code(q)

    with st.chat_message("assistant"):
        with st.spinner("Searching guideline excerpts..."):
            try:
                # Set language target for output formatting
                rag.language = user_lang  # type: ignore

                response, coverage, pairs = rag.answer_query(q)

                st.markdown(f"**{coverage_badge(coverage)}**")
                st.markdown(response)

                # Debug (optional quick view if you want later)
                # st.caption(f"Retrieved chunks: {len(pairs)}")

                st.session_state.messages.append({"role": "assistant", "content": response})

            except Exception as e:
                msg = f"Sorry — I hit an error while answering: {e}"
                st.error(msg)
                st.session_state.messages.append({"role": "assistant", "content": msg})
