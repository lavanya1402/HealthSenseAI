"""
app.py
Streamlit UI for HealthSenseAI (Groq + RAG).

Run with:
    streamlit run src/app.py
"""

import streamlit as st

from config import Settings, get_llm
from rag_pipeline import HealthSenseRAG


def init_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []


def main() -> None:
    # -------------------- Page configuration --------------------
    st.set_page_config(
        page_title="HealthSenseAI – Public Health Awareness Assistant",
        page_icon="🩺",
        layout="wide",
    )

    init_session_state()

    # -------------------- Header styling --------------------
    st.markdown(
        """
        <style>
        .main-title {
            font-size: 2.2rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }
        .sub-title {
            font-size: 0.95rem;
            color: #6c757d;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="main-title">HealthSenseAI</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">'
        'A Generative AI Assistant for <strong>Public Health Awareness & Early Risk Guidance</strong>. '
        'Educational use only – not a diagnostic tool.'
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("")

    # Load settings once (shared by sidebar and main area)
    settings = Settings.from_env()

    # -------------------- Sidebar --------------------
    with st.sidebar:
        st.header("Settings")
        st.write(
            "You can type your question in any language. "
            "The assistant will reply in the same language."
        )

        st.markdown("---")
        st.subheader("Knowledge Base")

        uploaded_pdfs = st.file_uploader(
            "Upload health guideline PDFs (WHO / MoHFW / national):",
            type=["pdf"],
            accept_multiple_files=True,
            help="Files will be stored in the project’s data/raw directory.",
        )

        if uploaded_pdfs:
            saved_files = []
            for up_file in uploaded_pdfs:
                save_path = settings.data_raw_dir / up_file.name
                with open(save_path, "wb") as f:
                    f.write(up_file.getbuffer())
                saved_files.append(up_file.name)

            st.success(
                f"Uploaded {len(saved_files)} PDF file(s). "
                "Restart the app so that they are included in the index."
            )

        pdf_count = len(list(settings.data_raw_dir.glob("*.pdf")))
        st.caption(
            f"Guideline PDF files currently detected in data/raw/: **{pdf_count}**"
        )

        st.markdown(
            "- Place WHO / national guideline PDFs in `data/raw/`.\n"
            "- The app automatically (re)builds a FAISS index from these files at startup."
        )

        st.markdown("---")
        st.subheader("Disclaimer")
        st.info(
            "This assistant is designed for public health education only.\n\n"
            "- It does not provide diagnosis or treatment.\n"
            "- It does not prescribe medication or dosages.\n"
            "- It should not replace consultation with qualified healthcare professionals."
        )

    # -------------------- RAG Engine --------------------
    llm = get_llm(settings)
    rag_engine = HealthSenseRAG(settings=settings, llm=llm)

    # Build index ONCE at app start
    rag_engine.build_or_load_index()

    # -------------------- Main Chat Area --------------------
    st.markdown("### Ask a health awareness question")

    with st.expander("Examples", expanded=False):
        st.markdown(
            """
**English**
- What are early warning signs of diabetes according to these guidelines?
- How can adults reduce their risk of hypertension through lifestyle changes?

**Hindi**
- शुगर के शुरुआती लक्षण क्या हो सकते हैं?
- उच्च रक्तचाप का जोखिम कम करने के लिए कौन-से उपाय बताए गए हैं?

**Marathi**
- मधुमेहाची प्रारंभिक लक्षणे कोणती आहेत?
- हाय ब्लड प्रेशर कमी करण्यासाठी जीवनशैलीत कोणते बदल करायला हवेत?

**Gujarati**
- ડાયાબિટીસના પ્રારંભિક લક્ષણો શું હોઈ શકે?
- ઊંચા બ્લડ પ્રેશરનું જોખમ ઓછું કરવા માટે જીવનશૈલીમાં શું ફેરફાર કરવાની સલાહ છે?
"""
        )

    # Display conversation history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    user_input = st.chat_input(
        "Type a question about symptoms, prevention, lifestyle, or screenings..."
    )

    if user_input:
        # Store and display user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Consulting public health guidelines..."):
                try:
                    response = rag_engine.answer_query(user_input)
                except Exception as e:
                    st.error(f"An error occurred while generating the answer: {e}")
                    return

                st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})

    st.markdown(
        "---\n"
        "Developed as an educational project demonstrating Generative AI + Retrieval-Augmented Generation (RAG) "
        "for public health awareness. Always seek advice from licensed medical professionals for any personal health concerns."
    )


if __name__ == "__main__":
    main()
