"""
app.py
Streamlit UI for HealthSenseAI (Groq + RAG).

Run with:
    streamlit run src/app.py
"""

import streamlit as st

from .config import Settings, get_llm
from .rag_pipeline import HealthSenseRAG
from .utils import language_label, LanguageCode


def init_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []


def main() -> None:
    st.set_page_config(
        page_title="HealthSenseAI – Public Health Awareness Assistant",
        page_icon="🩺",
        layout="wide",
    )

    init_session_state()

    st.title("🩺 HealthSenseAI")
    st.caption(
        "A Generative AI Assistant for **Public Health Awareness & Early Risk Guidance**.\n"
        "Educational use only – **not** a diagnostic tool."
    )

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")

        language: LanguageCode = st.selectbox(
            "Response language",
            options=["en", "hi", "mr"],
            format_func=language_label,
            index=0,
        )

        st.markdown("---")
        st.markdown(
            "📂 **PDF Status**\n\n"
            "- Place WHO / CDC / MoHFW guideline PDFs in `data/raw/`.\n"
            "- The app will automatically build or load a FAISS index."
        )

        st.markdown("---")
        st.markdown(
            "⚠️ **Disclaimer**\n\n"
            "This assistant is for public health education only.\n"
            "It does not give diagnosis, prescriptions, or emergency advice."
        )

    # Cache RAG engine
    @st.cache_resource(show_spinner=True)
    def load_rag_engine(selected_language: LanguageCode) -> HealthSenseRAG:
        settings = Settings.from_env()
        llm = get_llm(settings)
        rag = HealthSenseRAG(settings=settings, llm=llm, language=selected_language)
        rag.build_or_load_index()
        return rag

    rag_engine = load_rag_engine(language)

    st.markdown("### 💬 Ask a health awareness question")

    # Display history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input(
        "Type your question about symptoms, prevention, lifestyle, or screenings..."
    )

    if user_input:
        # Show + store user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking with public health guidelines..."):
                try:
                    response = rag_engine.answer_query(user_input)
                except FileNotFoundError as e:
                    st.error(
                        "No PDFs found in `data/raw/`.\n\n"
                        "Please add WHO / CDC / MoHFW guideline PDFs and then rerun.\n\n"
                        f"Details: {e}"
                    )
                    return
                except Exception as e:
                    st.error(f"An error occurred while generating the answer: {e}")
                    return

                st.markdown(response)
                st.session_state.messages.append(
                    {"role": "assistant", "content": response}
                )

    st.markdown(
        "---\n"
        "Developed as an educational project on Generative AI + RAG for public health awareness.\n"
        "Always consult licensed medical professionals for any personal health concerns."
    )


if __name__ == "__main__":
    main()
