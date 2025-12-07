"""
rag_pipeline.py
RAG pipeline for HealthSenseAI (Groq + LangChain + FAISS).

If guideline PDFs contain readable text:
    -> Build a FAISS index and use RAG.
If PDFs are image-only / non-extractable:
    -> Skip FAISS and fall back to general public-health answers
       (still with strong guardrails).
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from groq import Groq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

from config import Settings
from guards import apply_guardrails
from utils import build_system_prompt, LanguageCode


class HealthSenseRAG:
    """
    A focused RAG engine that:

    - Attempts to build / load a FAISS vector index from guideline PDFs.
    - If that fails due to non-extractable text, continues in "no-RAG" mode.
    """

    def __init__(
        self,
        settings: Settings,
        llm: Groq,
        language: LanguageCode = "en",
        top_k: int = 4,
        chunk_size: int = 900,
        chunk_overlap: int = 150,
    ) -> None:
        self.settings = settings
        self.client = llm
        self.language = language
        self.top_k = top_k
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self._embedding = HuggingFaceEmbeddings(
            model_name=self.settings.embedding_model_name
        )

        self._vectorstore: FAISS | None = None
        self.rag_enabled: bool = False

    # ----------------- Loading + Indexing -----------------
    def _load_pdfs(self) -> List[Document]:
        data_dir: Path = self.settings.data_raw_dir
        print(f"[HealthSenseRAG] Looking for PDFs in: {data_dir.resolve()}")

        pdf_files = list(data_dir.glob("*.pdf"))
        if not pdf_files:
            raise FileNotFoundError(
                f"No PDFs found in {data_dir.resolve()}. "
                "Please add WHO/CDC/MoHFW guideline PDFs there."
            )

        docs: List[Document] = []
        for pdf_path in pdf_files:
            print(f"[HealthSenseRAG] Loading PDF: {pdf_path.name}")
            loader = PyPDFLoader(str(pdf_path))
            pdf_docs = loader.load()
            docs.extend(pdf_docs)

        print(f"[HealthSenseRAG] Loaded {len(docs)} document pages from PDFs.")
        return docs

    def _split_documents(self, docs: List[Document]) -> List[Document]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        chunked_docs = splitter.split_documents(docs)
        print(f"[HealthSenseRAG] Split into {len(chunked_docs)} chunks.")
        return chunked_docs

    def build_or_load_index(self) -> None:
        index_path: Path = self.settings.index_dir
        faiss_file = index_path / "index.faiss"

        # 1) Load existing index
        if faiss_file.exists():
            print(f"[HealthSenseRAG] Loading FAISS index from: {index_path.resolve()}")
            self._vectorstore = FAISS.load_local(
                str(index_path),
                self._embedding,
                allow_dangerous_deserialization=True,
            )
            self.rag_enabled = True
            return

        # 2) Otherwise build new index
        print("[HealthSenseRAG] Building new FAISS index from PDFs...")

        try:
            docs = self._load_pdfs()
        except FileNotFoundError as e:
            print(f"[HealthSenseRAG] No PDFs found: {e}")
            self._vectorstore = None
            self.rag_enabled = False
            return

        filtered_docs = [
            d for d in docs if d.page_content and d.page_content.strip()
        ]
        print(
            f"[HealthSenseRAG] Filtered pages with text: {len(filtered_docs)} "
            f"(original: {len(docs)})"
        )

        if not filtered_docs:
            print(
                "[HealthSenseRAG] PDFs found but no readable text extracted. "
                "Proceeding in non-RAG mode (no FAISS index)."
            )
            self._vectorstore = None
            self.rag_enabled = False
            return

        chunked_docs = self._split_documents(filtered_docs)
        if not chunked_docs:
            print(
                "[HealthSenseRAG] Splitting produced zero chunks. "
                "Proceeding in non-RAG mode."
            )
            self._vectorstore = None
            self.rag_enabled = False
            return

        vectorstore = FAISS.from_documents(chunked_docs, self._embedding)
        index_path.mkdir(parents=True, exist_ok=True)
        vectorstore.save_local(str(index_path))
        print(f"[HealthSenseRAG] Saved FAISS index to: {index_path.resolve()}")

        self._vectorstore = vectorstore
        self.rag_enabled = True

    # ----------------- Retrieval -----------------
    def retrieve_context(self, query: str) -> List[Document]:
        if not self.rag_enabled:
            print("[HealthSenseRAG] RAG disabled – skipping retrieval.")
            return []

        if self._vectorstore is None:
            self.build_or_load_index()

        if self._vectorstore is None:
            print("[HealthSenseRAG] Vectorstore unavailable – skipping retrieval.")
            return []

        docs = self._vectorstore.similarity_search(query, k=self.top_k)
        print(f"[HealthSenseRAG] Retrieved {len(docs)} chunks for query.")
        return docs

    @staticmethod
    def _format_context(docs: List[Document]) -> str:
        context_chunks = [d.page_content.strip() for d in docs if d.page_content]
        return "\n\n---\n\n".join(context_chunks)

    # ----------------- Answering -----------------
    def answer_query(self, user_query: str) -> str:
        """
        Full answering pipeline.

        - If RAG is enabled and index exists:
            -> Retrieve context + answer using guidelines.
        - If RAG is disabled:
            -> Answer from general public-health knowledge only.

        Language behaviour:
        - Controlled by build_system_prompt(): reply in the SAME language
          as the user's question, with an extra safeguard for pure English.
        """
        # 1) Retrieve context (if any)
        docs = self.retrieve_context(user_query)
        context_text = self._format_context(docs) if docs else ""

        # 2) System prompt (public-health rules + language rules)
        system_prompt = build_system_prompt(self.language)

        # <<< NEW ENGLISH GUARD >>>
        # If the user's question is only ASCII (likely English),
        # force the model to reply strictly in English.
        if user_query.isascii():
            system_prompt += (
                "\n\nIMPORTANT: The user's question is written using only English "
                "characters. You MUST answer ONLY in clear, simple English. "
                "Do NOT reply in Hindi, Marathi, or any other language, and do NOT "
                "transliterate English sentences into another script."
            )

        # 3) Build the combined user message
        if docs:
            prefix = (
                "Here is context extracted from official public-health guidelines:\n\n"
                f"{context_text}\n\n"
                "User question:\n"
                f"{user_query}\n\n"
                "Use the guideline context when it is relevant. If the context is "
                "insufficient, you may use general public-health knowledge, but clearly "
                "state that the information may be incomplete.\n"
                "Do NOT provide diagnosis. Do NOT prescribe medications, dosages, or "
                "treatment plans. Always encourage users to consult qualified healthcare "
                "professionals for personal medical decisions."
            )
        else:
            prefix = (
                "No structured guideline snippets are available. "
                "Answer based on general, widely accepted public-health knowledge only.\n\n"
                "User question:\n"
                f"{user_query}\n\n"
                "Be especially clear that this is not medical advice and that the user "
                "must consult a qualified professional for any personal decision."
            )

        # 4) Call Groq LLM
        completion = self.client.chat.completions.create(
            model=self.settings.llm_model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prefix},
            ],
            temperature=0.2,
        )

        raw_answer = completion.choices[0].message.content.strip()

        # 5) Apply guardrails
        safe_answer = apply_guardrails(user_query, raw_answer)
        return safe_answer
