"""
rag_pipeline.py
RAG pipeline for HealthSenseAI (Groq + LangChain).

Steps:
1. Load health guideline PDFs from data/raw
2. Split into chunks
3. Embed with HuggingFace sentence transformer
4. Store in FAISS index (data/processed/faiss_index)
5. For each query:
   - Retrieve top-k chunks
   - Build prompt with context + user query
   - Ask the LLM (Groq via LangChain)
   - Apply guardrails
"""

from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage

from .config import Settings
from .guards import apply_guardrails
from .utils import build_system_prompt, LanguageCode


class HealthSenseRAG:
    def __init__(
        self,
        settings: Settings,
        llm: BaseChatModel,
        language: LanguageCode = "en",
        top_k: int = 4,
        chunk_size: int = 900,
        chunk_overlap: int = 150,
    ) -> None:
        self.settings = settings
        self.llm = llm
        self.language = language
        self.top_k = top_k
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self._embedding = HuggingFaceEmbeddings(
            model_name=self.settings.embedding_model_name
        )
        self._vectorstore: FAISS | None = None

    # -----------------------------
    # Indexing
    # -----------------------------
    def _load_pdfs(self) -> List[Document]:
        docs: List[Document] = []
        pdf_files = list(self.settings.data_raw_dir.glob("*.pdf"))
        if not pdf_files:
            raise FileNotFoundError(
                f"No PDFs found in {self.settings.data_raw_dir}. "
                "Please add WHO/CDC/MoHFW guideline PDFs there."
            )

        for pdf_path in pdf_files:
            loader = PyPDFLoader(str(pdf_path))
            pdf_docs = loader.load()
            docs.extend(pdf_docs)

        return docs

    def _split_documents(self, docs: List[Document]) -> List[Document]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        return splitter.split_documents(docs)

    def build_or_load_index(self) -> None:
        index_path: Path = self.settings.index_dir

        if (index_path / "index.faiss").exists():
            self._vectorstore = FAISS.load_local(
                str(index_path),
                self._embedding,
                allow_dangerous_deserialization=True,
            )
            return

        docs = self._load_pdfs()
        chunked_docs = self._split_documents(docs)

        vectorstore = FAISS.from_documents(chunked_docs, self._embedding)
        index_path.mkdir(parents=True, exist_ok=True)
        vectorstore.save_local(str(index_path))

        self._vectorstore = vectorstore

    # -----------------------------
    # Query
    # -----------------------------
    def retrieve_context(self, query: str) -> List[Document]:
        if self._vectorstore is None:
            self.build_or_load_index()

        assert self._vectorstore is not None
        docs = self._vectorstore.similarity_search(query, k=self.top_k)
        return docs

    def _format_context(self, docs: List[Document]) -> str:
        context_chunks = [d.page_content.strip() for d in docs]
        return "\n\n---\n\n".join(context_chunks)

    def answer_query(self, user_query: str) -> str:
        docs = self.retrieve_context(user_query)
        context_text = self._format_context(docs)

        system_prompt = build_system_prompt(self.language)

        user_prompt = (
            f"Here is context from public health guidelines:\n\n"
            f"{context_text}\n\n"
            f"User question:\n{user_query}\n\n"
            "Only use the context if it is relevant. If it is not enough, answer based "
            "on general public-health knowledge and say that the information may be "
            "incomplete. Do NOT invent specific statistics or clinical protocols."
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        response = self.llm.invoke(messages)
        answer_text = response.content.strip()

        safe_answer = apply_guardrails(user_query, answer_text)
        return safe_answer
