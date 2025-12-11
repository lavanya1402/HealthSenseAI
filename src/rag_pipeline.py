"""
rag_pipeline.py
RAG pipeline for HealthSenseAI (Groq + LangChain + FAISS).

Behaviour (STRICT RAG, zero hallucination):

- If a FAISS index exists AND guideline chunks are retrieved:
      → Answer ONLY from those guideline excerpts.
- If no relevant excerpts are retrieved:
      → Reply that the guideline does not provide information.
- If FAISS index cannot be built / loaded:
      → Reply that the guideline index is unavailable.

All answers:
- Are educational only (no diagnosis, no prescriptions).
- Include a "Sources (public health guidelines)" section when RAG is used.
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
    HealthSenseAI RAG engine.

    - Attempts to build / load a FAISS vector index from guideline PDFs.
    - If text cannot be extracted (e.g., image-only PDFs), runs in
      "index unavailable" mode and clearly reports that to the user.
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

    # ------------------------------------------------------------------
    # Loading + Indexing
    # ------------------------------------------------------------------
    def _load_pdfs(self) -> List[Document]:
        """
        Load PDFs from data_raw_dir and attach metadata:
        - source: filename (not full path, nicer in UI)
        - page:   1-based page index
        """
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

            for i, d in enumerate(pdf_docs):
                meta = dict(d.metadata or {})
                # Normalise metadata for cleaner citations
                meta["source"] = pdf_path.name
                meta["page"] = i + 1
                d.metadata = meta
                docs.append(d)

        print(
            f"[HealthSenseRAG] Loaded {len(docs)} document pages from PDFs with metadata."
        )
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
        """
        Build a FAISS index from guideline PDFs, or load an existing one.

        If anything goes wrong (no PDFs, no text, no chunks), RAG is disabled
        and the app will respond with an "index unavailable" message.
        """
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
                "Index cannot be built."
            )
            self._vectorstore = None
            self.rag_enabled = False
            return

        chunked_docs = self._split_documents(filtered_docs)
        if not chunked_docs:
            print(
                "[HealthSenseRAG] Splitting produced zero chunks. "
                "Index cannot be built."
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

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    def retrieve_context(self, query: str) -> List[Document]:
        """
        Retrieve top-k similar chunks from FAISS.

        If RAG is disabled or index cannot be loaded, returns an empty list.
        """
        if not self.rag_enabled:
            print("[HealthSenseRAG] RAG disabled – skipping retrieval.")
            return []

        if self._vectorstore is None:
            self.build_or_load_index()

        if self._vectorstore is None:
            print("[HealthSenseRAG] Vectorstore unavailable – skipping retrieval.")
            return []

        docs = self._vectorstore.similarity_search(query, k=self.top_k)

        for d in docs:
            print(
                f"[RAG] Retrieved from source={d.metadata.get('source')} "
                f"page={d.metadata.get('page')}"
            )

        return docs

    # ------------------------------------------------------------------
    # Answering (STRICT RAG)
    # ------------------------------------------------------------------
    def answer_query(self, user_query: str) -> str:
        """
        STRICT RAG ANSWERING (zero hallucination):

        - If RAG is enabled and guideline chunks exist:
              → Answer ONLY from those guideline excerpts.
        - If no matching guideline text exists:
              → Clearly say that the guideline does not provide
                information on this topic.
        - If RAG is disabled even after trying to build:
              → Clearly say that the guideline index is unavailable.

        All answers:
        - Are educational only.
        - Include a sources section when RAG is used.
        """

        # 0) If FAISS index not available, try (again) to build it.
        if not self.rag_enabled or self._vectorstore is None:
            print("[HealthSenseRAG] RAG not ready – attempting to (re)build index...")
            self.build_or_load_index()

        # Still not available → give clean message
        if not self.rag_enabled or self._vectorstore is None:
            return (
                "⚠️ **Guideline index unavailable**\n\n"
                "HealthSenseAI could not access a WHO/CDC/MoHFW guideline index. "
                "Please ensure that readable (text-based) PDFs are available in "
                "`data/raw/` and that at least one of them contains selectable text "
                "rather than only scanned images."
            )

        # 1) Retrieve guideline chunks
        docs = self.retrieve_context(user_query)

        # No matching guideline content
        if not docs:
            return (
                "⚠️ **Guideline Not Found**\n\n"
                "The uploaded guidelines do **not contain information clearly related** "
                "to your question. Please upload additional WHO/CDC/MoHFW PDFs or "
                "consult a qualified healthcare professional for advice."
            )

        # 2) Prepare contextual excerpts for the LLM (not shown directly to the user)
        context_for_llm: List[str] = []
        for i, d in enumerate(docs, start=1):
            src = d.metadata.get("source", "Guideline")
            page = d.metadata.get("page", "Unknown")
            context_for_llm.append(
                f"[Excerpt {i} | Source: {src} | Page: {page}]\n{d.page_content.strip()}"
            )

        formatted_context = "\n\n".join(context_for_llm)

        # 3) Base system prompt (language + safety framing)
        base_system_prompt = build_system_prompt(self.language)

        # 4) Strict RAG rules appended to system prompt
        system_prompt = (
            base_system_prompt
            + "\n\n"
            "You must obey the following STRICT RULES:\n"
            "- You MUST answer ONLY using the information in the provided guideline excerpts.\n"
            "- You are **not allowed** to use outside medical knowledge or assumptions.\n"
            "- If the excerpts do NOT contain enough information to answer, reply exactly:\n"
            "  'The guideline does not provide information on this topic.'\n"
            "- Do NOT invent or guess any facts.\n"
            "- Do NOT provide diagnosis.\n"
            "- Do NOT prescribe medicines, doses, or treatment plans.\n"
            "- Keep the tone educational and supportive.\n"
            "- Keep the answer concise: at most ~8 short bullet points or 200–250 words.\n"
            "- Do NOT repeat the same sentence or phrase multiple times.\n"
        )

        # 5) User message with context + question
        user_prompt = (
            f"### Guideline Excerpts\n{formatted_context}\n\n"
            f"### User Question\n{user_query}\n\n"
            "### Task\n"
            "Using ONLY the information in the excerpts above, give a short, clear answer "
            "in the same language as the user. If the excerpts do not contain enough "
            "information to safely answer, reply exactly:\n"
            "'The guideline does not provide information on this topic.'"
        )

        # 6) LLM call – temperature fixed to 0 for zero drift
        completion = self.client.chat.completions.create(
            model=self.settings.llm_model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            # You can adjust if needed
            max_tokens=400,
        )

        raw_answer = completion.choices[0].message.content.strip()

        # 7) Apply safety guardrails (medication / diagnosis filters etc.)
        safe_answer = apply_guardrails(user_query, raw_answer)

        # 8) Generate citation section
        seen = set()
        lines: List[str] = []

        for d in docs:
            src = d.metadata.get("source", None)
            page = d.metadata.get("page", None)
            if not src:
                continue

            key = (src, page)
            if key in seen:
                continue

            seen.add(key)
            if page is not None:
                lines.append(f"- **{src}** — page {page}")
            else:
                lines.append(f"- **{src}**")

        if lines:
            citation_block = (
                "\n\n---\n"
                "**Sources (public health guidelines)**\n"
                + "\n".join(lines)
            )
            safe_answer += citation_block

        return safe_answer
