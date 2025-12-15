"""
rag_pipeline.py
HealthSenseAI - OCR-enabled STRICT RAG pipeline (FAISS + HuggingFace embeddings + Groq)

- Supports SCANNED PDFs via OCR (UnstructuredPDFLoader strategy="ocr_only")
- Builds/loads FAISS index from extracted text chunks
- Strictly answers ONLY from retrieved excerpts
- If not covered, returns STRICT_FALLBACK
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ✅ OCR loader (for scanned PDFs)
from langchain_community.document_loaders import UnstructuredPDFLoader

from utils import build_system_prompt
from guards import apply_guardrails

STRICT_FALLBACK = "The guideline does not provide information on this topic."


def _clean_source(src: str) -> str:
    try:
        return Path(src).name
    except Exception:
        return src or "Guideline"


def _pick_raw_dir(settings: Any) -> Path:
    # prefer settings.data_raw_dir if present, else raw_data_dir, else fallback
    if hasattr(settings, "data_raw_dir"):
        return Path(getattr(settings, "data_raw_dir"))
    if hasattr(settings, "raw_data_dir"):
        return Path(getattr(settings, "raw_data_dir"))
    return Path("data/raw")


def _hash_text(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8", errors="ignore")).hexdigest()[:16]


def _manifest_path(index_dir: Path) -> Path:
    return index_dir / "manifest.json"


def _build_pdf_manifest(pdf_dir: Path) -> Dict[str, Any]:
    files = []
    for p in sorted(pdf_dir.glob("*.pdf")):
        try:
            stat = p.stat()
            files.append({"name": p.name, "size": stat.st_size, "mtime": int(stat.st_mtime)})
        except Exception:
            files.append({"name": p.name, "size": None, "mtime": None})
    return {"pdf_dir": str(pdf_dir), "files": files}


def _load_manifest(index_dir: Path) -> Optional[Dict[str, Any]]:
    mp = _manifest_path(index_dir)
    if not mp.exists():
        return None
    try:
        return json.loads(mp.read_text(encoding="utf-8"))
    except Exception:
        return None


def _save_manifest(index_dir: Path, manifest: Dict[str, Any]) -> None:
    _manifest_path(index_dir).write_text(json.dumps(manifest, indent=2), encoding="utf-8")


class HealthSenseRAG:
    """
    Returns: (response_text, coverage_label, pairs)
    coverage_label: CLEAR | PARTIAL | NONE
    pairs: list[(Document, score)] for debugging
    """

    def __init__(self, settings, llm, language: str = "en"):
        self.settings = settings
        self.client = llm  # Groq client
        self.language = language

        self.pdf_dir: Path = _pick_raw_dir(settings)
        self.index_dir: Path = Path(getattr(settings, "index_dir", "data/index"))

        self.pdf_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)

        self.top_k = int(getattr(settings, "top_k", 6))
        self.chunk_size = int(getattr(settings, "chunk_size", 900))
        self.chunk_overlap = int(getattr(settings, "chunk_overlap", 150))

        # L2 distance (lower = better). Keep these moderate.
        self.clear_score_threshold = float(getattr(settings, "clear_score_threshold", 1.20))
        self.partial_score_threshold = float(getattr(settings, "partial_score_threshold", 2.20))

        emb_name = getattr(settings, "embedding_model", None)
        if not emb_name:
            raise AttributeError("Settings.embedding_model missing.")
        self.embeddings = HuggingFaceEmbeddings(model_name=emb_name)

        self._vectorstore: Optional[FAISS] = None
        self.rag_enabled: bool = False

    # ----------------------- Index helpers -----------------------
    def _index_exists(self) -> bool:
        return (self.index_dir / "index.faiss").exists() and (self.index_dir / "index.pkl").exists()

    def _needs_rebuild(self) -> bool:
        """Rebuild if PDFs changed since last index build."""
        current = _build_pdf_manifest(self.pdf_dir)
        saved = _load_manifest(self.index_dir)
        return saved != current

    def force_rebuild(self) -> None:
        """Delete old index files safely and rebuild."""
        for fn in ["index.faiss", "index.pkl", "manifest.json"]:
            p = self.index_dir / fn
            if p.exists():
                p.unlink()
        self._vectorstore = None
        self.rag_enabled = False
        self.build_or_load_index()

    def _load_pdf_pages_with_ocr(self, pdf_path: Path):
        """
        OCR loader for scanned PDFs.
        Returns a list[Document].
        """
        loader = UnstructuredPDFLoader(
            str(pdf_path),
            strategy="ocr_only",   # ✅ force OCR for scanned pages
            mode="elements",       # produces smaller text elements
        )
        return loader.load()

    def build_or_load_index(self) -> None:
        pdfs = sorted(self.pdf_dir.glob("*.pdf"))
        if not pdfs:
            self._vectorstore = None
            self.rag_enabled = False
            return

        # If index exists but PDFs changed => rebuild
        if self._index_exists() and self._needs_rebuild():
            self.force_rebuild()
            return

        # Load if valid
        if self._index_exists():
            self._vectorstore = FAISS.load_local(
                str(self.index_dir),
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
            self.rag_enabled = True
            return

        # ✅ Build new index from OCR text
        pages = []
        for p in pdfs:
            try:
                pages.extend(self._load_pdf_pages_with_ocr(p))
            except Exception:
                # If one PDF fails OCR, skip it instead of crashing whole app
                continue

        # ✅ If OCR returns nothing, don't build empty FAISS
        if not pages:
            self._vectorstore = None
            self.rag_enabled = False
            return

        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        chunks = splitter.split_documents(pages)

        # ✅ Prevent IndexError / embed crash if chunks empty
        chunks = [c for c in chunks if (c.page_content or "").strip()]
        if not chunks:
            self._vectorstore = None
            self.rag_enabled = False
            return

        self._vectorstore = FAISS.from_documents(chunks, self.embeddings)
        self._vectorstore.save_local(str(self.index_dir))

        # Save manifest so we can detect PDF changes later
        _save_manifest(self.index_dir, _build_pdf_manifest(self.pdf_dir))

        self.rag_enabled = True

    # ----------------------- Retrieval -----------------------
    def _coverage(self, pairs: List[Tuple[Any, float]]) -> str:
        if not pairs:
            return "NONE"
        best = float(pairs[0][1])  # L2 distance lower=better
        if best <= self.clear_score_threshold:
            return "CLEAR"
        if best <= self.partial_score_threshold:
            return "PARTIAL"
        return "NONE"

    def retrieve_with_scores(self, user_query: str) -> List[Tuple[Any, float]]:
        if not self._vectorstore:
            return []
        pairs = self._vectorstore.similarity_search_with_score(user_query, k=self.top_k)
        return [(d, float(s)) for d, s in pairs]

    def _sources_block(self, docs: List[Any]) -> str:
        seen = set()
        lines = []
        for d in docs:
            src = _clean_source(d.metadata.get("source", "Guideline"))
            page = d.metadata.get("page", None)
            key = (src, page)
            if key in seen:
                continue
            seen.add(key)
            lines.append(f"- **{src}** — page {page}" if page is not None else f"- **{src}**")
        if not lines:
            return ""
        return "\n\n---\n**Sources (guidelines)**\n" + "\n".join(lines)

    # ----------------------- Answering (STRICT) -----------------------
    def answer_query(self, user_query: str) -> Tuple[str, str, List[Tuple[Any, float]]]:
        if not self.rag_enabled or not self._vectorstore:
            self.build_or_load_index()

        if not self.rag_enabled or not self._vectorstore:
            return (
                "⚠️ **Guideline index unavailable**\n\n"
                f"Either no PDFs exist in `{self.pdf_dir}`, or OCR could not extract text.\n\n"
                "✅ Fix:\n"
                "- Ensure PDFs are readable\n"
                "- Install OCR deps: `unstructured pytesseract pdf2image pillow`\n"
                "- Install Tesseract OCR (system)\n",
                "NONE",
                [],
            )

        pairs = self.retrieve_with_scores(user_query)
        coverage = self._coverage(pairs)

        if coverage == "NONE":
            return STRICT_FALLBACK, "NONE", pairs

        docs = [d for d, _ in pairs]

        excerpts = []
        for i, d in enumerate(docs, start=1):
            src = _clean_source(d.metadata.get("source", "Guideline"))
            page = d.metadata.get("page", "Unknown")
            excerpts.append(f"[Excerpt {i} | Source: {src} | Page: {page}]\n{(d.page_content or '').strip()}")

        formatted_context = "\n\n".join(excerpts)

        system_prompt = (
            build_system_prompt(self.language)
            + "\n\nSTRICT RULES:\n"
            "- Use ONLY the excerpts.\n"
            f"- If NOT found in excerpts, reply EXACTLY: {STRICT_FALLBACK}\n"
            "- No guessing/inference.\n\n"
            "MANDATORY OUTPUT FORMAT:\n"
            "Direct Answer:\n"
            "- 2–6 bullets, simple patient guidance strictly from excerpt.\n\n"
            "Guideline Evidence:\n"
            "- Quote 1–2 exact lines verbatim as blockquote using (> ).\n"
        )

        user_prompt = (
            f"### Guideline Excerpts\n{formatted_context}\n\n"
            f"### User Question\n{user_query}\n\n"
            "### Mandatory Output Format\n"
            "Direct Answer:\n"
            "- Give patient-friendly guidance strictly from excerpts.\n\n"
            "Guideline Evidence:\n"
            "- Quote exact supporting line(s) using blockquote (> ).\n\n"
            f"If not present, say exactly:\n{STRICT_FALLBACK}"
        )

        completion = self.client.chat.completions.create(
            model=self.settings.llm_model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=650,
        )

        raw = completion.choices[0].message.content.strip()
        safe, _ = apply_guardrails(user_query, raw)

        # Hard enforcement: evidence must contain at least one blockquote line
        if safe != STRICT_FALLBACK and ("Guideline Evidence:" in safe) and (">" not in safe):
            safe = STRICT_FALLBACK

        if safe != STRICT_FALLBACK:
            safe += self._sources_block(docs)

        return safe, coverage, pairs
