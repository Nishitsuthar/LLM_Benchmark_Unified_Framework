"""RAG evidence builder — Nishit.

Chunks PDFs, embeds them with MiniLM, stores in ChromaDB, and retrieves
the top-k most relevant chunks for the question. Ported from Nishit's
Sprint 3 RAG pipeline.

When a question carries a `source_pdf` field, only that PDF is indexed
and searched — prevents cross-company context bleed in multi-PDF folders.
"""

import hashlib
import re
from pathlib import Path

from unified_pipeline.base import BaseEvidenceBuilder, EvidenceResult

CHUNK_SIZE = 500      # tokens (words) per chunk
CHUNK_OVERLAP = 50    # words of overlap between consecutive chunks


class RagEvidenceBuilder(BaseEvidenceBuilder):
    def __init__(self):
        # Lazy import so others don't need these deps installed
        import chromadb
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.client = chromadb.PersistentClient(path=".chroma_cache")
        self._collections: dict = {}

    def build(self, question: str, config: dict) -> EvidenceResult:
        top_k = config.get("rag_top_k", 10)

        # If the question names a specific PDF, index only that file.
        # Otherwise fall back to the whole folder (e.g. single-PDF datasets).
        source_pdf = config.get("_source_pdf")
        if source_pdf:
            pdf_path = Path(config["evidence_path"]) / source_pdf
            if not pdf_path.exists():
                raise FileNotFoundError(f"source_pdf not found: {pdf_path}")
            collection = self._get_or_index_file(pdf_path)
        else:
            collection = self._get_or_index_folder(config["evidence_path"])

        # Cap n_results to actual collection size to avoid ChromaDB errors
        n_results = min(top_k, collection.count())
        if n_results == 0:
            return EvidenceResult(text="", metadata={"chunks_retrieved": 0, "top_k": top_k})

        results = collection.query(
            query_embeddings=[self.model.encode(question).tolist()],
            n_results=n_results,
        )
        chunks = results["documents"][0]

        return EvidenceResult(
            text="\n\n".join(chunks),
            metadata={"chunks_retrieved": len(chunks), "top_k": top_k},
        )

    def _collection_name(self, key: str) -> str:
        """Stable, safe ChromaDB collection name from an arbitrary path string."""
        h = hashlib.md5(key.encode()).hexdigest()[:12]
        slug = re.sub(r"[^a-zA-Z0-9\-]", "-", Path(key).name)[:50]
        return f"{slug}-{h}"

    def _get_or_index_file(self, pdf_path: Path):
        """Index a single PDF; reuse collection on subsequent calls."""
        key = str(pdf_path.resolve())
        if key in self._collections:
            return self._collections[key]

        collection = self.client.get_or_create_collection(self._collection_name(key))
        if collection.count() == 0:
            self._index_pdfs([pdf_path], collection)

        self._collections[key] = collection
        return collection

    def _get_or_index_folder(self, pdf_folder: str):
        """Index all PDFs in a folder; reuse collection on subsequent calls."""
        if pdf_folder in self._collections:
            return self._collections[pdf_folder]

        collection = self.client.get_or_create_collection(
            self._collection_name(pdf_folder)
        )
        if collection.count() == 0:
            pdf_files = sorted(Path(pdf_folder).glob("**/*.pdf"))
            if not pdf_files:
                raise FileNotFoundError(f"No PDF files found in: {pdf_folder}")
            self._index_pdfs(pdf_files, collection)

        self._collections[pdf_folder] = collection
        return collection

    def _index_pdfs(self, pdf_files: list, collection) -> None:
        """Chunk, embed, and upsert a list of PDF paths into a collection."""
        import fitz  # PyMuPDF

        all_chunks: list[str] = []
        all_ids: list[str] = []

        for pdf_path in pdf_files:
            doc = fitz.open(str(pdf_path))
            full_text = "\n".join(page.get_text() for page in doc)
            doc.close()

            words = full_text.split()
            chunks = _sliding_window(words, CHUNK_SIZE, CHUNK_OVERLAP)

            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_ids.append(f"{pdf_path.stem}_c{i}")

        if not all_chunks:
            return

        embeddings = self.model.encode(all_chunks, show_progress_bar=False).tolist()
        batch = 256
        for start in range(0, len(all_chunks), batch):
            collection.upsert(
                ids=all_ids[start : start + batch],
                documents=all_chunks[start : start + batch],
                embeddings=embeddings[start : start + batch],
            )


def _sliding_window(words: list[str], size: int, overlap: int) -> list[str]:
    """Split a word list into overlapping chunks, returning each as a string."""
    if not words:
        return []
    step = max(1, size - overlap)
    chunks = []
    for start in range(0, len(words), step):
        chunk = " ".join(words[start : start + size])
        if chunk:
            chunks.append(chunk)
        if start + size >= len(words):
            break
    return chunks
