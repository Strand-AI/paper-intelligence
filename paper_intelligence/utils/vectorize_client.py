"""RAG client with local HuggingFace embeddings and Cloudflare Vectorize backend."""

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Optional

import requests
from llama_index.core import Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


def get_device() -> str:
    """Get the best available device for embeddings."""
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


WORKER_URL = os.environ.get(
    "PAPER_INTELLIGENCE_URL",
    "https://paper-intelligence.daiyue0531.workers.dev",
)

_token_cache: str = ""


def _get_token() -> str:
    global _token_cache
    if _token_cache:
        return _token_cache
    _token_cache = os.environ.get("PAPER_INTELLIGENCE_TOKEN", "")
    if _token_cache:
        return _token_cache
    try:
        _token_cache = subprocess.check_output(
            ["op", "read", "op://CLI Secrets/Paper Intelligence Cloud API Token/credential"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        pass
    return _token_cache


def _worker_request(path: str, body: dict) -> dict:
    """Make an authenticated POST to the Worker."""
    resp = requests.post(
        f"{WORKER_URL}{path}",
        json=body,
        headers={"Authorization": f"Bearer {_get_token()}"},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


class VectorizeClient:
    """RAG client: local embeddings + Cloudflare Vectorize storage."""

    DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
    DEFAULT_CHUNK_SIZE = 512
    DEFAULT_CHUNK_OVERLAP = 50

    def __init__(
        self,
        paper_name: str,
        model_name: Optional[str] = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ):
        self.paper_name = paper_name
        self.model_name = model_name or self.DEFAULT_MODEL
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.device = get_device()

        self.embed_model = HuggingFaceEmbedding(
            model_name=self.model_name,
            device=self.device,
        )

    @staticmethod
    def _vector_id(paper_name: str, chunk_index: int) -> str:
        """Generate a vector ID that fits Vectorize's 64-byte limit."""
        # Use a short hash of the paper name + chunk index
        h = hashlib.sha256(paper_name.encode()).hexdigest()[:16]
        return f"{h}_{chunk_index}"

    def embed_and_store(self, documents: list[Document]) -> int:
        """Embed documents locally and store vectors in Vectorize.

        Returns the number of chunks stored.
        """
        vectors = []
        for i, doc in enumerate(documents):
            text = doc.get_content()
            embedding = self.embed_model.get_text_embedding(text)
            chunk_index = doc.metadata.get("chunk_index", i)
            vectors.append({
                "id": self._vector_id(self.paper_name, chunk_index),
                "values": embedding,
                "metadata": {
                    "paper_name": self.paper_name,
                    "text": text[:8000],  # Vectorize 10KB metadata limit
                    "header_path": doc.metadata.get("header_path", ""),
                    "start_line": doc.metadata.get("start_line", 0),
                    "end_line": doc.metadata.get("end_line", 0),
                    "chunk_index": chunk_index,
                },
            })

        if vectors:
            _worker_request("/upsert", {"vectors": vectors})

        return len(vectors)

    def query(self, query_text: str, top_k: int = 5) -> list[dict]:
        """Embed query locally, search Vectorize for similar chunks.

        Returns list of {text, score, metadata}.
        """
        query_embedding = self.embed_model.get_query_embedding(query_text)

        result = _worker_request("/query", {
            "vector": query_embedding,
            "top_k": top_k,
            "filter": {"paper_name": self.paper_name},
        })

        return [
            {
                "text": m["metadata"].get("text", ""),
                "score": m["score"],
                "metadata": m["metadata"],
            }
            for m in result.get("matches", [])
        ]

    def delete(self, num_chunks: int) -> bool:
        """Delete all vectors for this paper."""
        ids = [self._vector_id(self.paper_name, i) for i in range(num_chunks)]
        if ids:
            _worker_request("/delete", {"ids": ids})
        return True


def query_all_papers(
    query_text: str,
    top_k: int = 5,
    model_name: Optional[str] = None,
) -> list[dict]:
    """Search across ALL papers in Vectorize (no paper_name filter).

    Used for cross-paper RAG search.
    """
    device = get_device()
    embed_model = HuggingFaceEmbedding(
        model_name=model_name or VectorizeClient.DEFAULT_MODEL,
        device=device,
    )
    query_embedding = embed_model.get_query_embedding(query_text)

    result = _worker_request("/query", {
        "vector": query_embedding,
        "top_k": top_k,
    })

    return [
        {
            "text": m["metadata"].get("text", ""),
            "score": m["score"],
            "metadata": m["metadata"],
        }
        for m in result.get("matches", [])
    ]


def create_documents_from_markdown(
    markdown_path: str | Path,
    metadata: Optional[dict] = None,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> list[Document]:
    """Create LlamaIndex Documents from a markdown file with line number metadata.

    Pre-chunks the markdown using header-aware chunking to preserve line numbers.
    """
    from .markdown_parser import MarkdownParser

    path = Path(markdown_path)
    parser = MarkdownParser.from_file(path)
    chunks = parser.chunk_text(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    documents = []
    for chunk in chunks:
        doc_metadata = {
            "source": str(path),
            "filename": path.name,
            "start_line": chunk["start_line"],
            "end_line": chunk["end_line"],
            "header_path": chunk.get("header_path", ""),
        }
        if "chunk_index" in chunk:
            doc_metadata["chunk_index"] = chunk["chunk_index"]
        if metadata:
            doc_metadata.update(metadata)
        documents.append(Document(text=chunk["text"], metadata=doc_metadata))

    return documents
