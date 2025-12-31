"""LlamaIndex-based RAG client with ChromaDB backend and local HuggingFace embeddings."""

import os
from pathlib import Path
from typing import Optional

import chromadb
from llama_index.core import (
    Document,
    Settings,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore


def get_device() -> str:
    """Get the best available device for embeddings.

    Prefers MPS (Apple Silicon), then CUDA, then CPU.
    """
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


class RAGClient:
    """LlamaIndex-based RAG client with ChromaDB and local embeddings."""

    DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"  # Good balance of quality and speed
    DEFAULT_CHUNK_SIZE = 512
    DEFAULT_CHUNK_OVERLAP = 50

    def __init__(
        self,
        persist_directory: str | Path,
        model_name: Optional[str] = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ):
        """Initialize RAG client with persistent ChromaDB storage.

        Args:
            persist_directory: Directory for ChromaDB and index storage
            model_name: HuggingFace embedding model name
            chunk_size: Size of text chunks for embedding
            chunk_overlap: Overlap between chunks
        """
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        self.model_name = model_name or self.DEFAULT_MODEL
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.device = get_device()

        # Initialize embedding model with GPU support
        self.embed_model = HuggingFaceEmbedding(
            model_name=self.model_name,
            device=self.device,
        )

        # Set global settings (no LLM needed for embedding/retrieval only)
        Settings.embed_model = self.embed_model
        Settings.llm = None  # We don't use LLM for search

        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(
            path=str(self.persist_directory / "chroma"),
            settings=chromadb.Settings(
                anonymized_telemetry=False,
                allow_reset=True,
            ),
        )

        # Text splitter for chunking
        self.text_splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        # Cache for loaded indexes
        self._indexes: dict[str, VectorStoreIndex] = {}

    def _get_collection_path(self, collection_name: str) -> Path:
        """Get the path for a collection's storage."""
        return self.persist_directory / collection_name

    def create_index(
        self,
        collection_name: str,
        documents: list[Document],
    ) -> VectorStoreIndex:
        """Create a new index from documents.

        Args:
            collection_name: Name for the collection
            documents: List of LlamaIndex Document objects

        Returns:
            VectorStoreIndex for querying
        """
        # Get or create ChromaDB collection
        chroma_collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
        )

        # Create vector store
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

        # Create storage context
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # Build index
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            transformations=[self.text_splitter],
            show_progress=True,
        )

        # Persist index metadata
        collection_path = self._get_collection_path(collection_name)
        collection_path.mkdir(parents=True, exist_ok=True)
        storage_context.persist(persist_dir=str(collection_path))

        # Cache the index
        self._indexes[collection_name] = index

        return index

    def load_index(self, collection_name: str) -> Optional[VectorStoreIndex]:
        """Load an existing index.

        Args:
            collection_name: Name of the collection

        Returns:
            VectorStoreIndex or None if not found
        """
        # Check cache first
        if collection_name in self._indexes:
            return self._indexes[collection_name]

        collection_path = self._get_collection_path(collection_name)

        if not collection_path.exists():
            return None

        try:
            # Get ChromaDB collection
            chroma_collection = self.chroma_client.get_collection(
                name=collection_name,
            )

            # Create vector store
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

            # Load storage context
            storage_context = StorageContext.from_defaults(
                vector_store=vector_store,
                persist_dir=str(collection_path),
            )

            # Load index
            index = load_index_from_storage(
                storage_context,
                embed_model=self.embed_model,
            )

            # Cache the index
            self._indexes[collection_name] = index

            return index

        except Exception:
            return None

    def get_or_create_index(
        self,
        collection_name: str,
        documents: Optional[list[Document]] = None,
    ) -> Optional[VectorStoreIndex]:
        """Get existing index or create from documents.

        Args:
            collection_name: Name of the collection
            documents: Documents to use if creating new index

        Returns:
            VectorStoreIndex or None
        """
        index = self.load_index(collection_name)
        if index:
            return index

        if documents:
            return self.create_index(collection_name, documents)

        return None

    def query(
        self,
        collection_name: str,
        query_text: str,
        top_k: int = 5,
    ) -> list[dict]:
        """Query an index for similar documents.

        Args:
            collection_name: Name of the collection
            query_text: Query text
            top_k: Number of results to return

        Returns:
            List of results with text, score, and metadata
        """
        index = self.load_index(collection_name)
        if not index:
            return []

        # Create retriever
        retriever = index.as_retriever(similarity_top_k=top_k)

        # Retrieve nodes
        nodes = retriever.retrieve(query_text)

        results = []
        for node in nodes:
            results.append({
                "text": node.node.get_content(),
                "score": node.score,
                "metadata": node.node.metadata,
            })

        return results

    def list_collections(self) -> list[str]:
        """List all collection names."""
        return [c.name for c in self.chroma_client.list_collections()]

    def get_collection_count(self, collection_name: str) -> int:
        """Get the number of documents in a collection."""
        try:
            collection = self.chroma_client.get_collection(collection_name)
            return collection.count()
        except Exception:
            return 0

    def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection."""
        try:
            self.chroma_client.delete_collection(collection_name)
            if collection_name in self._indexes:
                del self._indexes[collection_name]

            # Remove storage directory
            collection_path = self._get_collection_path(collection_name)
            if collection_path.exists():
                import shutil
                shutil.rmtree(collection_path)

            return True
        except Exception:
            return False


def create_documents_from_markdown(
    markdown_path: str | Path,
    metadata: Optional[dict] = None,
) -> list[Document]:
    """Create LlamaIndex Documents from a markdown file.

    Args:
        markdown_path: Path to the markdown file
        metadata: Optional metadata to attach to documents

    Returns:
        List of Document objects
    """
    path = Path(markdown_path)
    content = path.read_text(encoding="utf-8")

    doc_metadata = {
        "source": str(path),
        "filename": path.name,
    }
    if metadata:
        doc_metadata.update(metadata)

    return [Document(text=content, metadata=doc_metadata)]
