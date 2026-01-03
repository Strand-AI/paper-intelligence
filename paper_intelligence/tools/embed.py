"""Document embedding tool using LlamaIndex with ChromaDB.

Embeddings are stored in each paper's directory for self-containment.
"""

from pathlib import Path
from typing import Optional

from ..utils.chromadb_client import RAGClient, create_documents_from_markdown


def embed_document(
    markdown_path: str,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> dict:
    """Create embeddings for a markdown document and store in local ChromaDB.

    Embeddings are stored in the same directory as the markdown file
    under a 'chroma/' subdirectory for self-containment.

    Args:
        markdown_path: Path to the markdown file
        chunk_size: Text chunk size for embedding
        chunk_overlap: Overlap between chunks

    Returns:
        Dictionary with:
        - db_path: Path to the ChromaDB storage
        - num_chunks: Number of chunks embedded
        - success: Boolean
        - message: Status message
    """
    md_path = Path(markdown_path).expanduser().resolve()

    if not md_path.exists():
        return {
            "db_path": None,
            "num_chunks": 0,
            "success": False,
            "message": f"Markdown file not found: {markdown_path}",
        }

    # Store embeddings in the paper's directory
    paper_dir = md_path.parent
    chroma_dir = paper_dir / "chroma"

    try:
        # Create RAG client with paper-local storage
        rag_client = RAGClient(
            persist_directory=chroma_dir,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        # Create pre-chunked documents with line number metadata
        documents = create_documents_from_markdown(
            md_path,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        # Use a fixed collection name since each paper has its own DB
        collection_name = "paper"

        # Create index (skip chunking since documents are pre-chunked with line metadata)
        rag_client.create_index(collection_name, documents, pre_chunked=True)

        # Get chunk count
        num_chunks = rag_client.get_collection_count(collection_name)

        # Update metadata
        from ..metadata import update_metadata_steps

        update_metadata_steps(paper_dir, "embed")

        return {
            "db_path": str(chroma_dir),
            "num_chunks": num_chunks,
            "success": True,
            "message": f"Successfully embedded document into {num_chunks} chunks",
            "device": rag_client.device,
        }

    except Exception as e:
        return {
            "db_path": None,
            "num_chunks": 0,
            "success": False,
            "message": f"Embedding failed: {str(e)}",
        }


def query_paper(
    paper_dir: str,
    query: str,
    top_k: int = 5,
) -> dict:
    """Query a paper's embeddings for similar content.

    Args:
        paper_dir: Path to the paper's directory (containing chroma/)
        query: Search query
        top_k: Number of results to return

    Returns:
        Dictionary with search results
    """
    paper_path = Path(paper_dir).expanduser().resolve()
    chroma_dir = paper_path / "chroma"

    if not chroma_dir.exists():
        return {
            "results": [],
            "num_results": 0,
            "success": False,
            "message": f"No embeddings found in {paper_dir}",
        }

    try:
        rag_client = RAGClient(persist_directory=chroma_dir)
        results = rag_client.query("paper", query, top_k)

        return {
            "results": results,
            "num_results": len(results),
            "success": True,
        }

    except Exception as e:
        return {
            "results": [],
            "num_results": 0,
            "success": False,
            "message": f"Query failed: {str(e)}",
        }
