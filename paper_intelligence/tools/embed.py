"""Document embedding tool using local HuggingFace model + Cloudflare Vectorize."""

from pathlib import Path

from ..utils.vectorize_client import VectorizeClient, create_documents_from_markdown


def embed_document(
    markdown_path: str,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> dict:
    """Embed a markdown document locally and store vectors in Cloudflare Vectorize.

    Args:
        markdown_path: Path to the markdown file
        chunk_size: Text chunk size for embedding
        chunk_overlap: Overlap between chunks

    Returns:
        Dictionary with num_chunks, success, message, device
    """
    md_path = Path(markdown_path).expanduser().resolve()

    if not md_path.exists():
        return {
            "num_chunks": 0,
            "success": False,
            "message": f"Markdown file not found: {markdown_path}",
        }

    paper_name = md_path.parent.name

    try:
        client = VectorizeClient(
            paper_name=paper_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        documents = create_documents_from_markdown(
            md_path,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        num_chunks = client.embed_and_store(documents)

        from ..metadata import update_metadata_steps
        update_metadata_steps(md_path.parent, "embed")

        return {
            "num_chunks": num_chunks,
            "success": True,
            "message": f"Embedded {num_chunks} chunks (device={client.device})",
            "device": client.device,
        }

    except Exception as e:
        return {
            "num_chunks": 0,
            "success": False,
            "message": f"Embedding failed: {str(e)}",
        }


def query_paper(
    paper_dir: str,
    query: str,
    top_k: int = 5,
) -> dict:
    """Query a paper's embeddings via Vectorize."""
    paper_name = Path(paper_dir).expanduser().resolve().name

    try:
        client = VectorizeClient(paper_name=paper_name)
        results = client.query(query, top_k)

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
