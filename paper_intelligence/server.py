"""Paper Intelligence MCP Server.

A local MCP server for intelligent paper/PDF management with:
- PDF to Markdown conversion using Marker
- Local embedding/RAG database using LlamaIndex + ChromaDB
- Markdown structure indexing
- Unified search (grep + semantic RAG)

Each paper is self-contained in its own directory.
"""

from typing import Literal, Optional

from mcp.server.fastmcp import FastMCP

# Initialize MCP server
mcp = FastMCP(
    "paper-intelligence",
    instructions="MCP server for intelligent paper/PDF management with RAG capabilities. "
    "Use process_paper to convert PDFs. Each paper gets its own self-contained directory.",
)


@mcp.tool()
def convert_pdf(
    pdf_path: str,
    output_dir: Optional[str] = None,
    use_llm: bool = False,
) -> dict:
    """Convert a PDF file to Markdown using Marker.

    Creates a sibling directory with the same name as the PDF.
    e.g., ~/Downloads/paper.pdf -> ~/Downloads/paper/paper.md

    Args:
        pdf_path: Path to the PDF file
        output_dir: Optional output directory (defaults to sibling dir of PDF)
        use_llm: Use LLM for enhanced accuracy (requires API key)

    Returns:
        Result with markdown_path, output_dir, and success status
    """
    from .tools.convert import convert_pdf as _convert_pdf

    return _convert_pdf(
        pdf_path=pdf_path,
        output_dir=output_dir,
        use_llm=use_llm,
    )


@mcp.tool()
def index_markdown(
    markdown_path: str,
    output_path: Optional[str] = None,
) -> dict:
    """Extract header hierarchy from a markdown file into a searchable JSON index.

    The index is used by grep search to provide header context for matches.

    Args:
        markdown_path: Path to the markdown file
        output_path: Path for the index JSON (defaults to same dir as markdown)

    Returns:
        Result with index_path, headers list, and success status
    """
    from .tools.index import index_markdown as _index_markdown

    return _index_markdown(
        markdown_path=markdown_path,
        output_path=output_path,
    )


@mcp.tool()
def embed_document(
    markdown_path: str,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> dict:
    """Create embeddings for a markdown document and store in local ChromaDB.

    Embeddings are stored in the paper's directory under chroma/ for self-containment.
    Uses LlamaIndex with HuggingFace embeddings (BAAI/bge-small-en-v1.5).
    Supports GPU acceleration on Apple Silicon (MPS) and NVIDIA GPUs (CUDA).

    Args:
        markdown_path: Path to the markdown file
        chunk_size: Text chunk size for embedding (default 512)
        chunk_overlap: Overlap between chunks (default 50)

    Returns:
        Result with db_path, num_chunks, and success status
    """
    from .tools.embed import embed_document as _embed_document

    return _embed_document(
        markdown_path=markdown_path,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


@mcp.tool()
def search(
    query: str,
    paper_dirs: list[str],
    mode: Literal["grep", "rag", "hybrid"] = "hybrid",
    top_k: int = 5,
    case_sensitive: bool = False,
    regex: bool = False,
    include_context: bool = True,
) -> dict:
    """Unified search across paper directories using grep and/or RAG.

    Supports three search modes:
    - "grep": Fast text/regex search with line numbers and header context
    - "rag": Semantic similarity search using embeddings
    - "hybrid": Combined grep + RAG with deduplication (default)

    Args:
        query: Search query (text, regex if regex=True, or semantic query)
        paper_dirs: List of paper directories to search (will find paper.md inside)
        mode: Search mode - "grep", "rag", or "hybrid" (default)
        top_k: Number of results to return (default 5)
        case_sensitive: Case sensitivity for grep (default False)
        regex: Treat query as regex pattern for grep (default False)
        include_context: Include surrounding context in results (default True)

    Returns:
        Result with results list, num_results, and success status
    """
    from .tools.search import search as _search

    return _search(
        query=query,
        paper_dirs=paper_dirs,
        mode=mode,
        top_k=top_k,
        case_sensitive=case_sensitive,
        regex=regex,
        include_context=include_context,
    )


@mcp.tool()
def get_paper_info(paper_dir: str) -> dict:
    """Get information about a paper directory.

    Shows what processing has been completed:
    - has_markdown: PDF conversion done
    - has_index: Header extraction done
    - has_embeddings: RAG embeddings created

    Args:
        paper_dir: Path to the paper directory

    Returns:
        Paper information including processing status
    """
    from .tools.search import get_paper_info as _get_paper_info

    return _get_paper_info(paper_dir=paper_dir)


@mcp.tool()
def process_paper(
    pdf_path: str,
    output_dir: Optional[str] = None,
    use_llm: bool = False,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> dict:
    """Full pipeline: Convert PDF, index headers, and create embeddings.

    This is the main tool for processing papers. It:
    1. Converts PDF to Markdown using Marker
    2. Extracts header hierarchy into JSON index
    3. Creates embeddings and stores in ChromaDB

    Output is self-contained in a single directory:
    ~/Downloads/paper.pdf -> ~/Downloads/paper/
        paper.md      (markdown)
        index.json    (headers)
        chroma/       (embeddings)
        images/       (if any)

    Args:
        pdf_path: Path to the PDF file
        output_dir: Output directory (defaults to sibling dir of PDF)
        use_llm: Use LLM for enhanced PDF conversion accuracy
        chunk_size: Text chunk size for embedding
        chunk_overlap: Overlap between chunks

    Returns:
        Combined result from all processing steps
    """
    from .tools.convert import convert_pdf as _convert_pdf
    from .tools.embed import embed_document as _embed_document
    from .tools.index import index_markdown as _index_markdown

    results = {
        "pdf_path": pdf_path,
        "steps": {},
        "success": True,
    }

    # Step 1: Convert PDF
    convert_result = _convert_pdf(
        pdf_path=pdf_path,
        output_dir=output_dir,
        use_llm=use_llm,
    )
    results["steps"]["convert"] = convert_result

    if not convert_result.get("success"):
        results["success"] = False
        results["message"] = f"PDF conversion failed: {convert_result.get('message')}"
        return results

    markdown_path = convert_result["markdown_path"]
    results["output_dir"] = convert_result.get("output_dir")
    results["markdown_path"] = markdown_path

    # Expose image info at top level for easy access
    if convert_result.get("images_dir"):
        results["images_dir"] = convert_result["images_dir"]
        results["image_count"] = convert_result.get("image_count", 0)

    # Step 2: Index headers
    index_result = _index_markdown(markdown_path=markdown_path)
    results["steps"]["index"] = index_result

    if not index_result.get("success"):
        results["success"] = False
        results["message"] = f"Indexing failed: {index_result.get('message')}"
        return results

    # Step 3: Create embeddings
    embed_result = _embed_document(
        markdown_path=markdown_path,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    results["steps"]["embed"] = embed_result

    if not embed_result.get("success"):
        results["success"] = False
        results["message"] = f"Embedding failed: {embed_result.get('message')}"
        return results

    # Build success message
    image_info = ""
    if results.get("image_count"):
        image_info = f", {results['image_count']} images extracted to {results['images_dir']}"

    results["message"] = (
        f"Successfully processed paper: "
        f"{index_result.get('header_count', 0)} headers indexed, "
        f"{embed_result.get('num_chunks', 0)} chunks embedded{image_info}"
    )

    return results


def main():
    """Run the MCP server."""
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--http":
        mcp.run(transport="streamable-http")
    else:
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
