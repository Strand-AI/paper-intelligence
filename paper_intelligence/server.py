"""Paper Intelligence MCP Server.

Provides AI agents with efficient, searchable access to PDF documents.
First-use PDF processing runs in the background (normally 1-3 minutes), so
search returns promptly with observable status instead of exceeding MCP deadlines.
"""

from typing import Literal

from mcp.server.fastmcp import FastMCP

# Initialize MCP server
mcp = FastMCP(
    "paper-intelligence",
    instructions=(
        "Search PDF documents efficiently. "
        "Pass PDF paths directly to search. First use queues 1-3 minute background "
        "processing and returns status='processing'; call get_paper_info on the returned "
        "paper_dir until status='ready', then retry search. Searches inspect only the "
        "sources explicitly supplied."
    ),
)


@mcp.tool()
def search(
    query: str,
    sources: list[str],
    mode: Literal["grep", "rag", "hybrid"] = "hybrid",
    top_k: int = 5,
    case_sensitive: bool = False,
    regex: bool = False,
    include_context: bool = True,
    use_llm: bool = False,
) -> dict:
    """Search PDF documents and paper directories.

    First use queues background processing (normally 1-3 minutes) and returns
    status="processing" promptly. Processing continues after the tool call returns.
    Call get_paper_info with the returned paper_dir and retry when status="ready".

    Args:
        query: Search query (text, regex pattern, or semantic query)
        sources: PDF paths or paper directories to search
        mode: "grep" (exact/regex), "rag" (semantic), or "hybrid" (both)
        top_k: Max results to return
        case_sensitive: Case sensitivity for grep
        regex: Treat query as regex for grep
        include_context: Include surrounding lines in results
        use_llm: Use LLM for better PDF conversion (slower)

    Returns:
        Search results with content, location, and relevance scores
    """
    from .tools.search import search as _search

    # Search is deliberately local and bounded. The previous unconditional whole-
    # library R2 pull could block every query for five minutes, even when all requested
    # sources were already indexed.
    return _search(
        query=query,
        sources=sources,
        mode=mode,
        top_k=top_k,
        case_sensitive=case_sensitive,
        regex=regex,
        include_context=include_context,
        use_llm=use_llm,
    )


@mcp.tool()
def get_paper_info(paper_dir: str) -> dict:
    """Check processing status without loading the embedding model.

    Args:
        paper_dir: Paper directory, or the original PDF path

    Returns:
        Processing status and metadata
    """
    from .tools.search import get_paper_info as _get_paper_info

    return _get_paper_info(paper_dir=paper_dir)


def main():
    """Run the MCP server."""
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--http":
        mcp.run(transport="streamable-http")
    else:
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
