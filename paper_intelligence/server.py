"""Paper Intelligence MCP Server.

Provides AI agents with efficient, searchable access to PDF documents.
PDFs are automatically processed on first search (1-3 minutes), then
all subsequent searches are instant.
"""

from typing import Literal

from mcp.server.fastmcp import FastMCP

# Initialize MCP server
mcp = FastMCP(
    "paper-intelligence",
    instructions=(
        "Search PDF documents efficiently. "
        "Pass PDF paths directly to search - they're auto-processed on first use "
        "(takes 1-3 minutes, then instant). "
        "Use get_paper_info to check processing status."
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

    PDFs are automatically processed on first use (1-3 minutes per PDF).
    Subsequent searches are instant.

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
    """Check processing status of a paper directory.

    Args:
        paper_dir: Path to paper directory

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
