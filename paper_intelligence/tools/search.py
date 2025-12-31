"""Unified search tool combining grep and RAG search.

Searches within self-contained paper directories.
"""

import json
import re
from pathlib import Path
from typing import Literal, Optional

from ..utils.chromadb_client import RAGClient
from ..utils.markdown_parser import MarkdownParser


def _find_paper_dirs(search_paths: list[str]) -> list[Path]:
    """Find all paper directories (directories containing paper.md)."""
    paper_dirs = []
    for path_str in search_paths:
        path = Path(path_str).expanduser().resolve()
        if path.is_file() and path.name == "paper.md":
            paper_dirs.append(path.parent)
        elif path.is_dir():
            # Check if this is a paper directory
            if (path / "paper.md").exists():
                paper_dirs.append(path)
            # Also check subdirectories
            for subdir in path.iterdir():
                if subdir.is_dir() and (subdir / "paper.md").exists():
                    paper_dirs.append(subdir)
    return paper_dirs


def grep_search(
    query: str,
    paper_dirs: list[Path],
    case_sensitive: bool = False,
    regex: bool = False,
) -> list[dict]:
    """Perform grep-style text search across paper markdown files.

    Args:
        query: Search query (text or regex pattern)
        paper_dirs: List of paper directories to search
        case_sensitive: Whether search is case sensitive
        regex: Whether query is a regex pattern

    Returns:
        List of matches with file, line number, content, and context
    """
    results = []

    # Compile pattern
    flags = 0 if case_sensitive else re.IGNORECASE
    if regex:
        pattern = re.compile(query, flags)
    else:
        pattern = re.compile(re.escape(query), flags)

    for paper_dir in paper_dirs:
        md_file = paper_dir / "paper.md"
        index_file = paper_dir / "index.json"

        if not md_file.exists():
            continue

        try:
            content = md_file.read_text(encoding="utf-8")
            lines = content.split("\n")

            # Load pre-built index if available, otherwise parse
            header_lookup = {}
            if index_file.exists():
                try:
                    index_data = json.loads(index_file.read_text())
                    for h in index_data.get("flat_headers", []):
                        header_lookup[h["line"]] = h["path"]
                except Exception:
                    pass

            # Fall back to parser if no index
            parser = None
            if not header_lookup:
                parser = MarkdownParser(content, str(md_file))

            for line_num, line in enumerate(lines, start=1):
                if pattern.search(line):
                    # Get header context
                    if header_lookup:
                        # Find nearest header before this line
                        header_context = ""
                        for h_line in sorted(header_lookup.keys(), reverse=True):
                            if h_line <= line_num:
                                header_context = header_lookup[h_line]
                                break
                    else:
                        header = parser.get_header_at_line(line_num)
                        header_context = header.path if header else ""

                    # Get surrounding context
                    context_start = max(0, line_num - 3)
                    context_end = min(len(lines), line_num + 2)
                    context_lines = lines[context_start:context_end]

                    results.append({
                        "paper_dir": str(paper_dir),
                        "paper_name": paper_dir.name,
                        "source": str(md_file),
                        "line_number": line_num,
                        "content": line.strip(),
                        "context": "\n".join(context_lines),
                        "header_context": header_context,
                        "match_type": "grep",
                    })

        except Exception:
            continue

    return results


def rag_search(
    query: str,
    paper_dirs: list[Path],
    top_k: int = 5,
) -> list[dict]:
    """Perform semantic RAG search using embeddings.

    Args:
        query: Search query
        paper_dirs: List of paper directories to search
        top_k: Number of results per paper

    Returns:
        List of matches with content, score, and metadata
    """
    results = []

    for paper_dir in paper_dirs:
        chroma_dir = paper_dir / "chroma"

        if not chroma_dir.exists():
            continue

        try:
            rag_client = RAGClient(persist_directory=chroma_dir)
            raw_results = rag_client.query("paper", query, top_k)

            for r in raw_results:
                results.append({
                    "paper_dir": str(paper_dir),
                    "paper_name": paper_dir.name,
                    "source": r.get("metadata", {}).get("source", str(paper_dir / "paper.md")),
                    "content": r.get("text", ""),
                    "score": r.get("score", 0.0),
                    "header_context": r.get("metadata", {}).get("header_path", ""),
                    "match_type": "rag",
                })

        except Exception:
            continue

    # Sort by score
    results.sort(key=lambda x: x.get("score", 0), reverse=True)
    return results[:top_k]


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

    Args:
        query: Search query (text, regex if regex=True, or semantic query)
        paper_dirs: List of paper directories to search
        mode: Search mode - "grep", "rag", or "hybrid" (default)
        top_k: Number of results to return (default 5)
        case_sensitive: Case sensitivity for grep (default False)
        regex: Treat query as regex pattern for grep (default False)
        include_context: Include surrounding context in results (default True)

    Returns:
        Result with results list, num_results, and success status
    """
    # Find all paper directories
    dirs = _find_paper_dirs(paper_dirs)

    if not dirs:
        return {
            "results": [],
            "num_results": 0,
            "query": query,
            "mode": mode,
            "success": True,
            "message": "No paper directories found",
        }

    results = []

    try:
        if mode in ("grep", "hybrid"):
            grep_results = grep_search(
                query=query,
                paper_dirs=dirs,
                case_sensitive=case_sensitive,
                regex=regex,
            )
            results.extend(grep_results)

        if mode in ("rag", "hybrid"):
            rag_results = rag_search(
                query=query,
                paper_dirs=dirs,
                top_k=top_k,
            )
            results.extend(rag_results)

        # Deduplicate by content similarity
        if mode == "hybrid":
            results = _deduplicate_results(results)

        # Sort results
        if mode == "rag":
            results.sort(key=lambda x: x.get("score", 0), reverse=True)
        elif mode == "hybrid":
            # Prioritize by score if available
            results.sort(
                key=lambda x: (
                    x.get("score", 0) if x.get("match_type") == "rag" else 0.5,
                ),
                reverse=True,
            )

        # Limit results
        results = results[:top_k * 2] if mode == "hybrid" else results[:top_k]

        # Remove context if not requested
        if not include_context:
            for r in results:
                r.pop("context", None)

        return {
            "results": results,
            "num_results": len(results),
            "papers_searched": len(dirs),
            "query": query,
            "mode": mode,
            "success": True,
        }

    except Exception as e:
        return {
            "results": [],
            "num_results": 0,
            "query": query,
            "mode": mode,
            "success": False,
            "message": f"Search failed: {str(e)}",
        }


def _deduplicate_results(results: list[dict]) -> list[dict]:
    """Deduplicate results based on content similarity."""
    seen_content = set()
    unique_results = []

    for r in results:
        content = r.get("content", "").strip().lower()[:200]
        if content not in seen_content:
            seen_content.add(content)
            unique_results.append(r)

    return unique_results


def get_paper_info(paper_dir: str) -> dict:
    """Get information about a paper directory.

    Args:
        paper_dir: Path to the paper directory

    Returns:
        Dictionary with paper information including:
        - Processing status (has_markdown, has_index, has_embeddings, has_images)
        - Version compatibility info
        - Image paths (if images exist)
        - Metadata from processing
    """
    from ..metadata import check_version_compatibility, read_metadata

    path = Path(paper_dir).expanduser().resolve()

    if not path.is_dir():
        return {
            "success": False,
            "message": f"Not a directory: {paper_dir}",
        }

    info = {
        "name": path.name,
        "path": str(path),
        "has_markdown": (path / "paper.md").exists(),
        "has_index": (path / "index.json").exists(),
        "has_embeddings": (path / "chroma").exists(),
        "has_images": (path / "images").exists(),
        "success": True,
    }

    # Add paths for easy access
    if info["has_markdown"]:
        info["markdown_path"] = str(path / "paper.md")
    if info["has_images"]:
        images_dir = path / "images"
        info["images_dir"] = str(images_dir)
        # List image files for convenience
        try:
            image_files = [f.name for f in images_dir.iterdir() if f.is_file()]
            info["image_files"] = image_files
            info["image_count"] = len(image_files)
        except Exception:
            info["image_files"] = []
            info["image_count"] = 0

    # Get header count from index
    if info["has_index"]:
        try:
            index_data = json.loads((path / "index.json").read_text())
            info["header_count"] = len(index_data.get("flat_headers", []))
        except Exception:
            info["header_count"] = 0

    # Get chunk count from ChromaDB
    if info["has_embeddings"]:
        try:
            rag_client = RAGClient(persist_directory=path / "chroma")
            info["chunk_count"] = rag_client.get_collection_count("paper")
        except Exception:
            info["chunk_count"] = 0

    # Check version compatibility
    version_info = check_version_compatibility(path)
    info["version"] = {
        "processed_version": version_info["processed_version"],
        "current_version": version_info["current_version"],
        "is_compatible": version_info["is_compatible"],
    }
    if version_info["message"]:
        info["version_warning"] = version_info["message"]

    # Include metadata if available
    metadata = read_metadata(path)
    if metadata:
        info["metadata"] = {
            "source_pdf": metadata.get("source_pdf"),
            "processed_at": metadata.get("processed_at"),
            "steps_completed": metadata.get("steps_completed", []),
        }

    return info
