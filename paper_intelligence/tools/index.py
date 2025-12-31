"""Markdown indexing tool for header extraction."""

import json
from pathlib import Path
from typing import Optional

from ..utils.markdown_parser import MarkdownParser


def index_markdown(
    markdown_path: str,
    output_path: Optional[str] = None,
) -> dict:
    """Extract header hierarchy from a markdown file into a searchable JSON index.

    Args:
        markdown_path: Path to the markdown file
        output_path: Path for the index JSON (defaults to same dir as markdown)

    Returns:
        Dictionary with:
        - index_path: Path to the generated index file
        - headers: List of extracted headers with hierarchy
        - success: Boolean
        - message: Status message
    """
    md_path = Path(markdown_path)

    if not md_path.exists():
        return {
            "index_path": None,
            "headers": [],
            "success": False,
            "message": f"Markdown file not found: {markdown_path}",
        }

    try:
        # Parse markdown and build index
        parser = MarkdownParser.from_file(md_path)
        index = parser.build_index()

        # Determine output path
        if output_path:
            idx_path = Path(output_path)
        else:
            idx_path = md_path.parent / "index.json"

        idx_path.parent.mkdir(parents=True, exist_ok=True)

        # Write index
        idx_path.write_text(index.to_json(), encoding="utf-8")

        # Update metadata
        from ..metadata import update_metadata_steps

        update_metadata_steps(md_path.parent, "index")

        # Return summary
        flat_headers = [h.to_dict() for h in index.flat_headers]

        return {
            "index_path": str(idx_path),
            "headers": flat_headers,
            "header_count": len(flat_headers),
            "success": True,
            "message": f"Successfully indexed {len(flat_headers)} headers",
        }

    except Exception as e:
        return {
            "index_path": None,
            "headers": [],
            "success": False,
            "message": f"Indexing failed: {str(e)}",
        }


def get_header_context(
    markdown_path: str,
    line_number: int,
) -> dict:
    """Get the header context for a specific line in a markdown file.

    Args:
        markdown_path: Path to the markdown file
        line_number: Line number to get context for

    Returns:
        Dictionary with header information
    """
    md_path = Path(markdown_path)

    if not md_path.exists():
        return {
            "header": None,
            "path": None,
            "success": False,
            "message": f"Markdown file not found: {markdown_path}",
        }

    try:
        parser = MarkdownParser.from_file(md_path)
        header = parser.get_header_at_line(line_number)

        if header:
            return {
                "header": header.text,
                "path": header.path,
                "level": header.level,
                "header_line": header.line,
                "success": True,
            }
        else:
            return {
                "header": None,
                "path": None,
                "success": True,
                "message": "No header found before this line",
            }

    except Exception as e:
        return {
            "header": None,
            "path": None,
            "success": False,
            "message": f"Failed to get header context: {str(e)}",
        }


def search_headers(
    markdown_path: str,
    query: str,
    case_sensitive: bool = False,
) -> dict:
    """Search headers in a markdown file.

    Args:
        markdown_path: Path to the markdown file
        query: Search query
        case_sensitive: Whether search is case sensitive

    Returns:
        Dictionary with matching headers
    """
    md_path = Path(markdown_path)

    if not md_path.exists():
        return {
            "matches": [],
            "success": False,
            "message": f"Markdown file not found: {markdown_path}",
        }

    try:
        parser = MarkdownParser.from_file(md_path)
        flat_headers = parser.build_flat_headers()

        matches = []
        search_query = query if case_sensitive else query.lower()

        for header in flat_headers:
            header_text = header.text if case_sensitive else header.text.lower()
            header_path = header.path if case_sensitive else header.path.lower()

            if search_query in header_text or search_query in header_path:
                matches.append({
                    "text": header.text,
                    "path": header.path,
                    "level": header.level,
                    "line": header.line,
                })

        return {
            "matches": matches,
            "match_count": len(matches),
            "success": True,
        }

    except Exception as e:
        return {
            "matches": [],
            "success": False,
            "message": f"Header search failed: {str(e)}",
        }
