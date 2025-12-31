"""Markdown parsing utilities for header extraction and text chunking."""

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class Header:
    """Represents a markdown header with hierarchy information."""
    level: int
    text: str
    line: int
    path: list[str]
    children: list["Header"] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "level": self.level,
            "text": self.text,
            "line": self.line,
            "path": self.path,
            "children": [child.to_dict() for child in self.children],
        }


@dataclass
class FlatHeader:
    """Flattened header representation for easy searching."""
    level: int
    text: str
    line: int
    path: str  # "Section > Subsection > Subsubsection"

    def to_dict(self) -> dict:
        return {
            "level": self.level,
            "text": self.text,
            "line": self.line,
            "path": self.path,
        }


@dataclass
class MarkdownIndex:
    """Complete markdown document index."""
    source: str
    generated_at: str
    headers: list[Header]
    flat_headers: list[FlatHeader]

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "generated_at": self.generated_at,
            "headers": [h.to_dict() for h in self.headers],
            "flat_headers": [h.to_dict() for h in self.flat_headers],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


class MarkdownParser:
    """Parser for extracting structure from markdown documents."""

    HEADER_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$")

    def __init__(self, content: str, source_path: Optional[str] = None):
        self.content = content
        self.source_path = source_path or "unknown"
        self.lines = content.split("\n")

    @classmethod
    def from_file(cls, path: Path | str) -> "MarkdownParser":
        """Create parser from a markdown file."""
        path = Path(path)
        content = path.read_text(encoding="utf-8")
        return cls(content, str(path))

    def extract_headers(self) -> list[tuple[int, int, str]]:
        """Extract all headers with their level and line number.

        Returns:
            List of (level, line_number, text) tuples
        """
        headers = []
        for line_num, line in enumerate(self.lines, start=1):
            match = self.HEADER_PATTERN.match(line.strip())
            if match:
                level = len(match.group(1))
                text = match.group(2).strip()
                headers.append((level, line_num, text))
        return headers

    def build_header_tree(self) -> list[Header]:
        """Build a hierarchical tree of headers."""
        raw_headers = self.extract_headers()
        if not raw_headers:
            return []

        root_headers: list[Header] = []
        stack: list[Header] = []  # Stack to track parent hierarchy

        for level, line, text in raw_headers:
            # Build path based on current stack
            # Pop stack until we find a parent with lower level
            while stack and stack[-1].level >= level:
                stack.pop()

            path = [h.text for h in stack] + [text]
            header = Header(level=level, text=text, line=line, path=path)

            if stack:
                stack[-1].children.append(header)
            else:
                root_headers.append(header)

            stack.append(header)

        return root_headers

    def build_flat_headers(self) -> list[FlatHeader]:
        """Build a flat list of headers with path strings."""
        raw_headers = self.extract_headers()
        if not raw_headers:
            return []

        flat_headers = []
        path_stack: list[tuple[int, str]] = []  # (level, text)

        for level, line, text in raw_headers:
            # Pop stack until we find a parent with lower level
            while path_stack and path_stack[-1][0] >= level:
                path_stack.pop()

            path_stack.append((level, text))
            path_str = " > ".join(item[1] for item in path_stack)

            flat_headers.append(FlatHeader(
                level=level,
                text=text,
                line=line,
                path=path_str,
            ))

        return flat_headers

    def build_index(self) -> MarkdownIndex:
        """Build complete document index."""
        return MarkdownIndex(
            source=self.source_path,
            generated_at=datetime.utcnow().isoformat() + "Z",
            headers=self.build_header_tree(),
            flat_headers=self.build_flat_headers(),
        )

    def get_header_at_line(self, line_number: int) -> Optional[FlatHeader]:
        """Get the nearest header before or at a given line number."""
        flat_headers = self.build_flat_headers()
        result = None
        for header in flat_headers:
            if header.line <= line_number:
                result = header
            else:
                break
        return result

    def chunk_by_headers(
        self,
        min_chunk_size: int = 100,
        max_chunk_size: int = 1000,
    ) -> list[dict]:
        """Split document into chunks based on header boundaries.

        Returns:
            List of dicts with 'text', 'header_path', 'start_line', 'end_line'
        """
        flat_headers = self.build_flat_headers()
        chunks = []

        if not flat_headers:
            # No headers, treat entire document as one chunk
            return [{
                "text": self.content,
                "header_path": "",
                "start_line": 1,
                "end_line": len(self.lines),
            }]

        # Add sections between headers
        for i, header in enumerate(flat_headers):
            start_line = header.line
            if i + 1 < len(flat_headers):
                end_line = flat_headers[i + 1].line - 1
            else:
                end_line = len(self.lines)

            section_lines = self.lines[start_line - 1:end_line]
            section_text = "\n".join(section_lines)

            if len(section_text) >= min_chunk_size:
                chunks.append({
                    "text": section_text,
                    "header_path": header.path,
                    "start_line": start_line,
                    "end_line": end_line,
                })

        return chunks

    def chunk_text(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ) -> list[dict]:
        """Split document into fixed-size chunks with overlap.

        Respects paragraph boundaries where possible.

        Returns:
            List of dicts with 'text', 'header_path', 'start_line', 'end_line', 'chunk_index'
        """
        chunks = []
        flat_headers = self.build_flat_headers()

        # Split into paragraphs first
        paragraphs = []
        current_para = []
        current_start = 1

        for i, line in enumerate(self.lines, start=1):
            if line.strip() == "":
                if current_para:
                    paragraphs.append({
                        "text": "\n".join(current_para),
                        "start_line": current_start,
                        "end_line": i - 1,
                    })
                    current_para = []
                current_start = i + 1
            else:
                current_para.append(line)

        if current_para:
            paragraphs.append({
                "text": "\n".join(current_para),
                "start_line": current_start,
                "end_line": len(self.lines),
            })

        # Combine paragraphs into chunks
        current_chunk_text = ""
        current_chunk_start = 1
        chunk_index = 0

        for para in paragraphs:
            if len(current_chunk_text) + len(para["text"]) <= chunk_size:
                if current_chunk_text:
                    current_chunk_text += "\n\n" + para["text"]
                else:
                    current_chunk_text = para["text"]
                    current_chunk_start = para["start_line"]
            else:
                if current_chunk_text:
                    # Get header context
                    header_path = ""
                    for h in flat_headers:
                        if h.line <= current_chunk_start:
                            header_path = h.path
                        else:
                            break

                    chunks.append({
                        "text": current_chunk_text,
                        "header_path": header_path,
                        "start_line": current_chunk_start,
                        "end_line": para["start_line"] - 1,
                        "chunk_index": chunk_index,
                    })
                    chunk_index += 1

                    # Overlap: keep last portion
                    if chunk_overlap > 0 and len(current_chunk_text) > chunk_overlap:
                        current_chunk_text = current_chunk_text[-chunk_overlap:] + "\n\n" + para["text"]
                    else:
                        current_chunk_text = para["text"]
                    current_chunk_start = para["start_line"]
                else:
                    current_chunk_text = para["text"]
                    current_chunk_start = para["start_line"]

        # Don't forget the last chunk
        if current_chunk_text:
            header_path = ""
            for h in flat_headers:
                if h.line <= current_chunk_start:
                    header_path = h.path
                else:
                    break

            chunks.append({
                "text": current_chunk_text,
                "header_path": header_path,
                "start_line": current_chunk_start,
                "end_line": len(self.lines),
                "chunk_index": chunk_index,
            })

        return chunks
