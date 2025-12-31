"""Tests for the markdown parser utility."""

import pytest

from paper_intelligence.utils.markdown_parser import MarkdownParser


SAMPLE_MARKDOWN = """# Introduction

This is the introduction section.

## Background

Some background information.

### Previous Work

Details about previous work.

## Methods

Description of methods.

# Results

The results section.

## Analysis

Analysis of results.
"""


def test_extract_headers():
    """Test header extraction."""
    parser = MarkdownParser(SAMPLE_MARKDOWN)
    headers = parser.extract_headers()

    assert len(headers) == 6
    assert headers[0] == (1, 1, "Introduction")
    assert headers[1] == (2, 5, "Background")
    assert headers[2] == (3, 9, "Previous Work")


def test_build_flat_headers():
    """Test flat header building with paths."""
    parser = MarkdownParser(SAMPLE_MARKDOWN)
    flat = parser.build_flat_headers()

    assert len(flat) == 6

    # Check path building
    assert flat[0].path == "Introduction"
    assert flat[1].path == "Introduction > Background"
    assert flat[2].path == "Introduction > Background > Previous Work"
    assert flat[3].path == "Introduction > Methods"
    assert flat[4].path == "Results"
    assert flat[5].path == "Results > Analysis"


def test_build_header_tree():
    """Test hierarchical header tree building."""
    parser = MarkdownParser(SAMPLE_MARKDOWN)
    tree = parser.build_header_tree()

    assert len(tree) == 2  # Two top-level headers
    assert tree[0].text == "Introduction"
    assert len(tree[0].children) == 2  # Background and Methods

    background = tree[0].children[0]
    assert background.text == "Background"
    assert len(background.children) == 1  # Previous Work


def test_get_header_at_line():
    """Test getting header context for a line."""
    parser = MarkdownParser(SAMPLE_MARKDOWN)

    # Line 7 should be under Background
    header = parser.get_header_at_line(7)
    assert header is not None
    assert header.text == "Background"
    assert header.path == "Introduction > Background"

    # Line 11 should be under Previous Work
    header = parser.get_header_at_line(11)
    assert header is not None
    assert header.text == "Previous Work"


def test_build_index():
    """Test complete index building."""
    parser = MarkdownParser(SAMPLE_MARKDOWN, "test.md")
    index = parser.build_index()

    assert index.source == "test.md"
    assert len(index.headers) == 2
    assert len(index.flat_headers) == 6

    # Test JSON serialization
    json_str = index.to_json()
    assert "Introduction" in json_str
    assert "test.md" in json_str


def test_chunk_text():
    """Test text chunking."""
    parser = MarkdownParser(SAMPLE_MARKDOWN)
    chunks = parser.chunk_text(chunk_size=100, chunk_overlap=10)

    assert len(chunks) > 0
    for chunk in chunks:
        assert "text" in chunk
        assert "header_path" in chunk
        assert "start_line" in chunk
        assert "end_line" in chunk


def test_empty_document():
    """Test handling of empty document."""
    parser = MarkdownParser("")

    assert parser.extract_headers() == []
    assert parser.build_flat_headers() == []
    assert parser.build_header_tree() == []


def test_no_headers():
    """Test document with no headers."""
    content = "Just some text without any headers."
    parser = MarkdownParser(content)

    assert parser.extract_headers() == []

    # Chunking should still work
    chunks = parser.chunk_text()
    assert len(chunks) == 1
    assert "Just some text" in chunks[0]["text"]
