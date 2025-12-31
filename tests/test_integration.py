"""Integration tests using a real PDF file.

These tests exercise the full pipeline: convert, index, embed, search.
They are SKIPPED in CI because they require ML model loading (slow).

Run locally with:
    pytest tests/test_integration.py -v
"""

import shutil
from pathlib import Path

import pytest

# Path to the test PDF
FIXTURES_DIR = Path(__file__).parent / "fixtures"
SAMPLE_PDF = FIXTURES_DIR / "sample.pdf"


@pytest.fixture(scope="module")
def fully_processed_paper(tmp_path_factory):
    """Process PDF once and share across ALL tests in this module.

    This fixture runs the complete pipeline once:
    1. Convert PDF to markdown
    2. Index headers
    3. Create embeddings

    All tests that need a processed paper should use this fixture
    to avoid redundant, slow operations.
    """
    from paper_intelligence.server import process_paper

    output_dir = tmp_path_factory.mktemp("paper")

    result = process_paper(
        pdf_path=str(SAMPLE_PDF),
        output_dir=str(output_dir),
    )

    assert result["success"], f"Pipeline failed: {result.get('message')}"

    return {
        "output_dir": output_dir,
        "markdown_path": Path(result["markdown_path"]),
        "result": result,
    }


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create a temporary output directory for isolated tests."""
    output_dir = tmp_path / "test_paper"
    output_dir.mkdir(parents=True, exist_ok=True)
    yield output_dir


class TestPDFConversion:
    """Tests for PDF to Markdown conversion."""

    def test_convert_pdf_creates_markdown(self, fully_processed_paper):
        """Test that PDF conversion creates a markdown file."""
        md_path = fully_processed_paper["markdown_path"]

        assert md_path.exists(), "Markdown file should be created"
        assert md_path.suffix == ".md", "Output should be a markdown file"

        content = md_path.read_text()
        assert len(content) > 100, "Markdown should have substantial content"

    def test_convert_pdf_creates_output_directory(self, fully_processed_paper):
        """Test that conversion creates the expected output directory."""
        output_dir = fully_processed_paper["output_dir"]

        assert output_dir.exists(), "Output directory should exist"
        assert output_dir.is_dir(), "Output should be a directory"

    def test_convert_pdf_returns_correct_structure(self, fully_processed_paper):
        """Test that conversion returns properly structured result."""
        result = fully_processed_paper["result"]

        assert "markdown_path" in result
        assert "output_dir" in result
        assert "success" in result
        assert result["success"] is True
        assert "message" in result

    def test_convert_pdf_writes_metadata(self, fully_processed_paper):
        """Test that conversion writes metadata file."""
        output_dir = fully_processed_paper["output_dir"]
        metadata_file = output_dir / "metadata.json"

        assert metadata_file.exists(), "Metadata file should be created"

    def test_convert_nonexistent_pdf_fails(self, temp_output_dir):
        """Test that converting a nonexistent PDF fails gracefully."""
        from paper_intelligence.tools.convert import convert_pdf

        result = convert_pdf(
            pdf_path="/nonexistent/path/to/file.pdf",
            output_dir=str(temp_output_dir),
        )

        assert result["success"] is False
        assert "not found" in result["message"].lower()

    def test_convert_non_pdf_fails(self, temp_output_dir):
        """Test that converting a non-PDF file fails gracefully."""
        from paper_intelligence.tools.convert import convert_pdf

        fake_file = temp_output_dir / "not_a_pdf.txt"
        fake_file.write_text("This is not a PDF")

        result = convert_pdf(
            pdf_path=str(fake_file),
            output_dir=str(temp_output_dir),
        )

        assert result["success"] is False
        assert "not a pdf" in result["message"].lower()


class TestMarkdownIndexing:
    """Tests for markdown header indexing."""

    def test_index_creates_file(self, fully_processed_paper):
        """Test that indexing creates an index.json file."""
        output_dir = fully_processed_paper["output_dir"]
        index_path = output_dir / "index.json"

        assert index_path.exists(), "Index file should be created"

    def test_index_result_structure(self, fully_processed_paper):
        """Test that index step returned correct structure."""
        result = fully_processed_paper["result"]

        assert "steps" in result
        assert "index" in result["steps"]
        assert result["steps"]["index"]["success"]
        assert "headers" in result["steps"]["index"]
        assert "header_count" in result["steps"]["index"]

    def test_index_extracts_headers(self, fully_processed_paper):
        """Test that indexing extracts headers from markdown."""
        result = fully_processed_paper["result"]["steps"]["index"]

        assert len(result["headers"]) > 0, "Should extract some headers"
        assert result["header_count"] > 0

    def test_index_header_structure(self, fully_processed_paper):
        """Test that extracted headers have correct structure."""
        result = fully_processed_paper["result"]["steps"]["index"]

        if result["headers"]:
            header = result["headers"][0]
            assert "text" in header
            assert "level" in header
            assert "line" in header
            assert "path" in header

    def test_get_header_context(self, fully_processed_paper):
        """Test getting header context for a specific line."""
        from paper_intelligence.tools.index import get_header_context

        md_path = fully_processed_paper["markdown_path"]
        result = get_header_context(str(md_path), 50)

        assert result["success"]

    def test_search_headers(self, fully_processed_paper):
        """Test searching within headers."""
        from paper_intelligence.tools.index import search_headers

        md_path = fully_processed_paper["markdown_path"]
        result = search_headers(str(md_path), "")

        assert result["success"]
        assert "matches" in result
        assert "match_count" in result


class TestEmbedding:
    """Tests for document embedding."""

    def test_embed_creates_chroma_db(self, fully_processed_paper):
        """Test that embedding creates ChromaDB storage."""
        output_dir = fully_processed_paper["output_dir"]
        db_path = output_dir / "chroma"

        assert db_path.exists(), "ChromaDB directory should be created"

    def test_embed_result_structure(self, fully_processed_paper):
        """Test that embed step returned correct structure."""
        result = fully_processed_paper["result"]

        assert "steps" in result
        assert "embed" in result["steps"]
        assert result["steps"]["embed"]["success"]
        assert "num_chunks" in result["steps"]["embed"]
        assert result["steps"]["embed"]["num_chunks"] > 0

    def test_query_paper(self, fully_processed_paper):
        """Test querying paper embeddings."""
        from paper_intelligence.tools.embed import query_paper

        output_dir = fully_processed_paper["output_dir"]
        result = query_paper(str(output_dir), "technology", top_k=3)

        assert result["success"]
        assert "results" in result
        assert result["num_results"] > 0


class TestSearch:
    """Tests for unified search functionality."""

    def test_grep_search(self, fully_processed_paper):
        """Test grep-style text search."""
        from paper_intelligence.tools.search import search

        output_dir = fully_processed_paper["output_dir"]

        result = search(
            query="the",  # Common word likely in any document
            paper_dirs=[str(output_dir)],
            mode="grep",
            top_k=5,
        )

        assert result["success"]
        assert "results" in result
        assert result["num_results"] > 0

    def test_rag_search(self, fully_processed_paper):
        """Test semantic RAG search."""
        from paper_intelligence.tools.search import search

        output_dir = fully_processed_paper["output_dir"]

        result = search(
            query="technology innovation",
            paper_dirs=[str(output_dir)],
            mode="rag",
            top_k=3,
        )

        assert result["success"]
        assert "results" in result

    def test_hybrid_search(self, fully_processed_paper):
        """Test hybrid grep + RAG search."""
        from paper_intelligence.tools.search import search

        output_dir = fully_processed_paper["output_dir"]

        result = search(
            query="data",
            paper_dirs=[str(output_dir)],
            mode="hybrid",
            top_k=5,
        )

        assert result["success"]
        assert "results" in result
        assert "mode" in result
        assert result["mode"] == "hybrid"

    def test_search_with_regex(self, fully_processed_paper):
        """Test regex search."""
        from paper_intelligence.tools.search import search

        output_dir = fully_processed_paper["output_dir"]

        result = search(
            query=r"\b\w+ing\b",  # Words ending in 'ing'
            paper_dirs=[str(output_dir)],
            mode="grep",
            regex=True,
            top_k=5,
        )

        assert result["success"]

    def test_search_case_sensitive(self, fully_processed_paper):
        """Test case-sensitive search."""
        from paper_intelligence.tools.search import search

        output_dir = fully_processed_paper["output_dir"]

        result = search(
            query="The",
            paper_dirs=[str(output_dir)],
            mode="grep",
            case_sensitive=True,
            top_k=5,
        )

        assert result["success"]


class TestPaperInfo:
    """Tests for paper info retrieval."""

    def test_get_paper_info(self, fully_processed_paper):
        """Test getting info for a fully processed paper."""
        from paper_intelligence.tools.search import get_paper_info

        output_dir = fully_processed_paper["output_dir"]
        result = get_paper_info(str(output_dir))

        assert result["success"]
        assert result["has_markdown"] is True
        assert result["has_index"] is True
        assert result["has_embeddings"] is True
        assert "name" in result
        assert "path" in result
        assert "header_count" in result
        assert "chunk_count" in result

    def test_get_paper_info_nonexistent(self):
        """Test getting info for nonexistent directory."""
        from paper_intelligence.tools.search import get_paper_info

        result = get_paper_info("/nonexistent/path")

        assert result["success"] is False


class TestFullPipeline:
    """Tests for the full process_paper pipeline."""

    def test_pipeline_output_structure(self, fully_processed_paper):
        """Test that processed paper has correct output structure."""
        result = fully_processed_paper["result"]
        output_dir = fully_processed_paper["output_dir"]

        # Check all steps completed
        assert "steps" in result
        assert "convert" in result["steps"]
        assert "index" in result["steps"]
        assert "embed" in result["steps"]

        # Check each step succeeded
        assert result["steps"]["convert"]["success"]
        assert result["steps"]["index"]["success"]
        assert result["steps"]["embed"]["success"]

        # Check result structure
        assert "output_dir" in result
        assert "markdown_path" in result
        assert "message" in result

        # Check output files exist
        assert (output_dir / "paper.md").exists()
        assert (output_dir / "index.json").exists()
        assert (output_dir / "chroma").exists()

        # Message should summarize the processing
        assert "headers" in result["message"].lower() or "chunks" in result["message"].lower()
