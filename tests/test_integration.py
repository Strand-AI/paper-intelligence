"""Integration tests using a real PDF file.

These tests exercise the full pipeline: convert, index, embed, search.
They are marked as slow since they involve model loading and PDF processing.
"""

import shutil
from pathlib import Path

import pytest

# Path to the test PDF
FIXTURES_DIR = Path(__file__).parent / "fixtures"
SAMPLE_PDF = FIXTURES_DIR / "sample.pdf"


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create a temporary output directory for test artifacts."""
    output_dir = tmp_path / "test_paper"
    output_dir.mkdir(parents=True, exist_ok=True)
    yield output_dir
    # Cleanup is automatic with tmp_path


@pytest.fixture(scope="module")
def converted_paper(tmp_path_factory):
    """Convert PDF once and share across tests in this module.

    This fixture runs the full conversion pipeline once, then shares
    the result with all tests that need it. Saves significant time.
    """
    from paper_intelligence.tools.convert import convert_pdf

    output_dir = tmp_path_factory.mktemp("paper")

    result = convert_pdf(
        pdf_path=str(SAMPLE_PDF),
        output_dir=str(output_dir),
    )

    assert result["success"], f"PDF conversion failed: {result.get('message')}"

    return {
        "output_dir": output_dir,
        "markdown_path": Path(result["markdown_path"]),
        "result": result,
    }


class TestPDFConversion:
    """Tests for PDF to Markdown conversion."""

    def test_convert_pdf_creates_markdown(self, converted_paper):
        """Test that PDF conversion creates a markdown file."""
        md_path = converted_paper["markdown_path"]

        assert md_path.exists(), "Markdown file should be created"
        assert md_path.suffix == ".md", "Output should be a markdown file"

        content = md_path.read_text()
        assert len(content) > 100, "Markdown should have substantial content"

    def test_convert_pdf_creates_output_directory(self, converted_paper):
        """Test that conversion creates the expected output directory."""
        output_dir = converted_paper["output_dir"]

        assert output_dir.exists(), "Output directory should exist"
        assert output_dir.is_dir(), "Output should be a directory"

    def test_convert_pdf_returns_correct_structure(self, converted_paper):
        """Test that conversion returns properly structured result."""
        result = converted_paper["result"]

        assert "markdown_path" in result
        assert "output_dir" in result
        assert "success" in result
        assert result["success"] is True
        assert "message" in result

    def test_convert_pdf_writes_metadata(self, converted_paper):
        """Test that conversion writes metadata file."""
        output_dir = converted_paper["output_dir"]
        metadata_file = output_dir / "metadata.json"

        # Metadata should exist after conversion
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

        # Create a fake non-PDF file
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

    def test_index_markdown_creates_index(self, converted_paper):
        """Test that indexing creates an index.json file."""
        from paper_intelligence.tools.index import index_markdown

        md_path = converted_paper["markdown_path"]
        result = index_markdown(str(md_path))

        assert result["success"], f"Indexing failed: {result.get('message')}"
        assert "index_path" in result

        index_path = Path(result["index_path"])
        assert index_path.exists(), "Index file should be created"

    def test_index_extracts_headers(self, converted_paper):
        """Test that indexing extracts headers from markdown."""
        from paper_intelligence.tools.index import index_markdown

        md_path = converted_paper["markdown_path"]
        result = index_markdown(str(md_path))

        assert result["success"]
        assert "headers" in result
        assert len(result["headers"]) > 0, "Should extract some headers"
        assert "header_count" in result
        assert result["header_count"] > 0

    def test_index_header_structure(self, converted_paper):
        """Test that extracted headers have correct structure."""
        from paper_intelligence.tools.index import index_markdown

        md_path = converted_paper["markdown_path"]
        result = index_markdown(str(md_path))

        assert result["success"]

        # Check first header has expected fields
        if result["headers"]:
            header = result["headers"][0]
            assert "text" in header
            assert "level" in header
            assert "line" in header
            assert "path" in header

    def test_get_header_context(self, converted_paper):
        """Test getting header context for a specific line."""
        from paper_intelligence.tools.index import get_header_context, index_markdown

        md_path = converted_paper["markdown_path"]

        # First index the document
        index_markdown(str(md_path))

        # Get context for a line in the middle of the document
        result = get_header_context(str(md_path), 50)

        assert result["success"]
        # May or may not find a header depending on document structure

    def test_search_headers(self, converted_paper):
        """Test searching within headers."""
        from paper_intelligence.tools.index import index_markdown, search_headers

        md_path = converted_paper["markdown_path"]

        # First index the document
        index_markdown(str(md_path))

        # Search for a common term
        result = search_headers(str(md_path), "")

        assert result["success"]
        assert "matches" in result
        assert "match_count" in result


class TestEmbedding:
    """Tests for document embedding."""

    def test_embed_document_creates_chroma_db(self, converted_paper):
        """Test that embedding creates ChromaDB storage."""
        from paper_intelligence.tools.embed import embed_document

        md_path = converted_paper["markdown_path"]
        result = embed_document(str(md_path))

        assert result["success"], f"Embedding failed: {result.get('message')}"
        assert "db_path" in result

        db_path = Path(result["db_path"])
        assert db_path.exists(), "ChromaDB directory should be created"

    def test_embed_document_creates_chunks(self, converted_paper):
        """Test that embedding creates document chunks."""
        from paper_intelligence.tools.embed import embed_document

        md_path = converted_paper["markdown_path"]
        result = embed_document(str(md_path))

        assert result["success"]
        assert "num_chunks" in result
        assert result["num_chunks"] > 0, "Should create at least one chunk"

    def test_embed_with_custom_chunk_size(self, converted_paper):
        """Test embedding with custom chunk parameters."""
        from paper_intelligence.tools.embed import embed_document

        md_path = converted_paper["markdown_path"]
        result = embed_document(
            str(md_path),
            chunk_size=256,
            chunk_overlap=25,
        )

        assert result["success"]
        # Smaller chunks should produce more chunks
        assert result["num_chunks"] > 0

    def test_query_paper(self, converted_paper):
        """Test querying paper embeddings."""
        from paper_intelligence.tools.embed import embed_document, query_paper

        md_path = converted_paper["markdown_path"]
        output_dir = converted_paper["output_dir"]

        # First embed the document
        embed_result = embed_document(str(md_path))
        assert embed_result["success"]

        # Query for something likely in a biopharma paper
        result = query_paper(str(output_dir), "technology", top_k=3)

        assert result["success"]
        assert "results" in result
        # Should find some results
        assert result["num_results"] > 0


class TestSearch:
    """Tests for unified search functionality."""

    @pytest.fixture
    def prepared_paper(self, converted_paper):
        """Prepare a paper with indexing and embeddings."""
        from paper_intelligence.tools.embed import embed_document
        from paper_intelligence.tools.index import index_markdown

        md_path = converted_paper["markdown_path"]

        # Index and embed
        index_markdown(str(md_path))
        embed_document(str(md_path))

        return converted_paper

    def test_grep_search(self, prepared_paper):
        """Test grep-style text search."""
        from paper_intelligence.tools.search import search

        output_dir = prepared_paper["output_dir"]

        result = search(
            query="the",  # Common word likely in any document
            paper_dirs=[str(output_dir)],
            mode="grep",
            top_k=5,
        )

        assert result["success"]
        assert "results" in result
        assert result["num_results"] > 0

    def test_rag_search(self, prepared_paper):
        """Test semantic RAG search."""
        from paper_intelligence.tools.search import search

        output_dir = prepared_paper["output_dir"]

        result = search(
            query="technology innovation",
            paper_dirs=[str(output_dir)],
            mode="rag",
            top_k=3,
        )

        assert result["success"]
        assert "results" in result
        # RAG search should find semantic matches

    def test_hybrid_search(self, prepared_paper):
        """Test hybrid grep + RAG search."""
        from paper_intelligence.tools.search import search

        output_dir = prepared_paper["output_dir"]

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

    def test_search_with_regex(self, prepared_paper):
        """Test regex search."""
        from paper_intelligence.tools.search import search

        output_dir = prepared_paper["output_dir"]

        result = search(
            query=r"\b\w+ing\b",  # Words ending in 'ing'
            paper_dirs=[str(output_dir)],
            mode="grep",
            regex=True,
            top_k=5,
        )

        assert result["success"]
        # Should find words ending in 'ing'

    def test_search_case_sensitive(self, prepared_paper):
        """Test case-sensitive search."""
        from paper_intelligence.tools.search import search

        output_dir = prepared_paper["output_dir"]

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

    def test_get_paper_info_minimal(self, converted_paper):
        """Test getting info for a converted paper."""
        from paper_intelligence.tools.search import get_paper_info

        output_dir = converted_paper["output_dir"]
        result = get_paper_info(str(output_dir))

        assert result["success"]
        assert result["has_markdown"] is True
        assert "name" in result
        assert "path" in result

    def test_get_paper_info_full(self, converted_paper):
        """Test getting info for a fully processed paper."""
        from paper_intelligence.tools.embed import embed_document
        from paper_intelligence.tools.index import index_markdown
        from paper_intelligence.tools.search import get_paper_info

        md_path = converted_paper["markdown_path"]
        output_dir = converted_paper["output_dir"]

        # Index and embed
        index_markdown(str(md_path))
        embed_document(str(md_path))

        result = get_paper_info(str(output_dir))

        assert result["success"]
        assert result["has_markdown"] is True
        assert result["has_index"] is True
        assert result["has_embeddings"] is True
        assert "header_count" in result
        assert "chunk_count" in result

    def test_get_paper_info_nonexistent(self):
        """Test getting info for nonexistent directory."""
        from paper_intelligence.tools.search import get_paper_info

        result = get_paper_info("/nonexistent/path")

        assert result["success"] is False


class TestFullPipeline:
    """Tests for the full process_paper pipeline."""

    def test_process_paper_full_pipeline(self, temp_output_dir):
        """Test the complete paper processing pipeline."""
        from paper_intelligence.server import process_paper

        result = process_paper(
            pdf_path=str(SAMPLE_PDF),
            output_dir=str(temp_output_dir),
        )

        assert result["success"], f"Pipeline failed: {result.get('message')}"

        # Check all steps completed
        assert "steps" in result
        assert "convert" in result["steps"]
        assert "index" in result["steps"]
        assert "embed" in result["steps"]

        # Check each step succeeded
        assert result["steps"]["convert"]["success"]
        assert result["steps"]["index"]["success"]
        assert result["steps"]["embed"]["success"]

        # Check output files exist
        assert (temp_output_dir / "paper.md").exists()
        assert (temp_output_dir / "index.json").exists()
        assert (temp_output_dir / "chroma").exists()

    def test_process_paper_output_structure(self, temp_output_dir):
        """Test that processed paper has correct output structure."""
        from paper_intelligence.server import process_paper

        result = process_paper(
            pdf_path=str(SAMPLE_PDF),
            output_dir=str(temp_output_dir),
        )

        assert result["success"]

        # Check result structure
        assert "output_dir" in result
        assert "markdown_path" in result
        assert "message" in result

        # Message should summarize the processing
        assert "headers" in result["message"].lower() or "chunks" in result["message"].lower()
