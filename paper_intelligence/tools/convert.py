"""PDF to Markdown conversion tool using Marker."""

import hashlib
import os
from pathlib import Path
from typing import Optional

# Set device preference for Apple Silicon MPS
if not os.environ.get("TORCH_DEVICE"):
    try:
        import torch
        if torch.backends.mps.is_available():
            os.environ["TORCH_DEVICE"] = "mps"
        elif torch.cuda.is_available():
            os.environ["TORCH_DEVICE"] = "cuda"
    except ImportError:
        pass


def pdf_hash(pdf_path: Path) -> str:
    """Compute SHA-256 hash of a PDF file."""
    h = hashlib.sha256()
    with open(pdf_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def find_duplicate(pdf_path: str | Path, papers_dir: str | Path | None = None) -> Optional[str]:
    """Check if a PDF already exists in the papers library by content hash.

    Args:
        pdf_path: Path to the PDF to check
        papers_dir: Directory containing paper subdirectories (default: parent of pdf_path)

    Returns:
        Name of the existing paper if duplicate found, None otherwise
    """
    from ..metadata import read_metadata

    pdf_path = Path(pdf_path).expanduser().resolve()
    if not pdf_path.exists():
        return None

    new_hash = pdf_hash(pdf_path)

    if papers_dir is None:
        papers_dir = pdf_path.parent
    papers_dir = Path(papers_dir).expanduser().resolve()

    for paper_dir in papers_dir.iterdir():
        if not paper_dir.is_dir():
            continue
        meta = read_metadata(paper_dir)
        if meta and meta.get("pdf_hash") == new_hash:
            return paper_dir.name

    return None


def get_output_dir(pdf_path: Path) -> Path:
    """Generate output directory for a PDF file.

    Creates a sibling directory with the same name as the PDF.
    e.g., ~/Downloads/paper.pdf -> ~/Downloads/paper/
    """
    return pdf_path.parent / pdf_path.stem


def convert_pdf(
    pdf_path: str,
    output_dir: Optional[str] = None,
    use_llm: bool = False,
) -> dict:
    """Convert a PDF file to Markdown using Marker.

    Args:
        pdf_path: Path to the PDF file
        output_dir: Optional output directory (defaults to sibling dir of PDF)
        use_llm: Use LLM for enhanced accuracy (requires API key)

    Returns:
        Dictionary with:
        - markdown_path: Path to the generated markdown file
        - output_dir: Path to the output directory
        - success: Boolean indicating success
        - message: Status message
        - images_dir: Path to extracted images (if any)
    """
    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict
    from marker.output import text_from_rendered

    pdf_path = Path(pdf_path).expanduser().resolve()

    if not pdf_path.exists():
        return {
            "markdown_path": None,
            "output_dir": None,
            "success": False,
            "message": f"PDF file not found: {pdf_path}",
        }

    if not pdf_path.suffix.lower() == ".pdf":
        return {
            "markdown_path": None,
            "output_dir": None,
            "success": False,
            "message": f"File is not a PDF: {pdf_path}",
        }

    # Determine output directory (sibling of PDF by default)
    if output_dir:
        out_dir = Path(output_dir).expanduser().resolve()
    else:
        out_dir = get_output_dir(pdf_path)

    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Build configuration
        config = {
            "output_format": "markdown",
        }

        if use_llm:
            config["use_llm"] = True

        # Create converter
        converter = PdfConverter(
            artifact_dict=create_model_dict(),
        )

        # Convert
        rendered = converter(str(pdf_path))
        markdown_text, _, images = text_from_rendered(rendered)

        # Write markdown
        md_path = out_dir / "paper.md"
        md_path.write_text(markdown_text, encoding="utf-8")

        # Save images if any
        images_dir = None
        if images:
            images_dir = out_dir / "images"
            images_dir.mkdir(exist_ok=True)
            for img_name, img_data in images.items():
                img_path = images_dir / img_name
                if isinstance(img_data, bytes):
                    img_path.write_bytes(img_data)
                else:
                    # PIL Image
                    img_data.save(str(img_path))

        # Write metadata
        from ..metadata import write_metadata

        image_count = len(images) if images else 0
        content_hash = pdf_hash(pdf_path)
        write_metadata(
            paper_dir=out_dir,
            source_pdf=pdf_path.name,
            steps_completed=["convert"],
            extra={"image_count": image_count, "pdf_hash": content_hash},
        )

        return {
            "markdown_path": str(md_path),
            "output_dir": str(out_dir),
            "success": True,
            "message": f"Successfully converted {pdf_path.name} to markdown",
            "images_dir": str(images_dir) if images_dir else None,
            "image_count": image_count,
        }

    except Exception as e:
        return {
            "markdown_path": None,
            "output_dir": None,
            "success": False,
            "message": f"Conversion failed: {str(e)}",
        }
