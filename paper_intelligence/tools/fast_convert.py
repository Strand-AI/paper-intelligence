"""Fast CPU-only PDF conversion path.

Routes text-based PDFs through `pdf-inspector` (pure Rust, ~100ms/paper, no ML
models) and extracts figures with PyMuPDF. Scanned / image-based PDFs, or PDFs
with broken font encodings, are left to the Marker path in `convert.py`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# pdf_type values that the fast path can handle
FAST_TYPES = {"text_based", "mixed"}


def can_fast_convert(pdf_path: str | Path) -> tuple[bool, dict]:
    """Return (usable, info) for the fast path without converting."""
    try:
        import pdf_inspector
    except ImportError:
        return False, {"reason": "pdf-inspector not installed"}

    result = pdf_inspector.classify_pdf(str(pdf_path))
    info = {
        "pdf_type": getattr(result, "pdf_type", None),
        "confidence": getattr(result, "confidence", None),
        "has_encoding_issues": getattr(result, "has_encoding_issues", False),
    }
    usable = (
        info["pdf_type"] in FAST_TYPES
        and not info["has_encoding_issues"]
        and (info["confidence"] is None or info["confidence"] >= 0.6)
    )
    if not usable:
        info.setdefault("reason", f"needs OCR path ({info['pdf_type']})")
    return usable, info


def extract_images(pdf_path: Path, images_dir: Path) -> int:
    """Dump embedded raster figures with PyMuPDF. Returns image count."""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        logger.warning("PyMuPDF not installed — skipping figure extraction")
        return 0

    doc = fitz.open(str(pdf_path))
    count = 0
    try:
        for page_index in range(doc.page_count):
            for img in doc.get_page_images(page_index, full=True):
                xref = img[0]
                pix = fitz.Pixmap(doc, xref)
                if pix.n - pix.alpha >= 4:  # CMYK -> RGB
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                if pix.width < 32 or pix.height < 32:  # skip rules/icons
                    continue
                images_dir.mkdir(parents=True, exist_ok=True)
                pix.save(str(images_dir / f"page{page_index + 1}_{xref}.png"))
                count += 1
    finally:
        doc.close()
    return count


def fast_convert_pdf(pdf_path: str, output_dir: Optional[str] = None) -> dict:
    """Convert a text-based PDF to markdown without Marker.

    Same return shape as `convert_pdf`, plus `engine` and `classification`.
    Returns success=False (with `fallback=True`) when the PDF needs OCR, so the
    caller can hand off to Marker.
    """
    import pdf_inspector

    from ..metadata import write_metadata
    from .convert import get_output_dir, pdf_hash

    pdf_path = Path(pdf_path).expanduser().resolve()
    if not pdf_path.exists():
        return {"success": False, "message": f"PDF file not found: {pdf_path}"}

    usable, info = can_fast_convert(pdf_path)
    if not usable:
        return {
            "success": False,
            "fallback": True,
            "classification": info,
            "message": f"Fast path unsuitable: {info.get('reason')}",
        }

    out_dir = Path(output_dir).expanduser().resolve() if output_dir else get_output_dir(pdf_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        result = pdf_inspector.process_pdf(str(pdf_path))
        markdown_text = result.markdown or ""
        if not markdown_text.strip():
            return {
                "success": False,
                "fallback": True,
                "classification": info,
                "message": "Fast path produced no text",
            }

        md_path = out_dir / "paper.md"
        md_path.write_text(markdown_text, encoding="utf-8")

        images_dir = out_dir / "images"
        image_count = extract_images(pdf_path, images_dir)

        write_metadata(
            paper_dir=out_dir,
            source_pdf=pdf_path.name,
            steps_completed=["convert"],
            extra={
                "image_count": image_count,
                "pdf_hash": pdf_hash(pdf_path),
                "engine": "pdf-inspector",
                "pdf_type": info["pdf_type"],
            },
        )

        return {
            "markdown_path": str(md_path),
            "output_dir": str(out_dir),
            "success": True,
            "engine": "pdf-inspector",
            "classification": info,
            "message": f"Converted {pdf_path.name} with pdf-inspector in "
                       f"{getattr(result, 'processing_time_ms', '?')}ms",
            "images_dir": str(images_dir) if image_count else None,
            "image_count": image_count,
        }
    except Exception as e:  # noqa: BLE001 — fall back rather than fail the pipeline
        return {
            "success": False,
            "fallback": True,
            "classification": info,
            "message": f"Fast conversion failed: {e}",
        }


def convert_pdf_auto(pdf_path: str, output_dir: Optional[str] = None, use_llm: bool = False) -> dict:
    """Try the fast path, fall back to Marker when the PDF needs it."""
    fast = fast_convert_pdf(pdf_path, output_dir)
    if fast.get("success"):
        return fast

    logger.info("Falling back to Marker: %s", fast.get("message"))
    from .convert import convert_pdf

    slow = convert_pdf(pdf_path, output_dir, use_llm=use_llm)
    slow.setdefault("engine", "marker")
    slow["fast_path_skipped_because"] = fast.get("message")
    return slow
