"""Metadata management for processed paper directories."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from ._version import __version__, get_compatibility_message, is_compatible

METADATA_FILENAME = "metadata.json"


def write_metadata(
    paper_dir: Path,
    source_pdf: Optional[str] = None,
    steps_completed: Optional[list[str]] = None,
    extra: Optional[dict] = None,
) -> dict:
    """Write metadata.json to a paper directory.

    Args:
        paper_dir: Path to the paper directory
        source_pdf: Original PDF filename (optional)
        steps_completed: List of completed steps (e.g., ["convert", "index", "embed"])
        extra: Additional metadata to include

    Returns:
        The metadata dict that was written
    """
    paper_dir = Path(paper_dir)
    metadata_path = paper_dir / METADATA_FILENAME

    # Read existing metadata if present
    existing = {}
    if metadata_path.exists():
        try:
            existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    # Build metadata
    metadata = {
        "paper_intelligence_version": __version__,
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "source_pdf": source_pdf or existing.get("source_pdf"),
        "steps_completed": steps_completed or existing.get("steps_completed", []),
    }

    # Merge extra data
    if extra:
        metadata.update(extra)

    # Write
    paper_dir.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return metadata


def read_metadata(paper_dir: Path) -> Optional[dict]:
    """Read metadata.json from a paper directory.

    Args:
        paper_dir: Path to the paper directory

    Returns:
        Metadata dict if found, None otherwise
    """
    paper_dir = Path(paper_dir)
    metadata_path = paper_dir / METADATA_FILENAME

    if not metadata_path.exists():
        return None

    try:
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def update_metadata_steps(paper_dir: Path, step: str) -> dict:
    """Add a step to the steps_completed list in metadata.

    Args:
        paper_dir: Path to the paper directory
        step: Step name to add (e.g., "convert", "index", "embed")

    Returns:
        Updated metadata dict
    """
    existing = read_metadata(paper_dir) or {}
    steps = existing.get("steps_completed", [])

    if step not in steps:
        steps.append(step)

    return write_metadata(
        paper_dir=paper_dir,
        source_pdf=existing.get("source_pdf"),
        steps_completed=steps,
    )


def check_version_compatibility(paper_dir: Path) -> dict:
    """Check if a paper directory was processed with a compatible version.

    Args:
        paper_dir: Path to the paper directory

    Returns:
        Dict with:
        - is_compatible: bool
        - processed_version: str or None
        - current_version: str
        - message: str or None (warning message if incompatible)
    """
    metadata = read_metadata(paper_dir)

    if metadata is None:
        # No metadata = legacy directory, might need re-processing
        return {
            "is_compatible": True,  # Assume compatible, but flag as unknown
            "processed_version": None,
            "current_version": __version__,
            "message": (
                "No metadata found. This paper may have been processed with an older "
                "version of paper-intelligence. Consider re-processing for best results."
            ),
        }

    processed_version = metadata.get("paper_intelligence_version", "0.0.0")
    compatible = is_compatible(processed_version)
    message = get_compatibility_message(processed_version)

    return {
        "is_compatible": compatible,
        "processed_version": processed_version,
        "current_version": __version__,
        "message": message,
    }
