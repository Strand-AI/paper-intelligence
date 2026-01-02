"""Version information for paper-intelligence."""

from __future__ import annotations

from typing import Optional

__version__ = "0.2.0"

# Version compatibility rules:
# - Major version change = incompatible, must re-process
# - Minor version change = new features, backward compatible
# - Patch version change = bug fixes, fully compatible


def parse_version(version_str: str) -> tuple[int, int, int]:
    """Parse a semver string into (major, minor, patch) tuple."""
    parts = version_str.split(".")
    return (
        int(parts[0]) if len(parts) > 0 else 0,
        int(parts[1]) if len(parts) > 1 else 0,
        int(parts[2]) if len(parts) > 2 else 0,
    )


def is_compatible(processed_version: str, current_version: str = __version__) -> bool:
    """Check if a processed version is compatible with the current version.

    Compatibility rules:
    - Same major version = compatible
    - Different major version = incompatible
    """
    processed = parse_version(processed_version)
    current = parse_version(current_version)

    return processed[0] == current[0]


def get_compatibility_message(processed_version: str) -> str | None:
    """Get a message about version compatibility, or None if compatible."""
    if is_compatible(processed_version):
        return None

    processed = parse_version(processed_version)
    current = parse_version(__version__)

    if processed[0] < current[0]:
        return (
            f"Paper was processed with paper-intelligence v{processed_version} "
            f"(current: v{__version__}). Major version mismatch - consider "
            f"re-processing with process_paper for full compatibility."
        )
    else:
        return (
            f"Paper was processed with a newer version v{processed_version} "
            f"(current: v{__version__}). Some features may not be available."
        )
