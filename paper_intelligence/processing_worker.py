"""Detached worker for first-use PDF conversion and indexing."""

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--paper-dir", required=True)
    parser.add_argument("--use-llm", action="store_true")
    args = parser.parse_args()

    # Import only in the child. Importing the processing stack in the MCP server can
    # make the request loop unresponsive while Marker, Torch, and embedding models load.
    from .tools.search import _run_processing_job

    _run_processing_job(
        Path(args.source),
        Path(args.paper_dir),
        use_llm=args.use_llm,
    )


if __name__ == "__main__":
    main()
