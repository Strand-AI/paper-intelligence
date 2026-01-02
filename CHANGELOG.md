# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.2.0] - 2026-01-01

### Changed
- **BREAKING**: Consolidated 6 MCP tools down to 2 for token efficiency (~3000 token savings)
- **BREAKING**: Renamed `paper_dirs` parameter to `sources` in search function
- `search()` now auto-processes PDFs on first use (1-3 minutes, then instant)
- Added `use_llm` parameter to search for enhanced PDF conversion accuracy

### Removed
- `convert_pdf` tool (functionality moved into `search`)
- `index_markdown` tool (functionality moved into `search`)
- `embed_document` tool (functionality moved into `search`)
- `process_paper` tool (functionality moved into `search`)

### Added
- Auto-processing of PDFs when passed directly to search
- Version compatibility checking with auto-reprocessing for incompatible versions
- Processing time warnings in tool descriptions

## [0.1.1] - 2025-12-31

### Added
- Changelog following Keep a Changelog format
- Automatic GitHub Releases with changelog extraction on tag push
- Changelog link in PyPI project metadata
- Integration tests with real PDF processing pipeline

### Changed
- Expanded README with quick start section and installation one-liners
- Added MCP client configuration for Claude Code CLI, VS Code, Cursor, and Windsurf
- Added debugging section with MCP Inspector instructions
- Added troubleshooting section for common issues

## [0.1.0] - 2025-12-31

### Added
- Initial release of paper-intelligence MCP server
- PDF to Markdown conversion using Marker
- Header hierarchy extraction into searchable JSON index
- Document embeddings with ChromaDB and HuggingFace (BAAI/bge-small-en-v1.5)
- GPU acceleration support for Apple Silicon (MPS) and NVIDIA (CUDA)
- Unified search with three modes: grep, RAG, and hybrid
- Self-contained paper directories with all artifacts
- Full processing pipeline via `process_paper` tool
