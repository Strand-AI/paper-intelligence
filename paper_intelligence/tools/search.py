"""Unified search tool combining grep and RAG search.

Searches within self-contained paper directories.
Auto-processes PDFs and paper directories as needed.
"""

import json
import os
import re
import sqlite3
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

from ..metadata import check_version_compatibility
from ..utils.markdown_parser import MarkdownParser

PROCESSING_STATE_FILENAME = ".paper-intelligence-processing.json"
PROCESSING_RETRY_AFTER_SECONDS = 30
PROCESSING_ETA = "usually 1-3 minutes"
MAX_SEMANTIC_PAPERS = 20
_PROCESSING_JOBS: dict[str, subprocess.Popen] = {}
_PROCESSING_LOCK = threading.Lock()


def _process_paper_if_needed(
    path: Path,
    use_llm: bool = False,
) -> tuple[Optional[Path], Optional[str]]:
    """Process a PDF or ensure a paper directory is fully processed.

    Args:
        path: Path to PDF file or paper directory
        use_llm: Use LLM for PDF conversion (if processing needed)

    Returns:
        Tuple of (paper_dir, error_message). If error, paper_dir is None.
    """
    from .convert import convert_pdf, get_output_dir
    from .embed import embed_document
    from .index import index_markdown

    # Handle PDF files
    if path.is_file() and path.suffix.lower() == ".pdf":
        # A direct PDF source is intentionally isolated from sibling content. Looking
        # for duplicates by scanning its parent made /tmp PDFs inspect unrelated temp
        # directories (and could hit permission errors).
        paper_dir = get_output_dir(path)

        # Check if already processed and compatible
        if (paper_dir / "paper.md").exists():
            version_info = check_version_compatibility(paper_dir)
            if version_info["is_compatible"]:
                # Already processed and compatible, ensure all steps complete
                return _ensure_fully_processed(paper_dir)
            # Version incompatible, re-process
            # (Fall through to convert)

        # Convert PDF
        result = convert_pdf(str(path), use_llm=use_llm)
        if not result.get("success"):
            return None, f"PDF conversion failed: {result.get('message')}"
        paper_dir = Path(result["output_dir"])

        # Index
        result = index_markdown(str(paper_dir / "paper.md"))
        if not result.get("success"):
            return None, f"Indexing failed: {result.get('message')}"

        # Embed
        result = embed_document(str(paper_dir / "paper.md"))
        if not result.get("success"):
            return None, f"Embedding failed: {result.get('message')}"

        return paper_dir, None

    # Handle paper directories
    elif path.is_dir():
        if (path / "paper.md").exists():
            # Check version compatibility
            version_info = check_version_compatibility(path)
            if not version_info["is_compatible"]:
                return None, (
                    f"Paper directory {path} was processed with incompatible version "
                    f"{version_info['processed_version']}. Please re-process the original PDF."
                )
            return _ensure_fully_processed(path)
        return None, f"Not a paper directory (no paper.md): {path}"

    return None, f"Not a PDF file or paper directory: {path}"


def _ensure_fully_processed(paper_dir: Path) -> tuple[Optional[Path], Optional[str]]:
    """Ensure a paper directory has index and embeddings.

    Args:
        paper_dir: Path to paper directory with paper.md

    Returns:
        Tuple of (paper_dir, error_message). If error, paper_dir is None.
    """
    from .embed import embed_document
    from .index import index_markdown

    md_path = paper_dir / "paper.md"

    # Index if missing
    if not (paper_dir / "index.json").exists():
        result = index_markdown(str(md_path))
        if not result.get("success"):
            return None, f"Indexing failed: {result.get('message')}"

    # Embed if missing
    if not (paper_dir / "chroma").exists():
        result = embed_document(str(md_path))
        if not result.get("success"):
            return None, f"Embedding failed: {result.get('message')}"

    return paper_dir, None


def _processing_state_path(paper_dir: Path) -> Path:
    return paper_dir / PROCESSING_STATE_FILENAME


def _read_processing_state(paper_dir: Path) -> Optional[dict]:
    state_path = _processing_state_path(paper_dir)
    try:
        return json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _write_processing_state(paper_dir: Path, state: dict) -> None:
    paper_dir.mkdir(parents=True, exist_ok=True)
    state_path = _processing_state_path(paper_dir)
    temporary_path = state_path.with_name(
        f".{state_path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    temporary_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    temporary_path.replace(state_path)


def _is_fully_processed(paper_dir: Path) -> bool:
    artifacts_exist = all(
        (paper_dir / name).exists() for name in ("paper.md", "index.json", "chroma")
    )
    return artifacts_exist and check_version_compatibility(paper_dir)["is_compatible"]


def _run_processing_job(source: Path, paper_dir: Path, use_llm: bool) -> None:
    started_at = datetime.now(timezone.utc).isoformat()
    _write_processing_state(paper_dir, {
        "status": "processing",
        "source": str(source),
        "paper_dir": str(paper_dir),
        "started_at": started_at,
        "worker_pid": os.getpid(),
        "estimated_completion": PROCESSING_ETA,
        "retry_after_seconds": PROCESSING_RETRY_AFTER_SECONDS,
        "message": "PDF processing is running in a detached background process.",
    })

    try:
        if source.is_file() and source.name == "paper.md":
            processed_dir, error = _process_paper_if_needed(source.parent, use_llm)
        else:
            processed_dir, error = _process_paper_if_needed(source, use_llm)
    except Exception as exc:
        processed_dir, error = None, f"Background processing failed: {exc}"

    completed_at = datetime.now(timezone.utc).isoformat()
    if processed_dir:
        _write_processing_state(paper_dir, {
            "status": "completed",
            "source": str(source),
            "paper_dir": str(processed_dir),
            "started_at": started_at,
            "completed_at": completed_at,
            "message": "Processing completed. Retry search with the same source.",
        })
    else:
        _write_processing_state(paper_dir, {
            "status": "failed",
            "source": str(source),
            "paper_dir": str(paper_dir),
            "started_at": started_at,
            "completed_at": completed_at,
            "message": error or "Paper processing failed.",
        })


def _pid_is_running(pid: object) -> bool:
    try:
        os.kill(int(pid), 0)
    except (OSError, TypeError, ValueError):
        return False
    return True


def _processing_is_active(key: str, state: Optional[dict]) -> bool:
    process = _PROCESSING_JOBS.get(key)
    if process is not None and process.poll() is None:
        return True
    return bool(
        state
        and state.get("status") in {"queued", "processing"}
        and _pid_is_running(state.get("worker_pid"))
    )


def _start_processing_process(source: Path, paper_dir: Path, use_llm: bool) -> subprocess.Popen:
    """Launch heavy PDF work outside the MCP process.

    Marker and Torch initialization can monopolize an in-process worker long enough
    that even the original tool response and status calls miss the MCP deadline.
    A detached interpreter keeps the MCP request loop responsive and lets processing
    survive a client timeout or server reconnect.
    """
    command = [
        sys.executable,
        "-m",
        "paper_intelligence.processing_worker",
        "--source",
        str(source),
        "--paper-dir",
        str(paper_dir),
    ]
    if use_llm:
        command.append("--use-llm")
    return subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        close_fds=True,
        start_new_session=True,
    )


def _status_response(source: Path, paper_dir: Path, state: dict) -> dict:
    key = str(paper_dir)
    return {
        **state,
        "status": state.get("status", "queued"),
        "source": str(source),
        "paper_dir": key,
        "estimated_completion": PROCESSING_ETA,
        "retry_after_seconds": PROCESSING_RETRY_AFTER_SECONDS,
        "next_step": (
            f"Call get_paper_info with paper_dir={key!r} in "
            f"{PROCESSING_RETRY_AFTER_SECONDS} seconds; when status is 'ready', "
            "retry search with the original source."
        ),
    }


def _schedule_processing(source: Path, paper_dir: Path, use_llm: bool) -> dict:
    """Start or resume one detached processing job and return immediately."""
    key = str(paper_dir)
    with _PROCESSING_LOCK:
        state = _read_processing_state(paper_dir)
        if _processing_is_active(key, state):
            return _status_response(source, paper_dir, state or {})

        queued_state = {
            "status": "queued",
            "source": str(source),
            "paper_dir": key,
            "queued_at": datetime.now(timezone.utc).isoformat(),
            "estimated_completion": PROCESSING_ETA,
            "retry_after_seconds": PROCESSING_RETRY_AFTER_SECONDS,
            "message": (
                "PDF processing has started in a detached background process and "
                "will continue after this call returns."
            ),
        }
        _write_processing_state(paper_dir, queued_state)
        try:
            process = _start_processing_process(source, paper_dir, use_llm)
        except OSError as exc:
            failed_state = {
                **queued_state,
                "status": "failed",
                "message": f"Could not start background processing: {exc}",
            }
            _write_processing_state(paper_dir, failed_state)
            return _status_response(source, paper_dir, failed_state)
        _PROCESSING_JOBS[key] = process
        queued_state["worker_pid"] = process.pid
        persisted_state = _read_processing_state(paper_dir)
        if not persisted_state or persisted_state.get("status") == "queued":
            _write_processing_state(paper_dir, queued_state)
        else:
            # The child may already have advanced the state to "processing".
            queued_state = persisted_state
        return _status_response(source, paper_dir, queued_state)


def _find_paper_dirs(
    search_paths: list[str],
    auto_process: bool = True,
    use_llm: bool = False,
) -> tuple[list[Path], list[dict]]:
    """Find all paper directories, optionally processing PDFs.

    Args:
        search_paths: List of PDF paths or paper directories
        auto_process: Whether to auto-process PDFs and incomplete directories
        use_llm: Use LLM for PDF conversion

    Returns:
        Tuple of (ready paper_dirs, background processing jobs)
    """
    paper_dirs = []
    processing_jobs = []

    for path_str in search_paths:
        path = Path(path_str).expanduser().resolve()

        # A direct PDF never discovers or reads sibling directories. First-use work
        # runs asynchronously because conversion and embedding normally exceed MCP
        # client deadlines.
        if path.is_file() and path.suffix.lower() == ".pdf":
            from .convert import get_output_dir

            paper_dir = get_output_dir(path)
            if _is_fully_processed(paper_dir):
                paper_dirs.append(paper_dir)
            elif auto_process:
                processing_jobs.append(_schedule_processing(path, paper_dir, use_llm))
            continue

        # Handle paper.md files directly.
        if path.is_file() and path.name == "paper.md":
            if _is_fully_processed(path.parent) or not auto_process:
                paper_dirs.append(path.parent)
            else:
                processing_jobs.append(_schedule_processing(path, path.parent, use_llm))
            continue

        # Directory discovery is limited to the directory explicitly supplied by the
        # caller and its immediate paper children.
        if path.is_dir():
            if (path / "paper.md").exists():
                candidates = [path]
            else:
                candidates = []
                try:
                    children = path.iterdir()
                    for subdir in children:
                        try:
                            if subdir.is_dir() and (subdir / "paper.md").exists():
                                candidates.append(subdir)
                        except OSError:
                            # An explicitly selected library may contain unreadable
                            # siblings; one sibling must not abort discovery of others.
                            continue
                except OSError:
                    pass

            for candidate in candidates:
                if _is_fully_processed(candidate) or not auto_process:
                    paper_dirs.append(candidate)
                else:
                    processing_jobs.append(
                        _schedule_processing(candidate / "paper.md", candidate, use_llm)
                    )

    return paper_dirs, processing_jobs


def grep_search(
    query: str,
    paper_dirs: list[Path],
    case_sensitive: bool = False,
    regex: bool = False,
    max_results: Optional[int] = None,
) -> list[dict]:
    """Perform grep-style text search across paper markdown files.

    Args:
        query: Search query (text or regex pattern)
        paper_dirs: List of paper directories to search
        case_sensitive: Whether search is case sensitive
        regex: Whether query is a regex pattern

    Returns:
        List of matches with file, line number, content, and context
    """
    results = []

    # Compile pattern
    flags = 0 if case_sensitive else re.IGNORECASE
    if regex:
        pattern = re.compile(query, flags)
    else:
        pattern = re.compile(re.escape(query), flags)

    for paper_dir in paper_dirs:
        md_file = paper_dir / "paper.md"
        index_file = paper_dir / "index.json"

        if not md_file.exists():
            continue

        try:
            content = md_file.read_text(encoding="utf-8")
            lines = content.split("\n")

            # Load pre-built index if available, otherwise parse
            header_lookup = {}
            if index_file.exists():
                try:
                    index_data = json.loads(index_file.read_text())
                    for h in index_data.get("flat_headers", []):
                        header_lookup[h["line"]] = h["path"]
                except Exception:
                    pass

            # Fall back to parser if no index
            parser = None
            if not header_lookup:
                parser = MarkdownParser(content, str(md_file))

            for line_num, line in enumerate(lines, start=1):
                if pattern.search(line):
                    # Get header context
                    if header_lookup:
                        # Find nearest header before this line
                        header_context = ""
                        for h_line in sorted(header_lookup.keys(), reverse=True):
                            if h_line <= line_num:
                                header_context = header_lookup[h_line]
                                break
                    else:
                        header = parser.get_header_at_line(line_num)
                        header_context = header.path if header else ""

                    # Get surrounding context
                    context_start = max(0, line_num - 3)
                    context_end = min(len(lines), line_num + 2)
                    context_lines = lines[context_start:context_end]

                    results.append({
                        "paper_dir": str(paper_dir),
                        "paper_name": paper_dir.name,
                        "source": str(md_file),
                        "line_number": line_num,
                        "content": line.strip(),
                        "context": "\n".join(context_lines),
                        "header_context": header_context,
                        "match_type": "grep",
                    })
                    if max_results is not None and len(results) >= max_results:
                        return results

        except Exception:
            continue

    return results


def _get_rag_client_class():
    from ..utils.chromadb_client import RAGClient

    return RAGClient


def rag_search(
    query: str,
    paper_dirs: list[Path],
    top_k: int = 5,
) -> list[dict]:
    """Perform semantic RAG search using embeddings.

    Args:
        query: Search query
        paper_dirs: List of paper directories to search
        top_k: Number of results per paper

    Returns:
        List of matches with content, score, and metadata
    """
    RAGClient = _get_rag_client_class()

    results = []
    embed_model = None

    for paper_dir in paper_dirs:
        chroma_dir = paper_dir / "chroma"

        if not chroma_dir.exists():
            continue

        try:
            rag_client = RAGClient(
                persist_directory=chroma_dir,
                embed_model=embed_model,
            )
            embed_model = rag_client.embed_model
            raw_results = rag_client.query("paper", query, top_k)

            for r in raw_results:
                metadata = r.get("metadata", {})
                result_entry = {
                    "paper_dir": str(paper_dir),
                    "paper_name": paper_dir.name,
                    "source": metadata.get("source", str(paper_dir / "paper.md")),
                    "content": r.get("text", ""),
                    "score": r.get("score", 0.0),
                    "header_context": metadata.get("header_path", ""),
                    "match_type": "rag",
                }
                # Include line numbers if available (from pre-chunked documents)
                if "start_line" in metadata:
                    result_entry["start_line"] = metadata["start_line"]
                if "end_line" in metadata:
                    result_entry["end_line"] = metadata["end_line"]
                results.append(result_entry)

        except Exception:
            continue

    # Sort by score
    results.sort(key=lambda x: x.get("score", 0), reverse=True)
    return results[:top_k]


def search(
    query: str,
    sources: list[str],
    mode: Literal["grep", "rag", "hybrid"] = "hybrid",
    top_k: int = 5,
    case_sensitive: bool = False,
    regex: bool = False,
    include_context: bool = True,
    use_llm: bool = False,
) -> dict:
    """Search PDF documents and paper directories.

    Queues first-use PDF processing in the background (normally 1-3 minutes)
    and returns actionable status immediately. Retry once get_paper_info is ready.

    Args:
        query: Search query (text, regex pattern, or semantic query)
        sources: List of PDF paths or paper directories to search
        mode: Search mode - "grep", "rag", or "hybrid" (default)
        top_k: Number of results to return (default 5)
        case_sensitive: Case sensitivity for grep (default False)
        regex: Treat query as regex pattern for grep (default False)
        include_context: Include surrounding context in results (default True)
        use_llm: Use LLM for enhanced PDF conversion accuracy (slower)

    Returns:
        Result with results list, num_results, and success status
    """
    # Discover only explicitly requested sources. Slow first-use processing is queued
    # so the MCP response stays within normal client deadlines.
    dirs, processing_jobs = _find_paper_dirs(sources, auto_process=True, use_llm=use_llm)

    if not dirs:
        if processing_jobs:
            return {
                "results": [],
                "num_results": 0,
                "query": query,
                "mode": mode,
                "success": True,
                "status": "processing",
                "message": (
                    "First-use processing takes about 1-3 minutes and is continuing "
                    "in the background. Check get_paper_info, then retry this search."
                ),
                "processing": processing_jobs,
            }
        return {
            "results": [],
            "num_results": 0,
            "query": query,
            "mode": mode,
            "success": True,
            "status": "no_sources",
            "message": "No paper directories found for the requested sources.",
        }

    results = []

    try:
        if mode in ("grep", "hybrid"):
            grep_results = grep_search(
                query=query,
                paper_dirs=dirs,
                case_sensitive=case_sensitive,
                regex=regex,
                max_results=top_k * 2 if mode == "hybrid" else top_k,
            )
            results.extend(grep_results)

        semantic_status = None
        if mode in ("rag", "hybrid"):
            if len(dirs) > MAX_SEMANTIC_PAPERS:
                semantic_status = {
                    "status": "skipped_source_limit",
                    "papers_requested": len(dirs),
                    "max_papers": MAX_SEMANTIC_PAPERS,
                    "message": (
                        "Semantic search was skipped to keep this MCP call bounded. "
                        f"Select at most {MAX_SEMANTIC_PAPERS} paper directories for "
                        "RAG, or use grep mode for the whole collection."
                    ),
                }
            else:
                rag_results = rag_search(
                    query=query,
                    paper_dirs=dirs,
                    top_k=top_k,
                )
                results.extend(rag_results)

        # Deduplicate by content similarity
        if mode == "hybrid":
            results = _deduplicate_results(results)

        # Sort results
        if mode == "rag":
            results.sort(key=lambda x: x.get("score", 0), reverse=True)
        elif mode == "hybrid":
            # Prioritize by score if available
            results.sort(
                key=lambda x: (
                    x.get("score", 0) if x.get("match_type") == "rag" else 0.5,
                ),
                reverse=True,
            )

        # Limit results
        results = results[:top_k * 2] if mode == "hybrid" else results[:top_k]

        # Remove context if not requested
        if not include_context:
            for r in results:
                r.pop("context", None)

        result = {
            "results": results,
            "num_results": len(results),
            "papers_searched": len(dirs),
            "query": query,
            "mode": mode,
            "success": True,
        }
        if semantic_status:
            result["semantic_search"] = semantic_status
        if processing_jobs or semantic_status:
            result["status"] = "partial"
            if processing_jobs:
                result["processing"] = processing_jobs
            messages = []
            if processing_jobs:
                messages.append(
                    "Searched ready sources; remaining first-use processing continues "
                    "in the background."
                )
            if semantic_status:
                messages.append(semantic_status["message"])
            result["message"] = " ".join(messages)
        else:
            result["status"] = "ready"
        return result

    except Exception as e:
        return {
            "results": [],
            "num_results": 0,
            "query": query,
            "mode": mode,
            "success": False,
            "message": f"Search failed: {str(e)}",
        }


def _deduplicate_results(results: list[dict]) -> list[dict]:
    """Deduplicate results based on content similarity."""
    seen_content = set()
    unique_results = []

    for r in results:
        content = r.get("content", "").strip().lower()[:200]
        if content not in seen_content:
            seen_content.add(content)
            unique_results.append(r)

    return unique_results


def _get_chroma_collection_count(chroma_dir: Path) -> Optional[int]:
    """Read Chroma's count without loading the embedding model.

    Status calls must remain cheap and must not download or initialize the semantic
    model merely to report an already-built collection's size.
    """
    database = chroma_dir / "chroma" / "chroma.sqlite3"
    if not database.exists():
        return None
    try:
        connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True, timeout=1)
        try:
            row = connection.execute("SELECT COUNT(*) FROM embeddings").fetchone()
            return int(row[0]) if row else 0
        finally:
            connection.close()
    except (OSError, sqlite3.Error, ValueError):
        return None


def get_paper_info(paper_dir: str) -> dict:
    """Get information about a paper directory.

    Args:
        paper_dir: Path to the paper directory

    Returns:
        Dictionary with paper information including:
        - Processing status (has_markdown, has_index, has_embeddings, has_images)
        - Version compatibility info
        - Image paths (if images exist)
        - Metadata from processing
    """
    from ..metadata import check_version_compatibility, read_metadata

    path = Path(paper_dir).expanduser().resolve()
    if path.is_file() and path.suffix.lower() == ".pdf":
        from .convert import get_output_dir
        path = get_output_dir(path)

    if not path.is_dir():
        return {
            "success": False,
            "message": f"Not a directory: {paper_dir}",
        }

    info = {
        "name": path.name,
        "path": str(path),
        "has_markdown": (path / "paper.md").exists(),
        "has_index": (path / "index.json").exists(),
        "has_embeddings": (path / "chroma").exists(),
        "has_images": (path / "images").exists(),
        "success": True,
    }

    version_info = check_version_compatibility(path)
    processing = _read_processing_state(path)
    if processing and processing.get("status") in {"queued", "processing", "failed"}:
        info["processing"] = processing
        info["status"] = processing["status"]
        if (
            info["status"] in {"queued", "processing"}
            and processing.get("worker_pid")
            and not _processing_is_active(str(path), processing)
        ):
            info["status"] = "stalled"
            info["message"] = (
                "The background worker stopped before processing completed. Retry "
                "search to start a new worker."
            )
    elif (
        info["has_markdown"]
        and info["has_index"]
        and info["has_embeddings"]
        and version_info["is_compatible"]
    ):
        info["status"] = "ready"
    else:
        info["status"] = "incomplete"

    # Add paths for easy access
    if info["has_markdown"]:
        info["markdown_path"] = str(path / "paper.md")
    if info["has_images"]:
        images_dir = path / "images"
        info["images_dir"] = str(images_dir)
        # List image files for convenience
        try:
            image_files = [f.name for f in images_dir.iterdir() if f.is_file()]
            info["image_files"] = image_files
            info["image_count"] = len(image_files)
        except Exception:
            info["image_files"] = []
            info["image_count"] = 0

    # Get header count from index
    if info["has_index"]:
        try:
            index_data = json.loads((path / "index.json").read_text())
            info["header_count"] = len(index_data.get("flat_headers", []))
        except Exception:
            info["header_count"] = 0

    # Count chunks directly from the local Chroma database. Constructing RAGClient
    # here initializes a HuggingFace model and made a status check take minutes.
    if info["has_embeddings"]:
        info["chunk_count"] = _get_chroma_collection_count(path / "chroma")

    # Report version compatibility.
    info["version"] = {
        "processed_version": version_info["processed_version"],
        "current_version": version_info["current_version"],
        "is_compatible": version_info["is_compatible"],
    }
    if version_info["message"]:
        info["version_warning"] = version_info["message"]

    # Include metadata if available
    metadata = read_metadata(path)
    if metadata:
        info["metadata"] = {
            "source_pdf": metadata.get("source_pdf"),
            "processed_at": metadata.get("processed_at"),
            "steps_completed": metadata.get("steps_completed", []),
        }

    return info
