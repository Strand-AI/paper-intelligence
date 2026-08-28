"""Regression tests for bounded, source-scoped MCP behavior."""

import importlib
import json
import sqlite3
import subprocess
import sys
from pathlib import Path


def _ready_paper(path: Path) -> Path:
    path.mkdir()
    (path / "paper.md").write_text("# Manual\n\nA bounded local search.\n", encoding="utf-8")
    (path / "index.json").write_text('{"flat_headers": []}', encoding="utf-8")
    (path / "chroma").mkdir()
    return path


def test_direct_pdf_does_not_discover_sibling_temp_content(tmp_path, monkeypatch):
    """A /tmp PDF must not inspect unrelated sibling metadata or directories."""
    search_module = importlib.import_module("paper_intelligence.tools.search")
    source = tmp_path / "requested.pdf"
    source.write_bytes(b"%PDF-1.4\n")
    unrelated = tmp_path / ".vnc-0"
    unrelated.mkdir()
    (unrelated / "metadata.json").write_text("unrelated", encoding="utf-8")

    scheduled = []

    def fake_schedule(job_source, paper_dir, use_llm):
        scheduled.append((job_source, paper_dir))
        return {"status": "queued", "paper_dir": str(paper_dir)}

    monkeypatch.setattr(search_module, "_schedule_processing", fake_schedule)

    ready, jobs = search_module._find_paper_dirs([str(source)])

    assert ready == []
    assert scheduled == [(source.resolve(), tmp_path / "requested")]
    assert jobs == [{"status": "queued", "paper_dir": str(tmp_path / "requested")}]


def test_first_use_search_returns_actionable_background_status(tmp_path, monkeypatch):
    search_module = importlib.import_module("paper_intelligence.tools.search")

    source = tmp_path / "slow.pdf"
    source.write_bytes(b"%PDF-1.4\n")
    paper_dir = tmp_path / "slow"

    monkeypatch.setattr(
        search_module,
        "_schedule_processing",
        lambda *_: {
            "status": "processing",
            "paper_dir": str(paper_dir),
            "retry_after_seconds": 30,
            "next_step": "Call get_paper_info, then retry search.",
        },
    )

    result = search_module.search("query", [str(source)], mode="grep")

    assert result["success"] is True
    assert result["status"] == "processing"
    assert result["results"] == []
    assert "continuing in the background" in result["message"]
    assert result["processing"][0]["paper_dir"] == str(paper_dir)
    assert "get_paper_info" in result["processing"][0]["next_step"]


def test_processing_job_persists_completion_status(tmp_path, monkeypatch):
    search_module = importlib.import_module("paper_intelligence.tools.search")

    source = tmp_path / "paper.pdf"
    source.write_bytes(b"%PDF-1.4\n")
    paper_dir = tmp_path / "paper"

    monkeypatch.setattr(
        search_module,
        "_process_paper_if_needed",
        lambda *_: (paper_dir, None),
    )

    search_module._run_processing_job(source, paper_dir, use_llm=False)
    state = json.loads(
        (paper_dir / search_module.PROCESSING_STATE_FILENAME).read_text(encoding="utf-8")
    )

    assert state["status"] == "completed"
    assert state["paper_dir"] == str(paper_dir)
    assert "Retry search" in state["message"]


def test_status_module_import_does_not_load_embedding_stack():
    code = (
        "import sys; from paper_intelligence.tools import get_paper_info, search; "
        "assert callable(get_paper_info) and callable(search); "
        "assert 'paper_intelligence.utils.chromadb_client' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True, timeout=10)


def test_get_paper_info_counts_chunks_without_embedding_model(tmp_path):
    from paper_intelligence.tools.search import get_paper_info

    source = tmp_path / "manual.pdf"
    source.write_bytes(b"%PDF-1.4\n")
    paper_dir = _ready_paper(tmp_path / "manual")
    database_dir = paper_dir / "chroma" / "chroma"
    database_dir.mkdir()
    connection = sqlite3.connect(database_dir / "chroma.sqlite3")
    connection.execute("CREATE TABLE embeddings (id TEXT)")
    connection.executemany("INSERT INTO embeddings VALUES (?)", [("a",), ("b",)])
    connection.commit()
    connection.close()

    result = get_paper_info(str(source))

    assert result["status"] == "ready"
    assert result["chunk_count"] == 2


def test_mcp_search_does_not_sync_whole_library(tmp_path, monkeypatch):
    """Already-indexed local searches must not enter the five-minute rclone path."""
    import paper_intelligence.server as server
    import paper_intelligence.sync as sync

    paper_dir = _ready_paper(tmp_path / "manual")

    def sync_must_not_run():
        raise AssertionError("search must not run an unconditional library sync")

    monkeypatch.setattr(sync, "pull", sync_must_not_run)
    result = server.search("bounded", [str(paper_dir)], mode="grep")

    assert result["success"] is True
    assert result["status"] == "ready"
    assert result["num_results"] == 1
