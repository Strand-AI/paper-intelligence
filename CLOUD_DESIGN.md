# Paper Intelligence Cloud

## Background

Paper Intelligence currently runs entirely locally: PDFs are stored on disk, Marker converts them to Markdown using local GPU/CPU, embeddings are generated with a local HuggingFace model (bge-small-en-v1.5), and vectors are stored in per-paper ChromaDB instances. The MCP server runs as a local process.

This works, but has limitations:
- Papers are only available on the machine they were processed on
- Every machine needs the full ML stack (torch, marker, embedding models)
- No shared access between team members (Oded + Yue)
- Processing ties up local compute

## The Idea

Move storage, indexing, and search to Cloudflare. The local MCP server becomes a thin API client. Papers are accessible from anywhere, by anyone on the team.

Strand has $250K in Cloudflare credits (YC perk) that are otherwise going mostly unused. This is a natural fit.

## Architecture

### Current (Local)

```
PDF on disk
  → Marker (local, torch, GPU/MPS/CPU) → Markdown + images
  → HuggingFace bge-small-en-v1.5 (local) → embeddings
  → ChromaDB (local, per-paper SQLite) → vector storage
  → Local grep + RAG → search results
  → MCP server (local, stdio) → Claude/IDE
```

### Proposed (Cloud)

```
┌─────────────────────────────────────────────────────┐
│ Cloudflare                                          │
│                                                     │
│  ┌──────────┐    ┌──────────────────┐               │
│  │    R2    │    │   Container      │               │
│  │ (storage)│◄───│ (Marker on CPU)  │               │
│  │          │    │  standard-2/4    │               │
│  │ - PDFs   │    │  scale to zero   │               │
│  │ - .md    │    └──────────────────┘               │
│  │ - images │                                       │
│  └────┬─────┘    ┌──────────────────┐               │
│       │          │   Workers AI     │               │
│       │          │ (embeddings)     │               │
│       │          └────────┬─────────┘               │
│       │                   │                         │
│       │          ┌────────▼─────────┐               │
│       │          │   Vectorize      │               │
│       │          │ (vector search)  │               │
│       │          └────────┬─────────┘               │
│       │                   │                         │
│       │          ┌────────▼─────────┐               │
│       │          │   D1 (SQLite)    │               │
│       │          │ - metadata       │               │
│       │          │ - header index   │               │
│       │          └────────┬─────────┘               │
│       │                   │                         │
│  ┌────▼───────────────────▼─────────┐               │
│  │          Worker (API)            │               │
│  │  POST /papers      - upload      │               │
│  │  GET  /papers/:id  - status      │               │
│  │  POST /search      - query       │               │
│  └──────────────┬───────────────────┘               │
│                 │                                    │
└─────────────────┼────────────────────────────────────┘
                  │ HTTPS
┌─────────────────▼────────────────────────────────────┐
│ Local MCP Server (thin client)                       │
│  - Proxies tool calls to Worker API                  │
│  - Same MCP interface: search(), get_paper_info()    │
│  - No torch, no marker, no embeddings locally        │
└──────────────────────────────────────────────────────┘
```

## Cloudflare Services Used

| Service | Role | Why |
|---------|------|-----|
| **R2** | Object storage for PDFs, Markdown, images | S3-compatible, no egress fees, cheap |
| **Containers** | Runs Marker for PDF→Markdown conversion | Needs Python + torch (CPU). Scale to zero when idle |
| **Workers AI** | Generates embeddings | Replaces local HuggingFace model. No infra to manage |
| **Vectorize** | Vector database for semantic search | Replaces per-paper ChromaDB. Managed, scalable |
| **D1** | SQLite database for metadata + header index | Replaces per-paper JSON files. Queryable |
| **Workers** | API layer (HTTP endpoints) | Orchestrates everything. Request/response, no agent loop |

## Processing Pipeline

### Upload & Process (async, background)

1. Client uploads PDF to Worker API
2. Worker stores PDF in R2
3. Worker triggers Container (Marker on CPU)
   - Container pulls PDF from R2
   - Marker converts to Markdown + extracts images
   - Container writes .md + images back to R2
   - Container signals completion to Worker
4. Worker reads Markdown from R2
5. Worker extracts header hierarchy → stores in D1
6. Worker chunks Markdown (header-aware, 512 tokens, 50 overlap)
7. Worker calls Workers AI for embeddings
8. Worker stores vectors in Vectorize
9. Worker updates paper status in D1

### Search (sync, fast)

1. Client sends query to Worker API
2. Worker runs in parallel:
   - **Grep**: fetch Markdown from R2, regex/text search with header context
   - **RAG**: embed query via Workers AI, search Vectorize for top-k
3. Worker deduplicates + merges results (same logic as current hybrid search)
4. Worker returns results with line numbers, header context, scores

## Container Details (Marker)

- **Instance type**: `standard-2` (1 vCPU, 6 GiB RAM, 12 GB disk) or `standard-4` (4 vCPU, 12 GiB RAM, 20 GB disk)
- **Runtime**: Python + torch (CPU) + marker-pdf
- **Scale to zero**: sleeps when idle, wakes on demand. Only pay for active processing
- **Image size concern**: torch CPU (~500MB) + Marker models + Python. Needs to fit in 12-20 GB disk. Should be fine but verify

### CPU Performance Benchmarks (from local M-series Mac)

| Paper Size | Pages | Time (CPU) |
|-----------|-------|------------|
| Large (Nature Medicine + supplements) | 85 | ~8 min |
| Medium (typical paper) | 15-20 | ~2-3 min (estimated) |
| Small (short paper) | 5-10 | ~1 min (estimated) |

On Cloudflare `standard-4` (4 vCPU), expect similar or slightly slower times. PyTorch does use multiple cores for inference but it's not a linear speedup.

This is acceptable — papers are processed once, then search is instant forever.

## Data Model (D1)

```sql
CREATE TABLE papers (
  id TEXT PRIMARY KEY,          -- uuid
  name TEXT NOT NULL,           -- human-readable name
  pdf_key TEXT NOT NULL,        -- R2 key for PDF
  markdown_key TEXT,            -- R2 key for Markdown
  status TEXT NOT NULL,         -- 'uploading' | 'converting' | 'indexing' | 'embedding' | 'ready' | 'error'
  error TEXT,                   -- error message if failed
  page_count INTEGER,
  markdown_length INTEGER,
  chunk_count INTEGER,
  created_at TEXT NOT NULL,
  processed_at TEXT,
  version TEXT NOT NULL         -- paper-intelligence version for compat
);

CREATE TABLE headers (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  paper_id TEXT NOT NULL REFERENCES papers(id),
  level INTEGER NOT NULL,       -- 1-6
  text TEXT NOT NULL,
  line_number INTEGER NOT NULL,
  path TEXT NOT NULL,            -- "Section > Subsection > ..."
  FOREIGN KEY (paper_id) REFERENCES papers(id)
);

CREATE TABLE images (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  paper_id TEXT NOT NULL REFERENCES papers(id),
  r2_key TEXT NOT NULL,
  filename TEXT NOT NULL,
  FOREIGN KEY (paper_id) REFERENCES papers(id)
);
```

## API Design (Worker)

### `POST /papers`

Upload a new paper for processing.

```
Content-Type: multipart/form-data
Body: { file: <pdf>, name?: <string> }

Response 202:
{ id: "uuid", name: "paper-name", status: "converting" }
```

### `GET /papers/:id`

Check paper status.

```
Response 200:
{
  id: "uuid",
  name: "paper-name",
  status: "ready",
  page_count: 24,
  chunk_count: 47,
  created_at: "2026-04-09T...",
  processed_at: "2026-04-09T..."
}
```

### `GET /papers`

List all papers.

```
Response 200:
{ papers: [{ id, name, status, created_at }] }
```

### `POST /search`

Search across papers.

```json
{
  "query": "attention mechanism",
  "mode": "hybrid",         // "grep" | "rag" | "hybrid"
  "top_k": 5,
  "paper_ids": ["uuid1"],   // optional, search subset
  "case_sensitive": false,
  "regex": false
}

Response 200:
{
  "results": [
    {
      "paper_id": "uuid",
      "paper_name": "...",
      "content": "matched text",
      "line_number": 142,
      "header_context": "Methods > Data",
      "score": 0.89,
      "match_type": "rag"
    }
  ],
  "query": "attention mechanism",
  "mode": "hybrid",
  "papers_searched": 3
}
```

### `DELETE /papers/:id`

Delete a paper and all associated data (R2 objects, D1 rows, Vectorize vectors).

## MCP Server Changes

The local MCP server keeps the same tool interface but swaps the implementation:

```python
# Before (local)
from paper_intelligence.tools.search import search
result = search(query="...", sources=["~/papers/foo.pdf"])

# After (cloud client)
import httpx
result = httpx.post("https://paper-intelligence.<account>.workers.dev/search", json={...})
```

**Tools stay the same:**
- `search(query, sources, mode, top_k, ...)` → calls `POST /search`
- `get_paper_info(paper_dir)` → calls `GET /papers/:id`

The `sources` parameter changes semantics — instead of local file paths, it accepts paper names or IDs. For backwards compatibility, if a local PDF path is passed, the MCP server uploads it first, waits for processing, then searches.

## Auth

Simple API key auth for now. Key stored in 1Password CLI Secrets vault, passed as `Authorization: Bearer <key>` header. The Worker validates it.

No need for anything fancier with 2 users.

## Migration Path

1. **Phase 1**: Build the cloud backend (Worker + R2 + Container + Vectorize + D1)
2. **Phase 2**: Add cloud client mode to MCP server (flag to switch local vs cloud)
3. **Phase 3**: Re-process existing papers from ~/Documents/papers/ into cloud
4. **Phase 4**: Default to cloud, keep local as fallback

Local mode stays intact — the cloud backend is additive, not a replacement. Useful for offline work or when you don't want to wait for network round trips.

## What This Enables

- **Shared paper library**: Oded and Yue search the same papers from any machine
- **No local ML stack**: New machines just need the thin MCP client, no torch/marker/embeddings
- **Process once, search everywhere**: Paper processed on upload, instantly searchable from any device
- **Burns Cloudflare credits on something useful**: R2 storage, Container compute, Workers AI embeddings, Vectorize queries

## Open Questions

- **Workers AI embedding model**: Which model to use? Need to check what's available and how it compares to bge-small-en-v1.5 quality-wise. Ideally something with similar or better retrieval quality.
- **Vectorize limits**: Check max vectors, dimensions, metadata size. Make sure it handles our scale (hundreds of papers, thousands of chunks).
- **Container cold start**: How long does it take a Marker container to wake from sleep? If it's 30+ seconds, might want to keep it warm during active use.
- **Markdown storage for grep**: Grep search needs the full Markdown. Fetching from R2 on every search adds latency. Could cache in D1 or use Workers KV, but Markdown can be large. Need to benchmark.
- **Image serving**: Do we need to serve extracted images? Current paper-intelligence references them in search results. If so, R2 presigned URLs work.
