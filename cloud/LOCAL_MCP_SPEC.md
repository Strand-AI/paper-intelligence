# Local MCP Server Spec

## Overview

A single local stdio MCP server (Node.js, published to npm) that wraps the Paper Intelligence Cloud REST API. Replaces both the current remote MCP server on the Worker and the local upload-only MCP server.

## Architecture

```
Claude Code  →  local MCP server (stdio)  →  Cloudflare Worker (REST API)
```

- The remote McpAgent on the Worker is removed. The Worker only exposes REST endpoints.
- The local MCP server reads files from disk (for upload) and proxies all other operations as HTTP requests to the Worker.
- One MCP server to install. One MCP server to configure.

## Distribution

- **npm package**: `paper-intelligence` (or `@strand-ai/paper-intelligence`)
- **Install**: `npx paper-intelligence` (stdio MCP server)
- **Registration**: `claude mcp add paper-intelligence -- npx paper-intelligence`

## Auth & Config

- **Worker URL**: hardcoded default (`https://paper-intelligence.daiyue0531.workers.dev`), overridable via `PAPER_INTELLIGENCE_URL` env var.
- **API token**: read from `PAPER_INTELLIGENCE_TOKEN` env var, with 1Password fallback (`op read 'op://CLI Secrets/Paper Intelligence Cloud API Token/credential'`).
- No CLI args needed.

## Tools

### `search`
Search across papers using grep, RAG, or hybrid mode.

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| query | string | required | Search query |
| mode | "grep" \| "rag" \| "hybrid" | "hybrid" | Search mode |
| top_k | number | 5 | Max results |
| sources | string[] | [] | Paper names/IDs to scope (empty = all) |
| case_sensitive | boolean | false | Case-sensitive grep |
| regex | boolean | false | Treat query as regex |

Calls: `POST /chat` — no, calls the search endpoint. Actually there's no dedicated search REST endpoint — search is currently only exposed via MCP. Need to add `POST /search` to the Worker REST API.

### `list_papers`
List all papers, optionally get details on a specific one.

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| source | string | optional | Paper name/ID to get details for |

- Without `source`: calls `GET /papers` → returns list with status, chunk counts.
- With `source`: calls `GET /papers/:id` + `GET /papers/:id/headers` → returns full details including section outline.

### `upload_paper`
Upload PDF files from local disk.

| Param | Type | Description |
|-------|------|-------------|
| paths | string[] | File paths to PDFs |

- Reads each file from disk.
- POSTs as multipart form to `POST /papers`.
- Returns immediately with status "converting" (fire and forget).
- Supports multiple files in one call.

### `delete_paper`
Delete a paper from the library.

| Param | Type | Description |
|-------|------|-------------|
| source | string | Paper name or ID |

Calls: `DELETE /papers/:id` (after resolving name → ID via list).

### `rename_paper`
Set a display name alias for a paper.

| Param | Type | Description |
|-------|------|-------------|
| source | string | Paper name or ID |
| alias | string | New display name (empty to reset) |

Calls: `PATCH /papers/:id` with `{ alias }`.

## Worker Changes

- Remove `McpAgent` / Durable Object for MCP. Remove `agents` and `@modelcontextprotocol/sdk` deps from the Worker.
- Add `POST /search` REST endpoint (currently search is only available via MCP).
- Keep all existing REST endpoints.
- The Worker becomes a pure REST API + static asset server (React app).

## Chat

- Chat (`POST /chat`) is a web UI feature only. No MCP tool for chat.
- Claude Code uses `search` and reasons over results directly.

## What Gets Removed

- `src/mcp.ts` — remote McpAgent
- `paper-intelligence-cloud` MCP server registration (HTTP)
- `paper-intelligence-upload` MCP server registration (stdio)
- `cloud/mcp-upload.mjs` — replaced by the npm package
- McpAgent Durable Object binding + migrations in wrangler.jsonc
- `agents` and `@modelcontextprotocol/sdk` Worker dependencies
