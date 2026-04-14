#!/usr/bin/env node

/**
 * Paper Intelligence MCP Server
 *
 * Local stdio MCP server that wraps the Paper Intelligence Cloud REST API.
 * Provides tools to search, list, upload, delete, and rename papers.
 *
 * Config (env vars):
 *   PAPER_INTELLIGENCE_URL   — Worker URL (default: https://paper-intelligence.daiyue0531.workers.dev)
 *   PAPER_INTELLIGENCE_TOKEN — Bearer token (falls back to 1Password CLI)
 */

import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";
import { readFile } from "fs/promises";
import { basename } from "path";
import { execSync } from "child_process";

// --- Config ---

const API_URL =
  process.env.PAPER_INTELLIGENCE_URL ||
  "https://paper-intelligence.daiyue0531.workers.dev";

let _token = process.env.PAPER_INTELLIGENCE_TOKEN || "";

function getToken() {
  if (_token) return _token;
  try {
    _token = execSync(
      "op read 'op://CLI Secrets/Paper Intelligence Cloud API Token/credential'",
      { encoding: "utf-8", stdio: ["pipe", "pipe", "pipe"] },
    ).trim();
    return _token;
  } catch {
    throw new Error(
      "Set PAPER_INTELLIGENCE_TOKEN or configure 1Password CLI (op)",
    );
  }
}

function headers() {
  return {
    Authorization: `Bearer ${getToken()}`,
    "Content-Type": "application/json",
  };
}

// --- API helpers ---

async function api(method, path, body) {
  const opts = { method, headers: headers() };
  if (body !== undefined) opts.body = JSON.stringify(body);
  const res = await fetch(`${API_URL}${path}`, opts);
  return res.json();
}

async function apiFormData(path, formData) {
  const res = await fetch(`${API_URL}${path}`, {
    method: "POST",
    headers: { Authorization: `Bearer ${getToken()}` },
    body: formData,
  });
  return res.json();
}

async function findPaper(source) {
  const { papers } = await api("GET", "/papers");
  const sl = source.toLowerCase();
  return papers.find(
    (p) =>
      p.id === source ||
      (p.name || "").toLowerCase().includes(sl) ||
      (p.title || "").toLowerCase().includes(sl) ||
      (p.alias || "").toLowerCase().includes(sl),
  );
}

// --- Tool definitions ---

const TOOLS = [
  {
    name: "search",
    description:
      "Search across papers using text search (grep), semantic search (RAG), or both (hybrid). " +
      "Returns matching passages with line numbers, section context, and relevance scores.",
    inputSchema: {
      type: "object",
      properties: {
        query: { type: "string", description: "Search query text" },
        mode: {
          type: "string",
          enum: ["grep", "rag", "hybrid"],
          default: "hybrid",
          description: "grep = exact text, rag = semantic, hybrid = both",
        },
        top_k: {
          type: "number",
          default: 5,
          description: "Max results to return",
        },
        sources: {
          type: "array",
          items: { type: "string" },
          default: [],
          description: "Paper names/IDs to search (empty = all)",
        },
        case_sensitive: {
          type: "boolean",
          default: false,
          description: "Case-sensitive grep",
        },
        regex: {
          type: "boolean",
          default: false,
          description: "Treat query as regex for grep",
        },
      },
      required: ["query"],
    },
  },
  {
    name: "list_papers",
    description:
      "List all papers in the library. Optionally pass a source name/ID to get " +
      "detailed info about a specific paper including section headers and metadata.",
    inputSchema: {
      type: "object",
      properties: {
        source: {
          type: "string",
          description:
            "Optional paper name or ID to get details for. Omit to list all.",
        },
      },
    },
  },
  {
    name: "upload_paper",
    description:
      "Upload local PDF files to the paper library. Files are sent to the cloud, " +
      "stored, and converted to searchable markdown via Marker (takes 2-5 min). " +
      "Returns immediately — use list_papers to check processing status.",
    inputSchema: {
      type: "object",
      properties: {
        paths: {
          type: "array",
          items: { type: "string" },
          description: "File paths to PDF files (e.g. ~/Documents/papers/paper.pdf)",
        },
      },
      required: ["paths"],
    },
  },
  {
    name: "delete_paper",
    description: "Delete a paper from the library (removes all data, vectors, and files).",
    inputSchema: {
      type: "object",
      properties: {
        source: { type: "string", description: "Paper name or ID to delete" },
      },
      required: ["source"],
    },
  },
  {
    name: "rename_paper",
    description: "Set a display name for a paper. Pass empty alias to reset to auto-detected title.",
    inputSchema: {
      type: "object",
      properties: {
        source: { type: "string", description: "Paper name or ID" },
        alias: { type: "string", description: "New display name (empty to reset)" },
      },
      required: ["source", "alias"],
    },
  },
];

// --- Tool handlers ---

async function handleSearch(args) {
  return api("POST", "/search", {
    query: args.query,
    mode: args.mode || "hybrid",
    top_k: args.top_k || 5,
    sources: args.sources || [],
    case_sensitive: args.case_sensitive || false,
    regex: args.regex || false,
  });
}

async function handleListPapers(args) {
  if (args.source) {
    const paper = await findPaper(args.source);
    if (!paper) return { error: `Paper not found: ${args.source}` };
    // Get full details including headers
    const [info, { headers: hdrs }] = await Promise.all([
      api("GET", `/papers/${paper.id}`),
      api("GET", `/papers/${paper.id}/headers`),
    ]);
    return { ...info, headers: hdrs };
  }
  return api("GET", "/papers");
}

async function handleUploadPaper(args) {
  const results = [];
  for (const filePath of args.paths) {
    const resolved = filePath.replace(/^~/, process.env.HOME || "~");
    const name = basename(resolved).replace(/\.pdf$/i, "");
    try {
      const fileData = await readFile(resolved);
      const formData = new FormData();
      formData.append("name", name);
      formData.append(
        "file",
        new Blob([fileData], { type: "application/pdf" }),
        basename(resolved),
      );
      const result = await apiFormData("/papers", formData);
      results.push({ name, ...result });
    } catch (err) {
      results.push({ name, success: false, error: err.message });
    }
  }
  return results;
}

async function handleDeletePaper(args) {
  const paper = await findPaper(args.source);
  if (!paper) return { error: `Paper not found: ${args.source}` };
  return api("DELETE", `/papers/${paper.id}`);
}

async function handleRenamePaper(args) {
  const paper = await findPaper(args.source);
  if (!paper) return { error: `Paper not found: ${args.source}` };
  return api("PATCH", `/papers/${paper.id}`, { alias: args.alias });
}

// --- MCP Server ---

const server = new Server(
  { name: "paper-intelligence", version: "1.0.0" },
  { capabilities: { tools: {} } },
);

server.setRequestHandler(ListToolsRequestSchema, async () => ({
  tools: TOOLS,
}));

server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;

  const handlers = {
    search: handleSearch,
    list_papers: handleListPapers,
    upload_paper: handleUploadPaper,
    delete_paper: handleDeletePaper,
    rename_paper: handleRenamePaper,
  };

  const handler = handlers[name];
  if (!handler) throw new Error(`Unknown tool: ${name}`);

  try {
    const result = await handler(args || {});
    return {
      content: [{ type: "text", text: JSON.stringify(result, null, 2) }],
    };
  } catch (err) {
    return {
      content: [{ type: "text", text: JSON.stringify({ error: err.message }) }],
      isError: true,
    };
  }
});

const transport = new StdioServerTransport();
await server.connect(transport);
