#!/usr/bin/env node

/**
 * Local MCP server for uploading papers to Paper Intelligence Cloud.
 * Reads PDF files from disk and POSTs them to the cloud Worker REST endpoint.
 *
 * Environment variables:
 *   PAPER_INTELLIGENCE_URL   — Worker URL (required)
 *   PAPER_INTELLIGENCE_TOKEN — Bearer token (required, or reads from 1Password)
 *
 * Usage:
 *   claude mcp add paper-intelligence-upload -- node /path/to/mcp-upload.mjs
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

const API_URL =
  process.env.PAPER_INTELLIGENCE_URL ||
  "https://paper-intelligence.daiyue0531.workers.dev";

function getToken() {
  if (process.env.PAPER_INTELLIGENCE_TOKEN) {
    return process.env.PAPER_INTELLIGENCE_TOKEN;
  }
  try {
    return execSync(
      "op read 'op://CLI Secrets/Paper Intelligence Cloud API Token/credential'",
      { encoding: "utf-8" },
    ).trim();
  } catch {
    throw new Error(
      "Set PAPER_INTELLIGENCE_TOKEN or configure 1Password CLI",
    );
  }
}

const server = new Server(
  { name: "paper-intelligence-upload", version: "0.1.0" },
  { capabilities: { tools: {} } },
);

server.setRequestHandler(ListToolsRequestSchema, async () => ({
  tools: [
    {
      name: "upload_paper",
      description:
        "Upload a local PDF file to Paper Intelligence Cloud. " +
        "The PDF is sent to the cloud Worker, stored in R2, and converted " +
        "to markdown via Marker. Conversion runs in the background (2-5 min). " +
        "Supports uploading multiple files at once.",
      inputSchema: {
        type: "object",
        properties: {
          paths: {
            type: "array",
            items: { type: "string" },
            description:
              "File paths to PDF files to upload (e.g. ['~/Documents/papers/my-paper.pdf'])",
          },
        },
        required: ["paths"],
      },
    },
  ],
}));

server.setRequestHandler(CallToolRequestSchema, async (request) => {
  if (request.params.name !== "upload_paper") {
    throw new Error(`Unknown tool: ${request.params.name}`);
  }

  const { paths } = request.params.arguments;
  const token = getToken();
  const results = [];

  for (const filePath of paths) {
    const resolvedPath = filePath.replace(/^~/, process.env.HOME || "~");
    const name = basename(resolvedPath).replace(/\.pdf$/i, "");

    try {
      const fileData = await readFile(resolvedPath);
      const formData = new FormData();
      formData.append("name", name);
      formData.append(
        "file",
        new Blob([fileData], { type: "application/pdf" }),
        basename(resolvedPath),
      );

      const res = await fetch(`${API_URL}/papers`, {
        method: "POST",
        headers: { Authorization: `Bearer ${token}` },
        body: formData,
      });

      const body = await res.json();
      results.push({
        name,
        success: res.ok,
        ...body,
      });
    } catch (err) {
      results.push({
        name,
        success: false,
        error: err.message,
      });
    }
  }

  return {
    content: [
      {
        type: "text",
        text: JSON.stringify(results, null, 2),
      },
    ],
  };
});

const transport = new StdioServerTransport();
await server.connect(transport);
