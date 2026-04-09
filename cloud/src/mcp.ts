import { McpAgent } from "agents/mcp";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { z } from "zod";
import type { Env, Paper } from "./types";
import { search } from "./search";

export class PaperIntelligenceMCP extends McpAgent<Env> {
  server = new McpServer({
    name: "paper-intelligence",
    version: "0.2.0",
  });

  async init() {
    this.server.tool(
      "search",
      "Search across papers using text search (grep), semantic search (RAG), or both (hybrid). " +
        "Returns matching passages with line numbers, section context, and relevance scores.",
      {
        query: z.string().describe("Search query text"),
        mode: z
          .enum(["grep", "rag", "hybrid"])
          .default("hybrid")
          .describe("grep = exact text match, rag = semantic similarity, hybrid = both merged"),
        top_k: z
          .number()
          .int()
          .min(1)
          .max(50)
          .default(5)
          .describe("Max results to return"),
        sources: z
          .array(z.string())
          .default([])
          .describe("Paper names or IDs to search (empty = search all)"),
        case_sensitive: z
          .boolean()
          .default(false)
          .describe("Case-sensitive grep matching"),
        regex: z
          .boolean()
          .default(false)
          .describe("Treat query as regular expression for grep"),
      },
      async ({ query, mode, top_k, sources, case_sensitive, regex }) => {
        const response = await search(
          this.env,
          query,
          mode,
          top_k,
          sources,
          case_sensitive,
          regex,
        );

        return {
          content: [
            {
              type: "text" as const,
              text: JSON.stringify(response, null, 2),
            },
          ],
        };
      },
    );

    this.server.tool(
      "get_paper_info",
      "Get detailed information about a specific paper including processing status, " +
        "section headers, chunk count, and metadata.",
      {
        source: z
          .string()
          .describe("Paper name or ID"),
      },
      async ({ source }) => {
        const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

        let paper: Paper | null;
        if (UUID_RE.test(source)) {
          paper = await this.env.DB.prepare(
            "SELECT * FROM papers WHERE id = ?",
          )
            .bind(source)
            .first<Paper>();
        } else {
          paper = await this.env.DB.prepare(
            "SELECT * FROM papers WHERE LOWER(name) LIKE LOWER(?) LIMIT 1",
          )
            .bind(`%${source}%`)
            .first<Paper>();
        }

        if (!paper) {
          return {
            content: [
              {
                type: "text" as const,
                text: JSON.stringify({
                  success: false,
                  message: `Paper not found: ${source}`,
                }),
              },
            ],
          };
        }

        // Get headers for this paper
        const { results: headers } = await this.env.DB.prepare(
          "SELECT level, text, line_number, path FROM headers WHERE paper_id = ? ORDER BY line_number",
        )
          .bind(paper.id)
          .all();

        // Get image count
        const imageRow = await this.env.DB.prepare(
          "SELECT COUNT(*) as count FROM images WHERE paper_id = ?",
        )
          .bind(paper.id)
          .first<{ count: number }>();

        return {
          content: [
            {
              type: "text" as const,
              text: JSON.stringify(
                {
                  success: true,
                  id: paper.id,
                  name: paper.name,
                  status: paper.status,
                  error: paper.error,
                  page_count: paper.page_count,
                  markdown_length: paper.markdown_length,
                  chunk_count: paper.chunk_count,
                  header_count: headers.length,
                  image_count: imageRow?.count ?? 0,
                  headers: headers.map((h) => ({
                    level: h.level,
                    text: h.text,
                    line: h.line_number,
                    path: h.path,
                  })),
                  created_at: paper.created_at,
                  processed_at: paper.processed_at,
                  version: paper.version,
                },
                null,
                2,
              ),
            },
          ],
        };
      },
    );

    this.server.tool(
      "list_papers",
      "List all papers in the library with their processing status.",
      {},
      async () => {
        const { results } = await this.env.DB.prepare(
          "SELECT id, name, status, page_count, chunk_count, created_at, processed_at FROM papers ORDER BY created_at DESC",
        ).all();

        return {
          content: [
            {
              type: "text" as const,
              text: JSON.stringify(
                {
                  success: true,
                  papers: results,
                  total: results.length,
                },
                null,
                2,
              ),
            },
          ],
        };
      },
    );
  }
}
