import type { Env, FlatHeader, SearchResult, SearchResponse } from "./types";
import { extractHeaders, getHeaderAtLine } from "./pipeline";

function escapeRegex(str: string): string {
  return str.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/**
 * Grep search: line-by-line regex matching on markdown text.
 * Returns matches with surrounding context (±3 lines).
 */
function grepSearch(
  paperId: string,
  paperName: string,
  markdown: string,
  headers: FlatHeader[],
  query: string,
  options: { caseSensitive: boolean; regex: boolean },
): SearchResult[] {
  const flags = options.caseSensitive ? "g" : "gi";
  let pattern: RegExp;
  try {
    pattern = new RegExp(
      options.regex ? query : escapeRegex(query),
      flags,
    );
  } catch {
    return []; // invalid regex
  }

  const lines = markdown.split("\n");
  const results: SearchResult[] = [];

  for (let i = 0; i < lines.length; i++) {
    pattern.lastIndex = 0;
    if (pattern.test(lines[i])) {
      const lineNum = i + 1;
      const contextStart = Math.max(0, i - 3);
      const contextEnd = Math.min(lines.length - 1, i + 3);
      const header = getHeaderAtLine(headers, lineNum);

      results.push({
        paper_id: paperId,
        paper_name: paperName,
        content: lines[i],
        line_number: lineNum,
        header_context: header?.path ?? "",
        score: 0.5, // default score for grep in hybrid mode
        match_type: "grep",
        context: lines.slice(contextStart, contextEnd + 1).join("\n"),
      });
    }
  }

  return results;
}

/**
 * RAG search: embed query, search Vectorize for semantically similar chunks.
 */
async function ragSearch(
  env: Env,
  query: string,
  paperIds: string[],
  topK: number,
): Promise<SearchResult[]> {
  // Embed the query
  const response = await env.AI.run("@cf/baai/bge-small-en-v1.5", {
    text: [query],
  });
  const result = response as { data: number[][] };
  const queryVector = result.data[0];

  // Query Vectorize — handle multi-paper filtering
  let allMatches: { id: string; score: number; metadata?: Record<string, unknown> }[] = [];

  if (paperIds.length === 0) {
    // Search all papers
    const result = await env.VECTORIZE.query(queryVector, {
      topK,
      returnMetadata: "all",
    });
    allMatches = result.matches;
  } else if (paperIds.length === 1) {
    // Single paper filter
    const result = await env.VECTORIZE.query(queryVector, {
      topK,
      filter: { paper_id: paperIds[0] },
      returnMetadata: "all",
    });
    allMatches = result.matches;
  } else {
    // Multiple papers: parallel queries, merge and re-sort
    const queries = paperIds.map((pid) =>
      env.VECTORIZE.query(queryVector, {
        topK,
        filter: { paper_id: pid },
        returnMetadata: "all",
      }),
    );
    const results = await Promise.all(queries);
    allMatches = results
      .flatMap((r) => r.matches)
      .sort((a, b) => b.score - a.score)
      .slice(0, topK);
  }

  return allMatches.map((m) => ({
    paper_id: (m.metadata?.paper_id as string) ?? "",
    paper_name: (m.metadata?.paper_name as string) ?? "",
    content: (m.metadata?.text as string) ?? "",
    line_number: (m.metadata?.start_line as number) ?? 0,
    end_line: (m.metadata?.end_line as number) ?? undefined,
    header_context: (m.metadata?.header_path as string) ?? "",
    score: m.score,
    match_type: "rag" as const,
  }));
}

/**
 * Resolve sources (names or IDs) to paper IDs.
 */
async function resolveSources(
  db: D1Database,
  sources: string[],
): Promise<string[]> {
  const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
  const ids: Set<string> = new Set();

  for (const source of sources) {
    if (UUID_RE.test(source)) {
      ids.add(source);
    } else {
      // Name-based lookup (case-insensitive substring)
      const { results } = await db
        .prepare(
          "SELECT id FROM papers WHERE LOWER(name) LIKE LOWER(?) AND status = 'ready'",
        )
        .bind(`%${source}%`)
        .all();
      for (const row of results) {
        ids.add(row.id as string);
      }
    }
  }

  return [...ids];
}

/**
 * Deduplicate results by content prefix (first 200 chars, lowercased).
 * Matches the Python hybrid dedup logic.
 */
function deduplicateResults(results: SearchResult[]): SearchResult[] {
  const seen = new Set<string>();
  const unique: SearchResult[] = [];

  for (const r of results) {
    const key = r.content.slice(0, 200).toLowerCase().trim();
    if (!seen.has(key)) {
      seen.add(key);
      unique.push(r);
    }
  }

  return unique;
}

/**
 * Main search function — supports grep, rag, and hybrid modes.
 */
export async function search(
  env: Env,
  query: string,
  mode: "grep" | "rag" | "hybrid" = "hybrid",
  topK: number = 5,
  sources: string[] = [],
  caseSensitive: boolean = false,
  regex: boolean = false,
): Promise<SearchResponse> {
  // Resolve sources to paper IDs
  const paperIds =
    sources.length > 0 ? await resolveSources(env.DB, sources) : [];

  // Get papers to search
  let papers: { id: string; name: string; markdown_text: string }[];
  if (paperIds.length > 0) {
    const placeholders = paperIds.map(() => "?").join(",");
    const { results } = await env.DB.prepare(
      `SELECT id, name, markdown_text FROM papers WHERE id IN (${placeholders}) AND status = 'ready'`,
    )
      .bind(...paperIds)
      .all();
    papers = results as typeof papers;
  } else {
    const { results } = await env.DB.prepare(
      "SELECT id, name, markdown_text FROM papers WHERE status = 'ready'",
    ).all();
    papers = results as typeof papers;
  }

  if (papers.length === 0) {
    return {
      results: [],
      num_results: 0,
      papers_searched: 0,
      query,
      mode,
      success: true,
    };
  }

  const searchPaperIds = papers.map((p) => p.id);
  let grepResults: SearchResult[] = [];
  let ragResults: SearchResult[] = [];

  if (mode === "grep" || mode === "hybrid") {
    // Run grep across all papers
    for (const paper of papers) {
      if (!paper.markdown_text) continue;
      const headers = extractHeaders(paper.markdown_text);
      const matches = grepSearch(
        paper.id,
        paper.name,
        paper.markdown_text,
        headers,
        query,
        { caseSensitive, regex },
      );
      grepResults.push(...matches);
    }
  }

  if (mode === "rag" || mode === "hybrid") {
    ragResults = await ragSearch(env, query, searchPaperIds, topK);
  }

  // Merge results based on mode
  let results: SearchResult[];
  if (mode === "grep") {
    results = grepResults.slice(0, topK);
  } else if (mode === "rag") {
    results = ragResults;
  } else {
    // Hybrid: RAG results first (by score), then grep results
    const combined = [
      ...ragResults,
      ...grepResults,
    ];
    results = deduplicateResults(combined).slice(0, topK * 2);
  }

  return {
    results,
    num_results: results.length,
    papers_searched: papers.length,
    query,
    mode,
    success: true,
  };
}
