export interface Env {
  DB: D1Database;
  BUCKET: R2Bucket;
  VECTORIZE: VectorizeIndex;
  AI: Ai;
  MARKER_CONTAINER: DurableObjectNamespace;
  API_TOKEN: string;
  OPENAI_API_KEY?: string;
  ANTHROPIC_API_KEY?: string;
}

export interface Paper {
  id: string;
  name: string;
  pdf_key: string | null;
  markdown_key: string | null;
  markdown_text: string | null;
  title: string | null;
  alias: string | null;
  status: "uploading" | "converting" | "indexing" | "embedding" | "ready" | "error";
  error: string | null;
  page_count: number | null;
  markdown_length: number | null;
  chunk_count: number | null;
  created_at: string;
  processed_at: string | null;
  version: string;
}

export interface FlatHeader {
  level: number;
  text: string;
  line: number;
  path: string;
}

export interface Chunk {
  text: string;
  header_path: string;
  start_line: number;
  end_line: number;
  chunk_index: number;
}

export interface SearchResult {
  paper_id: string;
  paper_name: string;
  content: string;
  line_number: number;
  end_line?: number;
  header_context: string;
  score: number;
  match_type: "grep" | "rag";
  context?: string;
}

export interface SearchResponse {
  results: SearchResult[];
  num_results: number;
  papers_searched: number;
  query: string;
  mode: string;
  success: boolean;
}
