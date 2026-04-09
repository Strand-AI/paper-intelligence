import type { Env, FlatHeader, Chunk } from "./types";

const HEADER_PATTERN = /^(#{1,6})\s+(.+)$/;
const CHUNK_SIZE = 512;
const CHUNK_OVERLAP = 50;
const EMBEDDING_BATCH_SIZE = 100;
const VECTORIZE_BATCH_SIZE = 1000;
const MAX_METADATA_TEXT = 8000; // Vectorize has 10KB metadata limit per vector

/**
 * Extract markdown headers with hierarchical paths.
 * Matches the Python MarkdownParser.build_flat_headers() logic.
 */
export function extractHeaders(markdown: string): FlatHeader[] {
  const lines = markdown.split("\n");
  const headers: FlatHeader[] = [];
  const stack: { level: number; text: string }[] = [];

  for (let i = 0; i < lines.length; i++) {
    const match = lines[i].match(HEADER_PATTERN);
    if (match) {
      const level = match[1].length;
      const text = match[2].trim();

      // Pop stack until we find a parent (strictly lower level number = higher heading)
      while (stack.length > 0 && stack[stack.length - 1].level >= level) {
        stack.pop();
      }

      stack.push({ level, text });

      headers.push({
        level,
        text,
        line: i + 1, // 1-indexed
        path: stack.map((h) => h.text).join(" > "),
      });
    }
  }

  return headers;
}

/**
 * Find the nearest header at or before a given line number.
 */
export function getHeaderAtLine(
  headers: FlatHeader[],
  lineNumber: number,
): FlatHeader | null {
  let result: FlatHeader | null = null;
  for (const header of headers) {
    if (header.line <= lineNumber) {
      result = header;
    } else {
      break;
    }
  }
  return result;
}

/**
 * Chunk markdown text into overlapping segments.
 * Respects paragraph boundaries and tracks line numbers.
 * Matches the Python MarkdownParser.chunk_text() logic:
 *   - chunk_size: 512 characters
 *   - chunk_overlap: 50 characters
 */
export function chunkText(
  markdown: string,
  headers: FlatHeader[],
  chunkSize: number = CHUNK_SIZE,
  chunkOverlap: number = CHUNK_OVERLAP,
): Chunk[] {
  const lines = markdown.split("\n");

  // Step 1: Split into paragraphs (groups of non-empty lines)
  interface Paragraph {
    text: string;
    startLine: number;
    endLine: number;
  }

  const paragraphs: Paragraph[] = [];
  let currentPara: string[] = [];
  let currentStart = 1;

  for (let i = 0; i < lines.length; i++) {
    const lineNum = i + 1;
    if (lines[i].trim() === "") {
      if (currentPara.length > 0) {
        paragraphs.push({
          text: currentPara.join("\n"),
          startLine: currentStart,
          endLine: lineNum - 1,
        });
        currentPara = [];
      }
      currentStart = lineNum + 1;
    } else {
      if (currentPara.length === 0) {
        currentStart = lineNum;
      }
      currentPara.push(lines[i]);
    }
  }
  if (currentPara.length > 0) {
    paragraphs.push({
      text: currentPara.join("\n"),
      startLine: currentStart,
      endLine: lines.length,
    });
  }

  // Step 2: Accumulate paragraphs into chunks
  const chunks: Chunk[] = [];
  let chunkText = "";
  let chunkStartLine: number | null = null;
  let chunkEndLine = 0;
  let prevSuffix = "";

  for (const para of paragraphs) {
    const wouldExceed =
      chunkText.length > 0 &&
      chunkText.length + para.text.length + 2 > chunkSize; // +2 for "\n\n"

    if (wouldExceed) {
      // Finalize current chunk
      const header = getHeaderAtLine(headers, chunkStartLine ?? 1);
      chunks.push({
        text: chunkText,
        header_path: header?.path ?? "",
        start_line: chunkStartLine ?? 1,
        end_line: chunkEndLine,
        chunk_index: chunks.length,
      });

      // Start new chunk with overlap from previous
      prevSuffix = chunkText.slice(-chunkOverlap);
      chunkText = prevSuffix + "\n\n" + para.text;
      chunkStartLine = para.startLine;
      chunkEndLine = para.endLine;
    } else {
      if (chunkText.length > 0) {
        chunkText += "\n\n" + para.text;
      } else {
        chunkText = prevSuffix ? prevSuffix + para.text : para.text;
        chunkStartLine = para.startLine;
      }
      chunkEndLine = para.endLine;
    }
  }

  // Don't forget last chunk
  if (chunkText.length > 0) {
    const header = getHeaderAtLine(headers, chunkStartLine ?? 1);
    chunks.push({
      text: chunkText,
      header_path: header?.path ?? "",
      start_line: chunkStartLine ?? 1,
      end_line: chunkEndLine,
      chunk_index: chunks.length,
    });
  }

  return chunks;
}

/**
 * Generate embeddings for text chunks using Workers AI.
 * Handles batching (max 100 per API call).
 */
async function embedTexts(env: Env, texts: string[]): Promise<number[][]> {
  const allEmbeddings: number[][] = [];

  for (let i = 0; i < texts.length; i += EMBEDDING_BATCH_SIZE) {
    const batch = texts.slice(i, i + EMBEDDING_BATCH_SIZE);
    const response = await env.AI.run("@cf/baai/bge-small-en-v1.5", {
      text: batch,
    });
    const result = response as { data: number[][] };
    allEmbeddings.push(...result.data);
  }

  return allEmbeddings;
}

/**
 * Full processing pipeline: extract headers, chunk, embed, store.
 * Called after markdown is available (either uploaded directly or converted from PDF).
 */
export async function processPaper(
  env: Env,
  paperId: string,
  name: string,
  markdown: string,
): Promise<void> {
  try {
    // Update status: indexing
    await env.DB.prepare(
      "UPDATE papers SET status = 'indexing' WHERE id = ?",
    )
      .bind(paperId)
      .run();

    // Store markdown in D1 for grep search
    await env.DB.prepare(
      "UPDATE papers SET markdown_text = ?, markdown_length = ? WHERE id = ?",
    )
      .bind(markdown, markdown.length, paperId)
      .run();

    // Store markdown in R2 for archival
    await env.BUCKET.put(`papers/${paperId}/paper.md`, markdown);

    // Extract headers and auto-parse title (first H1)
    const headers = extractHeaders(markdown);
    const firstH1 = headers.find((h) => h.level === 1);
    if (firstH1) {
      // Clean markdown/HTML artifacts from title
      const cleanTitle = firstH1.text
        .replace(/\*\*/g, "")
        .replace(/<[^>]+>/g, "")
        .trim();
      await env.DB.prepare("UPDATE papers SET title = ? WHERE id = ?")
        .bind(cleanTitle, paperId)
        .run();
    }

    // Store headers in D1
    if (headers.length > 0) {
      // Batch insert headers
      const stmt = env.DB.prepare(
        "INSERT INTO headers (paper_id, level, text, line_number, path) VALUES (?, ?, ?, ?, ?)",
      );
      const batch = headers.map((h) =>
        stmt.bind(paperId, h.level, h.text, h.line, h.path),
      );
      // D1 batch limit is 100 statements
      for (let i = 0; i < batch.length; i += 100) {
        await env.DB.batch(batch.slice(i, i + 100));
      }
    }

    // Update status: embedding
    await env.DB.prepare(
      "UPDATE papers SET status = 'embedding' WHERE id = ?",
    )
      .bind(paperId)
      .run();

    // Chunk the markdown
    const chunks = chunkText(markdown, headers);

    // Generate embeddings
    const texts = chunks.map((c) => c.text);
    const embeddings = await embedTexts(env, texts);

    // Store vectors in Vectorize
    const vectors = chunks.map((chunk, i) => ({
      id: `${paperId}_${chunk.chunk_index}`,
      values: embeddings[i],
      metadata: {
        paper_id: paperId,
        paper_name: name,
        text: chunk.text.slice(0, MAX_METADATA_TEXT),
        header_path: chunk.header_path,
        start_line: chunk.start_line,
        end_line: chunk.end_line,
        chunk_index: chunk.chunk_index,
      },
    }));

    for (let i = 0; i < vectors.length; i += VECTORIZE_BATCH_SIZE) {
      await env.VECTORIZE.upsert(vectors.slice(i, i + VECTORIZE_BATCH_SIZE));
    }

    // Update paper status to ready
    await env.DB.prepare(
      "UPDATE papers SET status = 'ready', chunk_count = ?, processed_at = datetime('now') WHERE id = ?",
    )
      .bind(chunks.length, paperId)
      .run();
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    await env.DB.prepare(
      "UPDATE papers SET status = 'error', error = ? WHERE id = ?",
    )
      .bind(message, paperId)
      .run();
    throw err;
  }
}

/**
 * Convert a PDF via the Marker container, then index + embed.
 * Runs in background via ctx.waitUntil() — the upload returns 202 immediately.
 */
export async function convertAndProcessPaper(
  env: Env,
  paperId: string,
  name: string,
  pdfData: ArrayBuffer,
): Promise<void> {
  try {
    await env.DB.prepare("UPDATE papers SET status = 'converting' WHERE id = ?")
      .bind(paperId)
      .run();

    // Send PDF to Marker container
    const containerId = env.MARKER_CONTAINER.idFromName("marker");
    const container = env.MARKER_CONTAINER.get(containerId);
    const response = await container.fetch("http://container/convert", {
      method: "POST",
      body: pdfData,
      headers: { "Content-Type": "application/pdf" },
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`Marker conversion failed (${response.status}): ${error}`);
    }

    const result = (await response.json()) as {
      markdown: string;
      images: { name: string; data: string }[];
      page_count?: number;
    };

    // Store images in R2
    if (result.images.length > 0) {
      const imgStmt = env.DB.prepare(
        "INSERT INTO images (paper_id, r2_key, filename) VALUES (?, ?, ?)",
      );
      const imgBatch: D1PreparedStatement[] = [];
      for (const img of result.images) {
        const key = `papers/${paperId}/images/${img.name}`;
        // Decode base64 to binary
        const binary = Uint8Array.from(atob(img.data), (c) => c.charCodeAt(0));
        await env.BUCKET.put(key, binary);
        imgBatch.push(imgStmt.bind(paperId, key, img.name));
      }
      for (let i = 0; i < imgBatch.length; i += 100) {
        await env.DB.batch(imgBatch.slice(i, i + 100));
      }
    }

    // Update page count if available
    if (result.page_count) {
      await env.DB.prepare("UPDATE papers SET page_count = ? WHERE id = ?")
        .bind(result.page_count, paperId)
        .run();
    }

    // Now index + embed the markdown
    await processPaper(env, paperId, name, result.markdown);
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    await env.DB.prepare(
      "UPDATE papers SET status = 'error', error = ? WHERE id = ?",
    )
      .bind(message, paperId)
      .run();
    throw err;
  }
}

/**
 * Delete all data for a paper: D1 rows, R2 objects, Vectorize vectors.
 */
export async function deletePaper(env: Env, paperId: string): Promise<void> {
  // Get paper info for cleanup
  const paper = await env.DB.prepare("SELECT * FROM papers WHERE id = ?")
    .bind(paperId)
    .first();
  if (!paper) throw new Error(`Paper ${paperId} not found`);

  // Delete Vectorize vectors
  const chunkCount = (paper.chunk_count as number) ?? 0;
  if (chunkCount > 0) {
    const ids = Array.from({ length: chunkCount }, (_, i) => `${paperId}_${i}`);
    // Vectorize deleteByIds limit is 100 per call
    for (let i = 0; i < ids.length; i += 100) {
      await env.VECTORIZE.deleteByIds(ids.slice(i, i + 100));
    }
  }

  // Delete R2 objects
  const r2Objects = await env.BUCKET.list({ prefix: `papers/${paperId}/` });
  if (r2Objects.objects.length > 0) {
    await env.BUCKET.delete(r2Objects.objects.map((o) => o.key));
  }

  // Delete D1 rows (cascade handles headers + images)
  await env.DB.prepare("DELETE FROM headers WHERE paper_id = ?")
    .bind(paperId)
    .run();
  await env.DB.prepare("DELETE FROM images WHERE paper_id = ?")
    .bind(paperId)
    .run();
  await env.DB.prepare("DELETE FROM papers WHERE id = ?").bind(paperId).run();
}
