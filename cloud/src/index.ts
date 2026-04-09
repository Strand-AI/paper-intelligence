import type { Env, Paper } from "./types";
import { processPaper, convertAndProcessPaper, deletePaper } from "./pipeline";
import { PaperIntelligenceMCP } from "./mcp";
import { handleChat } from "./chat";
import { HTML } from "./ui";
// Uncomment when Containers beta is enabled:
// export { MarkerContainer } from "./container";

export { PaperIntelligenceMCP };

export default {
  async fetch(
    request: Request,
    env: Env,
    ctx: ExecutionContext,
  ): Promise<Response> {
    const url = new URL(request.url);

    // CORS preflight
    if (request.method === "OPTIONS") {
      return new Response(null, {
        headers: {
          "Access-Control-Allow-Origin": "*",
          "Access-Control-Allow-Methods": "GET, POST, DELETE, OPTIONS",
          "Access-Control-Allow-Headers": "Authorization, Content-Type",
        },
      });
    }

    // UI — no auth required (auth is in the UI itself)
    if (url.pathname === "/" || url.pathname === "/index.html") {
      return new Response(HTML, {
        headers: { "Content-Type": "text/html; charset=utf-8" },
      });
    }

    // Auth check for all API routes
    const authHeader = request.headers.get("Authorization");
    if (authHeader !== `Bearer ${env.API_TOKEN}`) {
      return Response.json({ error: "Unauthorized" }, { status: 401 });
    }

    // MCP endpoint — Streamable HTTP
    if (url.pathname.startsWith("/mcp")) {
      return (
        PaperIntelligenceMCP.serve("/mcp") as {
          fetch: (req: Request, env: Env, ctx: ExecutionContext) => Promise<Response>;
        }
      ).fetch(request, env, ctx);
    }

    // Chat endpoint
    if (url.pathname === "/chat" && request.method === "POST") {
      return handleChat(request, env);
    }

    // REST: Upload paper
    if (url.pathname === "/papers" && request.method === "POST") {
      return handleUpload(request, env, ctx);
    }

    // REST: List papers
    if (url.pathname === "/papers" && request.method === "GET") {
      return handleListPapers(env);
    }

    // REST: Paper content (markdown text for rendering)
    const contentMatch = url.pathname.match(
      /^\/papers\/([\w-]+)\/content$/,
    );
    if (contentMatch && request.method === "GET") {
      return handleGetPaperContent(contentMatch[1], env);
    }

    // REST: Get paper
    const paperMatch = url.pathname.match(/^\/papers\/([\w-]+)$/);
    if (paperMatch && request.method === "GET") {
      return handleGetPaper(paperMatch[1], env);
    }

    // REST: Delete paper
    if (paperMatch && request.method === "DELETE") {
      return handleDeletePaper(paperMatch[1], env);
    }

    return Response.json({ error: "Not found" }, { status: 404 });
  },
} satisfies ExportedHandler<Env>;

// --- REST Handlers ---

async function handleUpload(
  request: Request,
  env: Env,
  ctx: ExecutionContext,
): Promise<Response> {
  const contentType = request.headers.get("Content-Type") ?? "";

  let name: string;
  let markdown: string | null = null;
  let pdfData: ArrayBuffer | null = null;

  if (contentType.includes("application/json")) {
    // JSON upload: { name, markdown }
    const body = (await request.json()) as {
      name?: string;
      markdown?: string;
    };
    if (!body.name) {
      return Response.json(
        { error: "Missing 'name' field" },
        { status: 400 },
      );
    }
    name = body.name;
    markdown = body.markdown ?? null;
  } else if (contentType.includes("multipart/form-data")) {
    // Multipart upload: PDF file + optional name
    const formData = await request.formData();
    const file = formData.get("file") as File | null;
    name = (formData.get("name") as string) ?? file?.name?.replace(/\.pdf$/i, "") ?? "unnamed";
    if (file) {
      pdfData = await file.arrayBuffer();
    }
    const md = formData.get("markdown") as string | null;
    if (md) markdown = md;
  } else {
    return Response.json(
      { error: "Content-Type must be application/json or multipart/form-data" },
      { status: 400 },
    );
  }

  if (!markdown && !pdfData) {
    return Response.json(
      { error: "Provide either 'markdown' text or a PDF 'file'" },
      { status: 400 },
    );
  }

  // Generate ID
  const id = crypto.randomUUID();

  // Insert paper record
  await env.DB.prepare(
    "INSERT INTO papers (id, name, status) VALUES (?, ?, ?)",
  )
    .bind(id, name, markdown ? "indexing" : "uploading")
    .run();

  // Store PDF in R2 if provided
  if (pdfData) {
    const pdfKey = `papers/${id}/paper.pdf`;
    await env.BUCKET.put(pdfKey, pdfData);
    await env.DB.prepare("UPDATE papers SET pdf_key = ? WHERE id = ?")
      .bind(pdfKey, id)
      .run();
  }

  if (markdown) {
    // Process immediately in background
    ctx.waitUntil(processPaper(env, id, name, markdown));
    return Response.json({ id, name, status: "indexing" }, { status: 202 });
  }

  // PDF upload: convert via Marker container (when available)
  if (env.MARKER_CONTAINER) {
    ctx.waitUntil(convertAndProcessPaper(env, id, name, pdfData!));
    return Response.json(
      { id, name, status: "converting" },
      { status: 202 },
    );
  }

  // Container not available — PDF stored, needs manual markdown upload
  return Response.json(
    {
      id,
      name,
      status: "uploading",
      message:
        "PDF stored in R2. Marker container not yet enabled — re-upload with 'markdown' field, or enable Containers beta.",
    },
    { status: 202 },
  );
}

async function handleListPapers(env: Env): Promise<Response> {
  const { results } = await env.DB.prepare(
    "SELECT id, name, status, page_count, chunk_count, created_at, processed_at FROM papers ORDER BY created_at DESC",
  ).all();
  return Response.json({ papers: results });
}

async function handleGetPaper(id: string, env: Env): Promise<Response> {
  const paper = await env.DB.prepare("SELECT * FROM papers WHERE id = ?")
    .bind(id)
    .first<Paper>();
  if (!paper) {
    return Response.json({ error: "Not found" }, { status: 404 });
  }
  // Don't return full markdown_text in REST response
  const { markdown_text: _, ...info } = paper;
  return Response.json(info);
}

async function handleGetPaperContent(
  id: string,
  env: Env,
): Promise<Response> {
  const paper = await env.DB.prepare(
    "SELECT id, name, markdown_text FROM papers WHERE id = ?",
  )
    .bind(id)
    .first<{ id: string; name: string; markdown_text: string | null }>();
  if (!paper) {
    return Response.json({ error: "Not found" }, { status: 404 });
  }
  return Response.json({ id: paper.id, name: paper.name, markdown: paper.markdown_text });
}

async function handleDeletePaper(id: string, env: Env): Promise<Response> {
  try {
    await deletePaper(env, id);
    return Response.json({ success: true, id });
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    return Response.json({ error: message }, { status: 404 });
  }
}
