import type { Env, Paper } from "./types";
import { processPaper, convertAndProcessPaper, deletePaper } from "./pipeline";
import { PaperIntelligenceMCP } from "./mcp";
import { handleChat } from "./chat";
import { HTML } from "./ui";
export { MarkerContainer } from "./container";

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
          "Access-Control-Allow-Methods": "GET, POST, PUT, PATCH, DELETE, OPTIONS",
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

    // Match /papers/:id sub-routes
    const subMatch = url.pathname.match(/^\/papers\/([\w-]+)\/(content|pdf|headers)$/);
    if (subMatch && request.method === "GET") {
      const [, id, sub] = subMatch;
      if (sub === "content") return handleGetPaperContent(id, env);
      if (sub === "pdf") return handleGetPaperPdf(id, env);
      if (sub === "headers") return handleGetPaperHeaders(id, env);
    }

    // REST: Get / Update / Delete paper
    const paperMatch = url.pathname.match(/^\/papers\/([\w-]+)$/);
    if (paperMatch) {
      const id = paperMatch[1];
      if (request.method === "GET") return handleGetPaper(id, env);
      if (request.method === "PATCH") return handleUpdatePaper(request, id, env);
      if (request.method === "DELETE") return handleDeletePaper(id, env);
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
    const formData = await request.formData();
    const file = formData.get("file") as File | null;
    name =
      (formData.get("name") as string) ??
      file?.name?.replace(/\.pdf$/i, "") ??
      "unnamed";
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

  const id = crypto.randomUUID();

  await env.DB.prepare(
    "INSERT INTO papers (id, name, status) VALUES (?, ?, ?)",
  )
    .bind(id, name, markdown ? "indexing" : "uploading")
    .run();

  if (pdfData) {
    const pdfKey = `papers/${id}/paper.pdf`;
    await env.BUCKET.put(pdfKey, pdfData);
    await env.DB.prepare("UPDATE papers SET pdf_key = ? WHERE id = ?")
      .bind(pdfKey, id)
      .run();
  }

  if (markdown) {
    ctx.waitUntil(processPaper(env, id, name, markdown));
    return Response.json({ id, name, status: "indexing" }, { status: 202 });
  }

  if (env.MARKER_CONTAINER) {
    ctx.waitUntil(convertAndProcessPaper(env, id, name, pdfData!));
    return Response.json(
      { id, name, status: "converting" },
      { status: 202 },
    );
  }

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
    "SELECT id, name, title, alias, status, error, pdf_key, page_count, chunk_count, created_at, processed_at FROM papers ORDER BY created_at DESC",
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
  const { markdown_text: _, ...info } = paper;
  return Response.json(info);
}

async function handleUpdatePaper(
  request: Request,
  id: string,
  env: Env,
): Promise<Response> {
  const body = (await request.json()) as { alias?: string };
  if (body.alias !== undefined) {
    await env.DB.prepare("UPDATE papers SET alias = ? WHERE id = ?")
      .bind(body.alias || null, id)
      .run();
  }
  return Response.json({ success: true, id });
}

async function handleGetPaperContent(
  id: string,
  env: Env,
): Promise<Response> {
  const paper = await env.DB.prepare(
    "SELECT id, name, title, alias, markdown_text FROM papers WHERE id = ?",
  )
    .bind(id)
    .first();
  if (!paper) {
    return Response.json({ error: "Not found" }, { status: 404 });
  }

  // Include headers for outline
  const { results: headers } = await env.DB.prepare(
    "SELECT level, text, line_number, path FROM headers WHERE paper_id = ? ORDER BY line_number",
  )
    .bind(id)
    .all();

  return Response.json({
    id: paper.id,
    name: paper.name,
    title: paper.title,
    alias: paper.alias,
    markdown: paper.markdown_text,
    headers,
  });
}

async function handleGetPaperPdf(id: string, env: Env): Promise<Response> {
  const paper = await env.DB.prepare(
    "SELECT pdf_key FROM papers WHERE id = ?",
  )
    .bind(id)
    .first<{ pdf_key: string | null }>();

  if (!paper?.pdf_key) {
    return Response.json({ error: "No PDF available" }, { status: 404 });
  }

  const obj = await env.BUCKET.get(paper.pdf_key);
  if (!obj) {
    return Response.json({ error: "PDF not found in R2" }, { status: 404 });
  }

  return new Response(obj.body, {
    headers: {
      "Content-Type": "application/pdf",
      "Content-Disposition": "inline",
    },
  });
}

async function handleGetPaperHeaders(
  id: string,
  env: Env,
): Promise<Response> {
  const { results } = await env.DB.prepare(
    "SELECT level, text, line_number, path FROM headers WHERE paper_id = ? ORDER BY line_number",
  )
    .bind(id)
    .all();
  return Response.json({ headers: results });
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
