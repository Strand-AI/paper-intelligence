/**
 * Paper Intelligence Worker — thin Vectorize proxy.
 *
 * POST /upsert   — store vectors with metadata
 * POST /query    — similarity search
 * POST /delete   — delete vectors by paper ID prefix
 */

interface Env {
  VECTORIZE: VectorizeIndex;
  API_TOKEN: string;
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    // Auth
    if (request.method === "OPTIONS") {
      return new Response(null, { status: 204 });
    }
    const auth = request.headers.get("Authorization");
    if (auth !== `Bearer ${env.API_TOKEN}`) {
      return Response.json({ error: "Unauthorized" }, { status: 401 });
    }

    const url = new URL(request.url);

    if (url.pathname === "/upsert" && request.method === "POST") {
      return handleUpsert(request, env);
    }
    if (url.pathname === "/query" && request.method === "POST") {
      return handleQuery(request, env);
    }
    if (url.pathname === "/delete" && request.method === "POST") {
      return handleDelete(request, env);
    }

    return Response.json({ error: "Not found" }, { status: 404 });
  },
} satisfies ExportedHandler<Env>;

async function handleUpsert(request: Request, env: Env): Promise<Response> {
  const body = (await request.json()) as {
    vectors: {
      id: string;
      values: number[];
      metadata?: Record<string, string | number | boolean>;
    }[];
  };

  // Vectorize upsert batch limit is 100
  for (let i = 0; i < body.vectors.length; i += 100) {
    await env.VECTORIZE.upsert(body.vectors.slice(i, i + 100));
  }

  return Response.json({ success: true, count: body.vectors.length });
}

async function handleQuery(request: Request, env: Env): Promise<Response> {
  const body = (await request.json()) as {
    vector: number[];
    top_k?: number;
    filter?: Record<string, string>;
  };

  const result = await env.VECTORIZE.query(body.vector, {
    topK: body.top_k || 5,
    filter: body.filter,
    returnMetadata: "all",
  });

  return Response.json({
    matches: result.matches.map((m) => ({
      id: m.id,
      score: m.score,
      metadata: m.metadata,
    })),
  });
}

async function handleDelete(request: Request, env: Env): Promise<Response> {
  const body = (await request.json()) as { ids: string[] };

  // Vectorize deleteByIds limit is 100
  for (let i = 0; i < body.ids.length; i += 100) {
    await env.VECTORIZE.deleteByIds(body.ids.slice(i, i + 100));
  }

  return Response.json({ success: true, deleted: body.ids.length });
}
