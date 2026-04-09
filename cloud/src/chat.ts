import type { Env } from "./types";
import { search } from "./search";

/**
 * RAG chat: search for relevant chunks, then stream an LLM response.
 * Normalizes both OpenAI and Anthropic SSE formats to: data: {"text":"..."}\n\n
 */
export async function handleChat(
  request: Request,
  env: Env,
): Promise<Response> {
  const body = (await request.json()) as {
    message: string;
    paper_ids?: string[];
    model?: string;
    history?: { role: string; content: string }[];
  };

  if (!body.message) {
    return Response.json({ error: "Missing 'message'" }, { status: 400 });
  }

  const model = body.model || "claude-opus-4-6";
  const isOpenAI =
    model.startsWith("gpt") || model.startsWith("o") || model.startsWith("chatgpt");

  // Check API key availability
  if (isOpenAI && !env.OPENAI_API_KEY) {
    return Response.json(
      { error: "OPENAI_API_KEY not configured" },
      { status: 500 },
    );
  }
  if (!isOpenAI && !env.ANTHROPIC_API_KEY) {
    return Response.json(
      { error: "ANTHROPIC_API_KEY not configured" },
      { status: 500 },
    );
  }

  // RAG: search for relevant chunks
  const searchResponse = await search(
    env,
    body.message,
    "hybrid",
    10,
    body.paper_ids || [],
    false,
    false,
  );

  // Build context from search results
  const context = searchResponse.results
    .map(
      (r) =>
        `[${r.paper_name}] [${r.header_context || "General"}] (line ${r.line_number})\n${r.content}`,
    )
    .join("\n\n---\n\n");

  const systemPrompt = `You are a research paper assistant. Answer questions based on the following excerpts from the user's paper library.

When referencing information, cite the paper name and section. If the excerpts don't contain enough information, say so.

${context ? `--- Paper Excerpts ---\n\n${context}` : "(No relevant excerpts found)"}`;

  // Build messages
  const history = body.history || [];
  const messages = [
    ...history,
    { role: "user" as const, content: body.message },
  ];

  // Stream LLM response
  let upstreamResponse: Response;

  if (isOpenAI) {
    upstreamResponse = await fetch(
      "https://api.openai.com/v1/chat/completions",
      {
        method: "POST",
        headers: {
          Authorization: `Bearer ${env.OPENAI_API_KEY}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          model,
          messages: [{ role: "system", content: systemPrompt }, ...messages],
          stream: true,
        }),
      },
    );
  } else {
    upstreamResponse = await fetch("https://api.anthropic.com/v1/messages", {
      method: "POST",
      headers: {
        "x-api-key": env.ANTHROPIC_API_KEY!,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model,
        max_tokens: 8192,
        system: systemPrompt,
        messages,
        stream: true,
      }),
    });
  }

  if (!upstreamResponse.ok) {
    const err = await upstreamResponse.text();
    return Response.json(
      { error: `LLM API error (${upstreamResponse.status}): ${err}` },
      { status: 502 },
    );
  }

  // Normalize the SSE stream
  const normalized = upstreamResponse.body!.pipeThrough(
    createNormalizer(isOpenAI ? "openai" : "anthropic"),
  );

  return new Response(normalized, {
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      Connection: "keep-alive",
    },
  });
}

function createNormalizer(
  provider: "openai" | "anthropic",
): TransformStream<Uint8Array, Uint8Array> {
  const encoder = new TextEncoder();
  const decoder = new TextDecoder();
  let buffer = "";

  return new TransformStream({
    transform(chunk, controller) {
      buffer += decoder.decode(chunk, { stream: true });
      const parts = buffer.split("\n");
      buffer = parts.pop() || "";

      for (const line of parts) {
        if (!line.startsWith("data: ")) continue;
        const data = line.slice(6).trim();
        if (!data) continue;
        if (data === "[DONE]") {
          controller.enqueue(encoder.encode("data: [DONE]\n\n"));
          continue;
        }
        try {
          const parsed = JSON.parse(data);
          let text = "";
          if (provider === "openai") {
            text = parsed.choices?.[0]?.delta?.content || "";
          } else if (parsed.type === "content_block_delta") {
            text = parsed.delta?.text || "";
          }
          if (text) {
            controller.enqueue(
              encoder.encode(`data: ${JSON.stringify({ text })}\n\n`),
            );
          }
        } catch {
          // skip unparseable lines
        }
      }
    },
    flush(controller) {
      controller.enqueue(encoder.encode("data: [DONE]\n\n"));
    },
  });
}
