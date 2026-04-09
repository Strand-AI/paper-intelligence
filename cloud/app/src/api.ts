const API = "";

function getToken(): string {
  return localStorage.getItem("pi_token") || "";
}

function authHeaders(): Record<string, string> {
  return {
    Authorization: `Bearer ${getToken()}`,
    "Content-Type": "application/json",
  };
}

export interface Paper {
  id: string;
  name: string;
  title: string | null;
  alias: string | null;
  status: string;
  error: string | null;
  pdf_key: string | null;
  page_count: number | null;
  chunk_count: number | null;
  created_at: string;
  processed_at: string | null;
}

export interface Header {
  level: number;
  text: string;
  line_number: number;
  path: string;
}

export interface PaperContent {
  id: string;
  name: string;
  title: string | null;
  alias: string | null;
  markdown: string | null;
  headers: Header[];
}

export async function listPapers(): Promise<Paper[]> {
  const res = await fetch(`${API}/papers`, { headers: authHeaders() });
  if (res.status === 401) throw new Error("Unauthorized");
  const data = await res.json();
  return data.papers || [];
}

export async function getPaperContent(id: string): Promise<PaperContent> {
  const res = await fetch(`${API}/papers/${id}/content`, {
    headers: authHeaders(),
  });
  if (!res.ok) throw new Error("Failed to load paper");
  return res.json();
}

export async function getPaperPdf(id: string): Promise<string> {
  const res = await fetch(`${API}/papers/${id}/pdf`, {
    headers: { Authorization: `Bearer ${getToken()}` },
  });
  if (!res.ok) throw new Error("Failed to load PDF");
  const blob = await res.blob();
  return URL.createObjectURL(blob);
}

export async function updatePaperAlias(
  id: string,
  alias: string,
): Promise<void> {
  await fetch(`${API}/papers/${id}`, {
    method: "PATCH",
    headers: authHeaders(),
    body: JSON.stringify({ alias: alias || null }),
  });
}

export async function deletePaper(id: string): Promise<void> {
  await fetch(`${API}/papers/${id}`, {
    method: "DELETE",
    headers: authHeaders(),
  });
}

export async function uploadPapers(
  files: File[],
  onProgress?: (name: string, status: string) => void,
): Promise<{ id: string; name: string }[]> {
  const results: { id: string; name: string }[] = [];
  for (const file of files) {
    const name = file.name.replace(/\.pdf$/i, "");
    onProgress?.(name, "uploading");
    const form = new FormData();
    form.append("name", name);
    form.append("file", file);
    const res = await fetch(`${API}/papers`, {
      method: "POST",
      headers: { Authorization: `Bearer ${getToken()}` },
      body: form,
    });
    if (res.ok) {
      const data = await res.json();
      results.push({ id: data.id, name });
      onProgress?.(name, "done");
    } else {
      onProgress?.(name, "error");
    }
  }
  return results;
}

export async function streamChat(
  message: string,
  paperIds: string[],
  history: { role: string; content: string }[],
  onChunk: (text: string) => void,
): Promise<string> {
  const res = await fetch(`${API}/chat`, {
    method: "POST",
    headers: authHeaders(),
    body: JSON.stringify({
      message,
      paper_ids: paperIds,
      model: "claude-opus-4-6",
      history,
    }),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(
      (err as { error?: string }).error || `Error ${res.status}`,
    );
  }

  const reader = res.body!.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let fullText = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split("\n\n");
    buffer = parts.pop() || "";
    for (const part of parts) {
      for (const line of part.split("\n")) {
        if (!line.startsWith("data: ")) continue;
        const data = line.slice(6).trim();
        if (data === "[DONE]") continue;
        try {
          const parsed = JSON.parse(data);
          if (parsed.text) {
            fullText += parsed.text;
            onChunk(fullText);
          }
        } catch {
          // skip
        }
      }
    }
  }
  return fullText;
}
