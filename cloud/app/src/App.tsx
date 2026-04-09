import { useState, useEffect, useCallback, useRef, useMemo } from "react";
import ReactMarkdown from "react-markdown";
import rehypeRaw from "rehype-raw";
import remarkGfm from "remark-gfm";
import {
  FileText,
  BookOpen,
  MessageSquare,
  Upload,
  Search,
  Pencil,
  X,
} from "lucide-react";
import Fuse from "fuse.js";
import {
  listPapers,
  getPaperContent,
  getPaperPdf,
  updatePaperAlias,
  uploadPapers,
  streamChat,
  type Paper,
  type Header,
} from "./api";
import { cn, displayName } from "./lib/utils";

export function App() {
  const [token, setToken] = useState(
    () => localStorage.getItem("pi_token") || "",
  );
  const [authed, setAuthed] = useState(!!token);
  const [tokenInput, setTokenInput] = useState("");

  function login() {
    if (!tokenInput.trim()) return;
    localStorage.setItem("pi_token", tokenInput.trim());
    setToken(tokenInput.trim());
    setAuthed(true);
  }

  function logout() {
    localStorage.removeItem("pi_token");
    setToken("");
    setAuthed(false);
  }

  if (!authed) {
    return (
      <div className="fixed inset-0 bg-ink/30 flex items-center justify-center">
        <div className="bg-surface border border-border rounded-[var(--radius-sm)] p-8 w-[400px] max-w-[90vw]">
          <h2 className="text-lg font-medium tracking-tight mb-1">
            Paper Intelligence
          </h2>
          <p className="text-muted text-sm mb-6">
            Enter your API token to continue.
          </p>
          <input
            type="password"
            className="w-full px-3 py-2.5 border border-border rounded-[var(--radius-sm)] text-sm outline-none focus:border-brand"
            placeholder="Token"
            value={tokenInput}
            onChange={(e) => setTokenInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && login()}
            autoFocus
          />
          <button
            onClick={login}
            className="w-full mt-4 py-2.5 bg-brand text-white rounded-[var(--radius-sm)] text-sm font-medium hover:bg-brand-hover cursor-pointer"
          >
            Continue
          </button>
        </div>
      </div>
    );
  }

  return <Main token={token} onLogout={logout} />;
}

function Main({ token, onLogout }: { token: string; onLogout: () => void }) {
  const [papers, setPapers] = useState<Paper[]>([]);
  const [activePaperId, setActivePaperId] = useState<string | null>(null);
  const [filter, setFilter] = useState("");
  const [markdown, setMarkdown] = useState<string | null>(null);
  const [headers, setHeaders] = useState<Header[]>([]);
  const [view, setView] = useState<"pdf" | "md">("pdf");
  const [pdfUrl, setPdfUrl] = useState<string | null>(null);
  const [showOutline, setShowOutline] = useState(false);
  const [showChat, setShowChat] = useState(false);
  const [showUpload, setShowUpload] = useState(false);
  const [aliasModal, setAliasModal] = useState<Paper | null>(null);
  const [aliasValue, setAliasValue] = useState("");

  const activePaper = papers.find((p) => p.id === activePaperId) || null;

  // Load papers
  const loadPapers = useCallback(async () => {
    try {
      const list = await listPapers();
      setPapers(list);
    } catch {
      onLogout();
    }
  }, [onLogout]);

  useEffect(() => {
    loadPapers();
  }, [loadPapers]);

  // Poll for processing papers
  useEffect(() => {
    const processing = papers.some((p) =>
      ["uploading", "converting", "indexing", "embedding"].includes(p.status),
    );
    if (!processing) return;
    const id = setInterval(loadPapers, 3000);
    return () => clearInterval(id);
  }, [papers, loadPapers]);

  // Select paper
  async function selectPaper(id: string) {
    setActivePaperId(id);
    setMarkdown(null);
    setPdfUrl(null);
    const paper = papers.find((p) => p.id === id);
    setView(paper?.pdf_key ? "pdf" : "md");
    try {
      const content = await getPaperContent(id);
      setMarkdown(content.markdown);
      setHeaders(content.headers);
      if (paper?.pdf_key) {
        const url = await getPaperPdf(id);
        setPdfUrl(url);
      }
    } catch {
      // error
    }
  }

  // Fuse.js fuzzy search
  const fuse = useMemo(
    () =>
      new Fuse(papers, {
        keys: [
          { name: "title", weight: 2 },
          { name: "alias", weight: 2 },
          { name: "name", weight: 1 },
        ],
        threshold: 0.4,
        ignoreLocation: true,
      }),
    [papers],
  );

  const filteredPapers = filter.trim()
    ? fuse.search(filter).map((r) => r.item)
    : papers;

  // Alias save
  async function saveAlias() {
    if (!aliasModal) return;
    await updatePaperAlias(aliasModal.id, aliasValue);
    setPapers((prev) =>
      prev.map((p) =>
        p.id === aliasModal.id ? { ...p, alias: aliasValue || null } : p,
      ),
    );
    setAliasModal(null);
  }

  return (
    <div
      className={cn(
        "h-screen grid grid-rows-[52px_1fr]",
        showChat
          ? "grid-cols-[280px_1fr_400px]"
          : "grid-cols-[280px_1fr]",
      )}
    >
      {/* Header */}
      <header className="col-span-full bg-brand text-white flex items-center justify-between px-6">
        <h1 className="text-[15px] font-medium tracking-wide">
          Paper Intelligence
        </h1>
        <button
          onClick={onLogout}
          className="text-white/70 border border-white/20 px-3.5 py-1 rounded-[var(--radius-sm)] text-[13px] hover:text-white hover:border-white/40 cursor-pointer"
        >
          Log out
        </button>
      </header>

      {/* Sidebar */}
      <aside className="bg-surface border-r border-border flex flex-col overflow-hidden">
        <div className="p-4">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-muted" />
            <input
              type="text"
              className="w-full pl-9 pr-3 py-2 bg-canvas border border-border rounded-[var(--radius-sm)] text-[13px] outline-none focus:border-brand focus:bg-surface"
              placeholder="Filter papers..."
              value={filter}
              onChange={(e) => setFilter(e.target.value)}
            />
          </div>
        </div>
        <div className="flex items-center justify-between px-4 pb-2">
          <span className="text-[11px] text-muted uppercase tracking-wider font-medium">
            {filteredPapers.length} paper
            {filteredPapers.length !== 1 ? "s" : ""}
          </span>
          <button
            onClick={() => setShowUpload(true)}
            className="text-muted hover:text-brand cursor-pointer"
            title="Upload papers"
          >
            <Upload className="w-3.5 h-3.5" />
          </button>
        </div>
        <div className="flex-1 overflow-y-auto">
          {filteredPapers.map((p) => (
            <div
              key={p.id}
              onClick={() => selectPaper(p.id)}
              className={cn(
                "px-4 py-3 cursor-pointer border-b border-border text-[13px] leading-snug transition-colors",
                p.id === activePaperId
                  ? "bg-brand-light border-l-[3px] border-l-brand"
                  : "hover:bg-canvas",
              )}
            >
              <div className="line-clamp-2">{displayName(p)}</div>
              <div
                className={cn(
                  "text-[11px] mt-1 flex items-center gap-1.5",
                  p.status === "ready"
                    ? "text-brand"
                    : p.status === "error"
                      ? "text-red-700"
                      : "text-muted",
                )}
              >
                <span
                  className={cn(
                    "w-[5px] h-[5px] rounded-full inline-block",
                    p.status === "ready"
                      ? "bg-brand"
                      : p.status === "error"
                        ? "bg-red-700"
                        : "bg-amber-600 animate-pulse",
                  )}
                />
                {p.status === "ready"
                  ? "Ready"
                  : p.status === "error"
                    ? "Error"
                    : p.status.charAt(0).toUpperCase() + p.status.slice(1)}
                {p.chunk_count ? ` \u00B7 ${p.chunk_count} chunks` : ""}
              </div>
            </div>
          ))}
        </div>
      </aside>

      {/* Main area */}
      <div className="flex flex-col overflow-hidden bg-canvas">
        {/* Toolbar */}
        {activePaper && (
          <div className="flex items-center gap-4 px-6 py-2.5 bg-surface border-b border-border min-h-[48px]">
            <button
              onClick={() => setShowOutline(!showOutline)}
              className={cn(
                "px-3 py-1 border rounded-[var(--radius-sm)] text-[12px] cursor-pointer font-normal",
                showOutline
                  ? "bg-brand-light border-brand text-brand"
                  : "border-border hover:bg-canvas",
              )}
            >
              Outline
            </button>
            <span className="flex-1 min-w-0 flex items-center gap-2">
              <span className="truncate text-sm font-medium tracking-tight">
                {displayName(activePaper)}
              </span>
              <button
                onClick={() => {
                  setAliasValue(activePaper.alias || displayName(activePaper));
                  setAliasModal(activePaper);
                }}
                className="text-muted hover:text-brand cursor-pointer shrink-0"
                title="Rename paper"
              >
                <Pencil className="w-3 h-3" />
              </button>
            </span>
            {activePaper.pdf_key && (
              <div className="flex border border-border rounded-[var(--radius-sm)] overflow-hidden">
                <button
                  onClick={() => setView("md")}
                  className={cn(
                    "px-3 py-1 text-[12px] cursor-pointer border-r border-border",
                    view === "md"
                      ? "bg-brand text-white"
                      : "bg-surface hover:bg-canvas",
                  )}
                >
                  <BookOpen className="w-3.5 h-3.5 inline mr-1" />
                  Markdown
                </button>
                <button
                  onClick={() => setView("pdf")}
                  className={cn(
                    "px-3 py-1 text-[12px] cursor-pointer",
                    view === "pdf"
                      ? "bg-brand text-white"
                      : "bg-surface hover:bg-canvas",
                  )}
                >
                  <FileText className="w-3.5 h-3.5 inline mr-1" />
                  PDF
                </button>
              </div>
            )}
            <button
              onClick={() => setShowChat(!showChat)}
              className={cn(
                "px-3 py-1 border rounded-[var(--radius-sm)] text-[12px] cursor-pointer font-normal",
                showChat
                  ? "bg-brand-light border-brand text-brand"
                  : "border-border hover:bg-canvas",
              )}
            >
              <MessageSquare className="w-3.5 h-3.5 inline mr-1" />
              Chat
            </button>
          </div>
        )}

        {/* Content area */}
        <div className="flex-1 flex overflow-hidden">
          {/* Outline */}
          {showOutline && headers.length > 0 && (
            <div className="w-[240px] shrink-0 border-r border-border bg-surface overflow-y-auto py-4">
              <div className="px-4 pb-2 text-[11px] text-muted uppercase tracking-wider font-medium">
                Contents
              </div>
              {headers.map((h, i) => (
                <div
                  key={i}
                  onClick={() => {
                    const el = document.getElementById(`heading-${i}`);
                    el?.scrollIntoView({ behavior: "smooth", block: "start" });
                  }}
                  className={cn(
                    "py-1 text-[13px] leading-snug cursor-pointer text-muted hover:text-ink truncate",
                    h.level === 1 && "px-4",
                    h.level === 2 && "pl-7 pr-4 text-[12px]",
                    h.level === 3 && "pl-10 pr-4 text-[12px]",
                    h.level >= 4 && "pl-13 pr-4 text-[12px]",
                  )}
                  title={h.text}
                >
                  {h.text}
                </div>
              ))}
            </div>
          )}

          {/* Content viewer */}
          {!activePaper ? (
            <div className="flex-1 flex items-center justify-center text-muted text-[15px]">
              Select a paper to begin
            </div>
          ) : view === "pdf" && pdfUrl ? (
            <iframe
              src={`${pdfUrl}#navpanes=0`}
              className="flex-1 border-none"
            />
          ) : view === "pdf" && !pdfUrl ? (
            <div className="flex-1 flex items-center justify-center text-muted text-sm">
              Loading PDF...
            </div>
          ) : (
            <div className="flex-1 overflow-y-auto bg-surface">
              <article className="max-w-[780px] px-8 py-8 prose prose-sm prose-neutral prose-headings:font-medium prose-headings:tracking-tight prose-p:leading-relaxed prose-pre:bg-canvas prose-pre:border prose-pre:border-border prose-pre:rounded-[var(--radius-sm)] prose-blockquote:border-l-brand prose-a:text-brand prose-a:no-underline hover:prose-a:underline prose-img:rounded-[var(--radius-sm)] prose-table:text-[13px] prose-th:bg-canvas">
                <ReactMarkdown
                  remarkPlugins={[remarkGfm]}
                  rehypePlugins={[rehypeRaw]}
                  components={{
                    h1: ({ children, ...props }) => {
                      const idx = headers.findIndex(
                        (h) => h.level === 1 && h.text === String(children),
                      );
                      return (
                        <h1 id={idx >= 0 ? `heading-${idx}` : undefined} {...props}>
                          {children}
                        </h1>
                      );
                    },
                    h2: ({ children, ...props }) => {
                      const idx = headers.findIndex(
                        (h) => h.level === 2 && h.text === String(children),
                      );
                      return (
                        <h2 id={idx >= 0 ? `heading-${idx}` : undefined} {...props}>
                          {children}
                        </h2>
                      );
                    },
                    h3: ({ children, ...props }) => {
                      const idx = headers.findIndex(
                        (h) => h.level === 3 && h.text === String(children),
                      );
                      return (
                        <h3 id={idx >= 0 ? `heading-${idx}` : undefined} {...props}>
                          {children}
                        </h3>
                      );
                    },
                  }}
                >
                  {markdown || ""}
                </ReactMarkdown>
              </article>
            </div>
          )}
        </div>
      </div>

      {/* Chat panel */}
      {showChat && <ChatPanel paperId={activePaperId} paperName={activePaper ? displayName(activePaper) : ""} />}

      {/* Alias modal */}
      {aliasModal && (
        <div
          className="fixed inset-0 bg-ink/30 flex items-center justify-center z-50"
          onClick={() => setAliasModal(null)}
        >
          <div
            className="bg-surface border border-border rounded-lg p-6 w-[420px] max-w-[90vw]"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="font-medium">Rename paper</div>
            <p className="text-[13px] text-muted mt-1">
              Set a custom display name. Leave empty to use the auto-detected
              title.
            </p>
            <input
              className="w-full px-3 py-2.5 border border-border rounded-[var(--radius-sm)] text-sm mt-4 outline-none focus:border-brand"
              placeholder="Display name"
              value={aliasValue}
              onChange={(e) => setAliasValue(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && saveAlias()}
              autoFocus
            />
            <div className="flex gap-2 justify-end mt-4">
              <button
                onClick={() => setAliasModal(null)}
                className="px-4 py-2 border border-border rounded-[var(--radius-sm)] text-[13px] cursor-pointer hover:bg-canvas"
              >
                Cancel
              </button>
              <button
                onClick={saveAlias}
                className="px-4 py-2 bg-brand text-white rounded-[var(--radius-sm)] text-[13px] font-medium cursor-pointer hover:bg-brand-hover"
              >
                Save
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Upload modal */}
      {showUpload && (
        <UploadModal
          onClose={() => {
            setShowUpload(false);
            loadPapers();
          }}
        />
      )}
    </div>
  );
}

function ChatPanel({
  paperId,
  paperName,
}: {
  paperId: string | null;
  paperName: string;
}) {
  const [messages, setMessages] = useState<
    { role: string; content: string }[]
  >([]);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);
  const prevPaperId = useRef(paperId);

  // Reset chat when paper changes
  useEffect(() => {
    if (paperId !== prevPaperId.current) {
      setMessages([]);
      prevPaperId.current = paperId;
    }
  }, [paperId]);

  useEffect(() => {
    scrollRef.current?.scrollTo(0, scrollRef.current.scrollHeight);
  }, [messages]);

  async function send() {
    if (!input.trim() || streaming) return;
    const msg = input.trim();
    setInput("");
    const newMsgs = [...messages, { role: "user", content: msg }];
    setMessages([...newMsgs, { role: "assistant", content: "..." }]);
    setStreaming(true);
    try {
      const history = newMsgs.slice(0, -1);
      const fullText = await streamChat(
        msg,
        paperId ? [paperId] : [],
        history,
        (text) => {
          setMessages([...newMsgs, { role: "assistant", content: text }]);
        },
      );
      setMessages([...newMsgs, { role: "assistant", content: fullText }]);
    } catch (e) {
      setMessages([
        ...newMsgs,
        {
          role: "assistant",
          content: `Error: ${e instanceof Error ? e.message : String(e)}`,
        },
      ]);
    }
    setStreaming(false);
  }

  return (
    <div className="border-l border-border bg-chat-bg flex flex-col overflow-hidden">
      <div className="flex items-center justify-between px-5 py-3.5 border-b border-border bg-surface text-sm font-medium">
        <span>Chat</span>
        <span className="text-[12px] text-muted truncate max-w-[220px] font-normal">
          {paperName}
        </span>
      </div>
      <div ref={scrollRef} className="flex-1 overflow-y-auto px-5 py-4 flex flex-col gap-2.5">
        {messages.map((m, i) => (
          <div
            key={i}
            className={cn(
              "px-3.5 py-2.5 rounded-lg text-sm leading-relaxed max-w-[95%]",
              m.role === "user"
                ? "bg-brand-light self-end rounded-br-sm"
                : "bg-surface border border-border self-start rounded-bl-sm",
            )}
          >
            {m.role === "assistant" ? (
              <div className="prose prose-sm prose-neutral prose-p:my-1 prose-p:first:mt-0">
                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                  {m.content}
                </ReactMarkdown>
              </div>
            ) : (
              m.content
            )}
          </div>
        ))}
      </div>
      <div className="flex gap-2.5 px-5 py-3.5 border-t border-border bg-surface">
        <textarea
          className="flex-1 px-3 py-2.5 border border-border rounded-[var(--radius-sm)] text-sm resize-none outline-none focus:border-brand min-h-[42px] max-h-[120px]"
          rows={1}
          placeholder="Ask about this paper..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              send();
            }
          }}
          onInput={(e) => {
            const t = e.target as HTMLTextAreaElement;
            t.style.height = "auto";
            t.style.height = Math.min(t.scrollHeight, 120) + "px";
          }}
        />
        <button
          onClick={send}
          disabled={streaming}
          className="px-5 py-2.5 bg-brand text-white rounded-[var(--radius-sm)] text-sm font-medium cursor-pointer hover:bg-brand-hover disabled:opacity-40 disabled:cursor-not-allowed self-end"
        >
          Send
        </button>
      </div>
    </div>
  );
}

function UploadModal({ onClose }: { onClose: () => void }) {
  const [files, setFiles] = useState<File[]>([]);
  const [uploading, setUploading] = useState(false);
  const [status, setStatus] = useState<Record<string, string>>({});
  const fileInputRef = useRef<HTMLInputElement>(null);

  async function doUpload() {
    if (!files.length) return;
    setUploading(true);
    await uploadPapers(files, (name, s) => {
      setStatus((prev) => ({ ...prev, [name]: s }));
    });
    setUploading(false);
  }

  const allDone =
    files.length > 0 &&
    Object.keys(status).length === files.length &&
    Object.values(status).every((s) => s === "done" || s === "error");

  return (
    <div
      className="fixed inset-0 bg-ink/30 flex items-center justify-center z-50"
      onClick={onClose}
    >
      <div
        className="bg-surface border border-border rounded-lg p-6 w-[500px] max-w-[90vw]"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between mb-1">
          <div className="font-medium">Upload papers</div>
          <button onClick={onClose} className="text-muted hover:text-ink cursor-pointer">
            <X className="w-4 h-4" />
          </button>
        </div>
        <p className="text-[13px] text-muted mb-4">
          Select PDF files to upload. They will be converted to markdown and
          indexed automatically.
        </p>

        <input
          ref={fileInputRef}
          type="file"
          accept=".pdf"
          multiple
          className="hidden"
          onChange={(e) => {
            if (e.target.files) {
              setFiles(Array.from(e.target.files));
              setStatus({});
            }
          }}
        />

        <button
          onClick={() => fileInputRef.current?.click()}
          className="w-full py-8 border-2 border-dashed border-border rounded-lg text-sm text-muted hover:border-brand hover:text-brand cursor-pointer transition-colors"
        >
          <Upload className="w-5 h-5 mx-auto mb-2" />
          Click to select PDFs
        </button>

        {files.length > 0 && (
          <div className="mt-4 max-h-[200px] overflow-y-auto">
            {files.map((f) => {
              const name = f.name.replace(/\.pdf$/i, "");
              const s = status[name];
              return (
                <div
                  key={f.name}
                  className="flex items-center justify-between py-1.5 text-[13px]"
                >
                  <span className="truncate">{f.name}</span>
                  <span
                    className={cn(
                      "text-[11px] ml-2 shrink-0",
                      s === "done"
                        ? "text-brand"
                        : s === "error"
                          ? "text-red-700"
                          : "text-muted",
                    )}
                  >
                    {s === "done"
                      ? "Uploaded"
                      : s === "uploading"
                        ? "Uploading..."
                        : s === "error"
                          ? "Failed"
                          : ""}
                  </span>
                </div>
              );
            })}
          </div>
        )}

        <div className="flex gap-2 justify-end mt-4">
          <button
            onClick={onClose}
            className="px-4 py-2 border border-border rounded-[var(--radius-sm)] text-[13px] cursor-pointer hover:bg-canvas"
          >
            {allDone ? "Done" : "Cancel"}
          </button>
          {!allDone && (
            <button
              onClick={doUpload}
              disabled={!files.length || uploading}
              className="px-4 py-2 bg-brand text-white rounded-[var(--radius-sm)] text-[13px] font-medium cursor-pointer hover:bg-brand-hover disabled:opacity-40"
            >
              {uploading ? "Uploading..." : `Upload ${files.length} file${files.length !== 1 ? "s" : ""}`}
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
