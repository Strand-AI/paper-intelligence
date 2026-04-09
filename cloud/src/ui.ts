export const HTML = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Paper Intelligence</title>
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"><\/script>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
:root {
  --bg: #f8f9fa; --sidebar: #fff; --content: #fff; --header: #1a1a2e;
  --accent: #3b82f6; --text: #1f2937; --muted: #6b7280; --border: #e5e7eb;
  --chat-bg: #f9fafb; --user-msg: #dbeafe; --ai-msg: #fff;
  --font: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
}
body { font-family: var(--font); background: var(--bg); color: var(--text); height: 100vh; overflow: hidden; }

/* Auth */
#auth-modal { position: fixed; inset: 0; background: rgba(0,0,0,0.5); display: flex; align-items: center; justify-content: center; z-index: 100; }
#auth-modal.hidden { display: none; }
.auth-box { background: #fff; padding: 2rem; border-radius: 12px; width: 400px; max-width: 90vw; }
.auth-box h2 { margin-bottom: 1rem; }
.auth-box input { width: 100%; padding: 0.75rem; border: 1px solid var(--border); border-radius: 8px; font-size: 14px; margin-bottom: 1rem; }
.auth-box button { width: 100%; padding: 0.75rem; background: var(--accent); color: #fff; border: none; border-radius: 8px; font-size: 14px; cursor: pointer; }

/* Layout: sidebar | main content area | chat (optional) */
#app { display: none; height: 100vh; grid-template-columns: 260px 1fr; grid-template-rows: 48px 1fr; }
#app.active { display: grid; }
#app.chat-open { grid-template-columns: 260px 1fr 380px; }
header { grid-column: 1 / -1; background: var(--header); color: #fff; display: flex; align-items: center; justify-content: space-between; padding: 0 1rem; }
header h1 { font-size: 15px; font-weight: 600; }
.header-right { display: flex; align-items: center; gap: 0.75rem; }
.header-right select, .header-right button { background: rgba(255,255,255,0.1); color: #fff; border: 1px solid rgba(255,255,255,0.2); padding: 4px 8px; border-radius: 6px; font-size: 13px; cursor: pointer; }

/* Sidebar */
#sidebar { background: var(--sidebar); border-right: 1px solid var(--border); display: flex; flex-direction: column; overflow: hidden; }
.sidebar-search { padding: 0.75rem; border-bottom: 1px solid var(--border); }
.sidebar-search input { width: 100%; padding: 8px 12px; border: 1px solid var(--border); border-radius: 6px; font-size: 13px; }
.paper-count { padding: 6px 16px; font-size: 11px; color: var(--muted); border-bottom: 1px solid var(--border); text-transform: uppercase; letter-spacing: 0.05em; }
.paper-list { flex: 1; overflow-y: auto; }
.paper-item { padding: 10px 14px; cursor: pointer; border-bottom: 1px solid var(--border); font-size: 13px; line-height: 1.4; }
.paper-item:hover { background: var(--bg); }
.paper-item.active { background: #eff6ff; border-left: 3px solid var(--accent); }
.paper-item .paper-title { overflow: hidden; text-overflow: ellipsis; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; }
.paper-item .status { font-size: 11px; color: var(--muted); margin-top: 3px; display: flex; align-items: center; gap: 4px; }
.paper-item .status.ready { color: #10b981; }
.paper-item .status.error { color: #ef4444; }
.status-dot { width: 6px; height: 6px; border-radius: 50%; display: inline-block; }
.status-dot.ready { background: #10b981; }
.status-dot.error { background: #ef4444; }
.status-dot.processing { background: #f59e0b; animation: pulse 1s infinite; }
@keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.4; } }

/* Main content area */
#main { display: flex; flex-direction: column; overflow: hidden; }

/* Toolbar */
#toolbar { display: none; padding: 8px 16px; border-bottom: 1px solid var(--border); background: #fff; align-items: center; gap: 12px; font-size: 13px; min-height: 44px; }
#toolbar.active { display: flex; }
#toolbar .paper-display-name { font-weight: 600; flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
#toolbar .paper-display-name .edit-alias { display: none; cursor: pointer; color: var(--muted); margin-left: 4px; font-size: 11px; }
#toolbar .paper-display-name:hover .edit-alias { display: inline; }
.view-toggle { display: flex; border: 1px solid var(--border); border-radius: 6px; overflow: hidden; }
.view-toggle button { padding: 4px 12px; border: none; background: #fff; font-size: 12px; cursor: pointer; color: var(--text); }
.view-toggle button.active { background: var(--accent); color: #fff; }
.view-toggle button:not(:last-child) { border-right: 1px solid var(--border); }
#outline-toggle, #chat-toggle { padding: 4px 10px; border: 1px solid var(--border); background: #fff; border-radius: 6px; font-size: 12px; cursor: pointer; color: var(--text); }
#outline-toggle.active, #chat-toggle.active { background: #eff6ff; border-color: var(--accent); color: var(--accent); }

/* Main body: outline + content */
#main-body { flex: 1; display: flex; overflow: hidden; }

/* Outline / TOC */
#outline { width: 0; overflow-y: auto; overflow-x: hidden; border-right: 1px solid var(--border); background: var(--sidebar); transition: width 0.15s; flex-shrink: 0; }
#outline.open { width: 220px; padding: 12px 0; }
#outline .toc-item { padding: 5px 14px; font-size: 12px; line-height: 1.4; cursor: pointer; color: var(--muted); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
#outline .toc-item:hover { color: var(--text); background: var(--bg); }
#outline .toc-item.active { color: var(--accent); font-weight: 500; }
#outline .toc-item.l2 { padding-left: 26px; }
#outline .toc-item.l3 { padding-left: 38px; }
#outline .toc-item.l4 { padding-left: 50px; }

/* Content viewer */
#content { flex: 1; overflow-y: auto; padding: 2rem; }
#content.empty { display: flex; align-items: center; justify-content: center; color: var(--muted); font-size: 15px; }
#content h1 { font-size: 1.5rem; margin: 1.5rem 0 0.75rem; }
#content h2 { font-size: 1.3rem; margin: 1.25rem 0 0.5rem; }
#content h3 { font-size: 1.1rem; margin: 1rem 0 0.5rem; }
#content p { margin: 0.5rem 0; line-height: 1.7; }
#content pre { background: #f1f5f9; padding: 1rem; border-radius: 8px; overflow-x: auto; font-size: 13px; margin: 0.75rem 0; }
#content code { font-size: 13px; background: #f1f5f9; padding: 2px 6px; border-radius: 4px; }
#content pre code { background: none; padding: 0; }
#content blockquote { border-left: 3px solid var(--accent); padding-left: 1rem; color: var(--muted); margin: 0.75rem 0; }
#content table { border-collapse: collapse; margin: 0.75rem 0; font-size: 14px; }
#content th, #content td { border: 1px solid var(--border); padding: 8px 12px; text-align: left; }
#content th { background: var(--bg); font-weight: 600; }
#content img { max-width: 100%; border-radius: 8px; margin: 0.5rem 0; }
#content .md-view { max-width: 860px; }
#pdf-frame { width: 100%; height: 100%; border: none; }

/* Chat panel (right side, hidden by default) */
#chat { display: none; border-left: 1px solid var(--border); background: var(--chat-bg); flex-direction: column; overflow: hidden; }
#app.chat-open #chat { display: flex; }
#chat-header { display: flex; align-items: center; justify-content: space-between; padding: 10px 14px; font-size: 13px; font-weight: 600; user-select: none; border-bottom: 1px solid var(--border); background: #fff; }
#chat-header .chat-title { display: flex; align-items: center; gap: 6px; }
#chat-header .chat-context { font-weight: 400; color: var(--muted); font-size: 11px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: 200px; }
#chat-messages { flex: 1; overflow-y: auto; padding: 12px 14px; display: flex; flex-direction: column; gap: 8px; }
.msg { padding: 8px 12px; border-radius: 10px; font-size: 14px; line-height: 1.6; max-width: 95%; }
.msg.user { background: var(--user-msg); align-self: flex-end; border-bottom-right-radius: 4px; }
.msg.assistant { background: var(--ai-msg); align-self: flex-start; border: 1px solid var(--border); border-bottom-left-radius: 4px; }
.msg.assistant p { margin: 0.25rem 0; }
.msg.assistant p:first-child { margin-top: 0; }
.msg.assistant code { font-size: 12px; }
#chat-input-row { display: flex; gap: 8px; padding: 10px 14px; border-top: 1px solid var(--border); background: #fff; }
#chat-input { flex: 1; padding: 8px 12px; border: 1px solid var(--border); border-radius: 8px; font-size: 14px; font-family: var(--font); resize: none; min-height: 40px; max-height: 120px; }
#chat-send { padding: 8px 16px; background: var(--accent); color: #fff; border: none; border-radius: 8px; cursor: pointer; font-size: 14px; align-self: flex-end; }
#chat-send:disabled { opacity: 0.5; cursor: not-allowed; }

/* Alias edit modal */
#alias-modal { display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.3); z-index: 50; align-items: center; justify-content: center; }
#alias-modal.open { display: flex; }
#alias-modal .modal-box { background: #fff; padding: 1.5rem; border-radius: 12px; width: 400px; max-width: 90vw; }
#alias-modal input { width: 100%; padding: 0.5rem; border: 1px solid var(--border); border-radius: 6px; font-size: 14px; margin: 0.75rem 0; }
#alias-modal .modal-actions { display: flex; gap: 8px; justify-content: flex-end; }
#alias-modal button { padding: 6px 16px; border-radius: 6px; font-size: 13px; cursor: pointer; }
#alias-modal .btn-save { background: var(--accent); color: #fff; border: none; }
#alias-modal .btn-cancel { background: #fff; border: 1px solid var(--border); }
</style>
</head>
<body>

<div id="auth-modal">
  <div class="auth-box">
    <h2>Paper Intelligence</h2>
    <p style="color:#6b7280;margin-bottom:1rem;font-size:14px">Enter your API token to continue.</p>
    <input type="password" id="auth-token" placeholder="Bearer token" autofocus
      onkeydown="if(event.key==='Enter')login()">
    <button onclick="login()">Continue</button>
  </div>
</div>

<div id="app">
  <header>
    <h1>Paper Intelligence</h1>
    <div class="header-right">
      <select id="model-select">
        <option value="claude-opus-4-6">Claude Opus 4.6</option>
        <option value="claude-sonnet-4-6">Claude Sonnet 4.6</option>
        <option value="gpt-4.1">GPT-4.1</option>
        <option value="o3">o3</option>
      </select>
      <button onclick="logout()">Logout</button>
    </div>
  </header>

  <div id="sidebar">
    <div class="sidebar-search"><input type="text" id="paper-search" placeholder="Filter papers..."></div>
    <div class="paper-count" id="paper-count"></div>
    <div class="paper-list" id="paper-list"></div>
  </div>

  <div id="main">
    <div id="toolbar">
      <button id="outline-toggle" onclick="toggleOutline()">Outline</button>
      <span class="paper-display-name" id="paper-display-name">
        <span id="display-name-text"></span>
        <span class="edit-alias" onclick="openAliasModal()">&boxbox; rename</span>
      </span>
      <div class="view-toggle" id="view-toggle" style="display:none">
        <button class="active" onclick="setView('md')">Markdown</button>
        <button onclick="setView('pdf')">PDF</button>
      </div>
      <button id="chat-toggle" onclick="toggleChat()">Chat</button>
    </div>
    <div id="main-body">
      <div id="outline"></div>
      <div id="content" class="empty">Select a paper from the sidebar</div>
    </div>
  </div>

  <div id="chat">
    <div id="chat-header">
      <div class="chat-title">Chat</div>
      <div class="chat-context" id="chat-context-label"></div>
    </div>
    <div id="chat-messages"></div>
    <div id="chat-input-row">
      <textarea id="chat-input" rows="1" placeholder="Ask about this paper..."
        onkeydown="if(event.key==='Enter'&&!event.shiftKey){event.preventDefault();sendChat()}"
        oninput="this.style.height='auto';this.style.height=Math.min(this.scrollHeight,120)+'px'"></textarea>
      <button id="chat-send" onclick="sendChat()">Send</button>
    </div>
  </div>
</div>

<div id="alias-modal">
  <div class="modal-box">
    <div style="font-weight:600;font-size:15px">Rename Paper</div>
    <input type="text" id="alias-input" placeholder="Display name (leave empty to use auto-detected title)">
    <div class="modal-actions">
      <button class="btn-cancel" onclick="closeAliasModal()">Cancel</button>
      <button class="btn-save" onclick="saveAlias()">Save</button>
    </div>
  </div>
</div>

<script>
let TOKEN = localStorage.getItem('pi_token') || '';
let papers = [];
let activePaperId = null;
let activeView = 'md';
let cachedContent = {};
let chatHistory = [];
let pollTimer = null;

const API = '';

function headers() { return { 'Authorization': 'Bearer ' + TOKEN, 'Content-Type': 'application/json' }; }

function displayName(p) {
  return p.alias || p.title || p.name.replace(/_/g, ' ').replace(/-/g, ' ');
}

function statusInfo(s) {
  const map = { uploading: 'Uploading', converting: 'Converting PDF', indexing: 'Indexing', embedding: 'Embedding', ready: 'Ready', error: 'Error' };
  const processing = ['uploading','converting','indexing','embedding'].includes(s);
  return { label: map[s] || s, processing, cls: s === 'ready' ? 'ready' : s === 'error' ? 'error' : 'processing' };
}

// Auth
if (TOKEN) { document.getElementById('auth-modal').classList.add('hidden'); document.getElementById('app').classList.add('active'); init(); }

function login() {
  TOKEN = document.getElementById('auth-token').value.trim();
  if (!TOKEN) return;
  localStorage.setItem('pi_token', TOKEN);
  document.getElementById('auth-modal').classList.add('hidden');
  document.getElementById('app').classList.add('active');
  init();
}

function logout() { TOKEN = ''; localStorage.removeItem('pi_token'); location.reload(); }

async function init() {
  const res = await fetch(API + '/papers', { headers: headers() });
  if (res.status === 401) { logout(); return; }
  const data = await res.json();
  papers = data.papers || [];
  renderPaperList();
  startPollingIfNeeded();
}

function renderPaperList(filter) {
  const f = (filter || document.getElementById('paper-search').value).toLowerCase();
  const filtered = papers.filter(p => {
    const dn = displayName(p).toLowerCase();
    const nm = (p.name || '').toLowerCase();
    return !f || dn.includes(f) || nm.includes(f);
  });
  document.getElementById('paper-count').textContent = filtered.length + ' paper' + (filtered.length !== 1 ? 's' : '');
  const list = document.getElementById('paper-list');
  list.innerHTML = filtered.map(p => {
    const si = statusInfo(p.status);
    return '<div class="paper-item' + (p.id === activePaperId ? ' active' : '') + '" onclick="selectPaper(\\'' + p.id + '\\')">' +
      '<div class="paper-title">' + escapeHtml(displayName(p)) + '</div>' +
      '<div class="status ' + si.cls + '"><span class="status-dot ' + si.cls + '"></span>' + si.label +
      (p.chunk_count ? ' &middot; ' + p.chunk_count + ' chunks' : '') + '</div></div>';
  }).join('');
}

function startPollingIfNeeded() {
  if (pollTimer) clearInterval(pollTimer);
  const hasProcessing = papers.some(p => ['uploading','converting','indexing','embedding'].includes(p.status));
  if (!hasProcessing) return;
  pollTimer = setInterval(async () => {
    const res = await fetch(API + '/papers', { headers: headers() });
    if (!res.ok) return;
    const data = await res.json();
    papers = data.papers || [];
    renderPaperList();
    if (!papers.some(p => ['uploading','converting','indexing','embedding'].includes(p.status))) {
      clearInterval(pollTimer); pollTimer = null;
    }
  }, 3000);
}

document.getElementById('paper-search').addEventListener('input', e => renderPaperList(e.target.value));

async function selectPaper(id) {
  activePaperId = id;
  activeView = 'md';
  renderPaperList();
  const toolbar = document.getElementById('toolbar');
  toolbar.classList.add('active');
  const content = document.getElementById('content');
  content.className = '';
  content.innerHTML = '<p style="color:#6b7280;padding:2rem">Loading...</p>';

  const paper = papers.find(p => p.id === id);
  document.getElementById('display-name-text').textContent = displayName(paper);
  document.getElementById('chat-context-label').textContent = displayName(paper);

  // Show PDF toggle if PDF exists
  const vt = document.getElementById('view-toggle');
  vt.style.display = paper && paper.pdf_key ? 'flex' : 'none';
  updateViewToggle();

  // Fetch content + headers
  const res = await fetch(API + '/papers/' + id + '/content', { headers: headers() });
  if (!res.ok) { content.innerHTML = '<p style="padding:2rem">Error loading paper</p>'; return; }
  const data = await res.json();
  cachedContent[id] = data;

  renderContent(data);
  renderOutline(data.headers || []);

  chatHistory = [];
  document.getElementById('chat-messages').innerHTML = '';
}

function renderContent(data) {
  const content = document.getElementById('content');
  if (activeView === 'pdf') {
    const paper = papers.find(p => p.id === activePaperId);
    if (paper && paper.pdf_key) {
      content.innerHTML = '<p style="padding:2rem;color:var(--muted)">Loading PDF...</p>';
      content.style.padding = '0';
      // Fetch PDF with auth header, then display via blob URL
      fetch(API + '/papers/' + activePaperId + '/pdf', {
        headers: { 'Authorization': 'Bearer ' + TOKEN }
      }).then(r => r.blob()).then(blob => {
        const url = URL.createObjectURL(blob);
        content.innerHTML = '<iframe id="pdf-frame" src="' + url + '"></iframe>';
      }).catch(() => {
        content.innerHTML = '<p style="padding:2rem;color:var(--muted)">Error loading PDF</p>';
      });
    } else {
      content.innerHTML = '<p style="padding:2rem;color:var(--muted)">No PDF available</p>';
    }
  } else {
    content.style.padding = '2rem';
    content.innerHTML = '<div class="md-view">' + marked.parse(data.markdown || '') + '</div>';
    // Add ids to headings for outline scroll
    content.querySelectorAll('h1,h2,h3,h4,h5,h6').forEach((el, i) => { el.id = 'heading-' + i; });
  }
}

function renderOutline(hdrs) {
  const outline = document.getElementById('outline');
  if (!hdrs.length) { outline.innerHTML = ''; return; }
  outline.innerHTML = hdrs.map((h, i) => {
    const lvl = Math.min(h.level, 4);
    return '<div class="toc-item l' + lvl + '" onclick="scrollToHeading(' + i + ')" title="' + escapeHtml(h.text) + '">' + escapeHtml(h.text) + '</div>';
  }).join('');
}

function scrollToHeading(i) {
  const el = document.getElementById('heading-' + i);
  if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function toggleOutline() {
  const outline = document.getElementById('outline');
  outline.classList.toggle('open');
  document.getElementById('outline-toggle').classList.toggle('active');
}

function setView(v) {
  activeView = v;
  updateViewToggle();
  const data = cachedContent[activePaperId];
  if (data) renderContent(data);
}

function updateViewToggle() {
  const btns = document.querySelectorAll('.view-toggle button');
  btns.forEach(b => b.classList.toggle('active', (activeView === 'md' && b.textContent === 'Markdown') || (activeView === 'pdf' && b.textContent === 'PDF')));
}

function toggleChat() {
  document.getElementById('app').classList.toggle('chat-open');
  document.getElementById('chat-toggle').classList.toggle('active');
}

// Alias
function openAliasModal() {
  const paper = papers.find(p => p.id === activePaperId);
  if (!paper) return;
  document.getElementById('alias-input').value = paper.alias || '';
  document.getElementById('alias-modal').classList.add('open');
  document.getElementById('alias-input').focus();
}
function closeAliasModal() { document.getElementById('alias-modal').classList.remove('open'); }
async function saveAlias() {
  const alias = document.getElementById('alias-input').value.trim();
  await fetch(API + '/papers/' + activePaperId, {
    method: 'PATCH', headers: headers(),
    body: JSON.stringify({ alias: alias }),
  });
  const paper = papers.find(p => p.id === activePaperId);
  if (paper) paper.alias = alias || null;
  renderPaperList();
  document.getElementById('display-name-text').textContent = displayName(paper);
  document.getElementById('chat-context-label').textContent = displayName(paper);
  closeAliasModal();
}

// Chat
async function sendChat() {
  const input = document.getElementById('chat-input');
  const msg = input.value.trim();
  if (!msg) return;
  input.value = '';
  input.style.height = 'auto';

  const messages = document.getElementById('chat-messages');
  messages.innerHTML += '<div class="msg user">' + escapeHtml(msg) + '</div>';
  const aiMsg = document.createElement('div');
  aiMsg.className = 'msg assistant';
  aiMsg.textContent = '...';
  messages.appendChild(aiMsg);
  messages.scrollTop = messages.scrollHeight;

  const btn = document.getElementById('chat-send');
  btn.disabled = true;
  chatHistory.push({ role: 'user', content: msg });

  try {
    const res = await fetch(API + '/chat', {
      method: 'POST', headers: headers(),
      body: JSON.stringify({
        message: msg,
        paper_ids: activePaperId ? [activePaperId] : [],
        model: document.getElementById('model-select').value,
        history: chatHistory.slice(0, -1),
      }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      aiMsg.textContent = 'Error: ' + (err.error || res.status);
      btn.disabled = false;
      return;
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let fullText = '';
    aiMsg.textContent = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const parts = buffer.split('\\n\\n');
      buffer = parts.pop() || '';
      for (const part of parts) {
        for (const line of part.split('\\n')) {
          if (!line.startsWith('data: ')) continue;
          const data = line.slice(6).trim();
          if (data === '[DONE]') continue;
          try {
            const parsed = JSON.parse(data);
            if (parsed.text) {
              fullText += parsed.text;
              aiMsg.innerHTML = marked.parse(fullText);
              messages.scrollTop = messages.scrollHeight;
            }
          } catch {}
        }
      }
    }
    chatHistory.push({ role: 'assistant', content: fullText });
  } catch (e) {
    aiMsg.textContent = 'Error: ' + e.message;
  }
  btn.disabled = false;
}

function escapeHtml(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}
<\/script>
</body>
</html>`;
