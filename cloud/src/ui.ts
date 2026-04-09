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
  --bg: #f2f1ed;
  --surface: #fff;
  --accent: #004d3b;
  --accent-light: #e8f0ed;
  --accent-hover: #003d2f;
  --text: #00120a;
  --muted: #6b7280;
  --border: #e5e7eb;
  --chat-bg: #f7f6f3;
  --user-msg: #e8f0ed;
  --ai-msg: #fff;
  --font: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, sans-serif;
}
body { font-family: var(--font); background: var(--bg); color: var(--text); height: 100vh; overflow: hidden; font-size: 14px; line-height: 1.6; -webkit-font-smoothing: antialiased; }

/* Auth */
#auth-modal { position: fixed; inset: 0; background: rgba(0,18,10,0.4); display: flex; align-items: center; justify-content: center; z-index: 100; }
#auth-modal.hidden { display: none; }
.auth-box { background: var(--surface); padding: 2.5rem; border-radius: 8px; width: 400px; max-width: 90vw; border: 1px solid var(--border); }
.auth-box h2 { font-size: 20px; font-weight: 500; letter-spacing: -0.02em; margin-bottom: 0.5rem; color: var(--text); }
.auth-box p { color: var(--muted); font-size: 14px; margin-bottom: 1.5rem; }
.auth-box input { width: 100%; padding: 10px 12px; border: 1px solid var(--border); border-radius: 4px; font-size: 14px; margin-bottom: 1.25rem; font-family: var(--font); outline: none; }
.auth-box input:focus { border-color: var(--accent); }
.auth-box button { width: 100%; padding: 10px; background: var(--accent); color: #fff; border: none; border-radius: 4px; font-size: 14px; font-weight: 500; cursor: pointer; font-family: var(--font); }
.auth-box button:hover { background: var(--accent-hover); }

/* Layout */
#app { display: none; height: 100vh; grid-template-columns: 280px 1fr; grid-template-rows: 52px 1fr; }
#app.active { display: grid; }
#app.chat-open { grid-template-columns: 280px 1fr 400px; }

/* Header */
header { grid-column: 1 / -1; background: var(--accent); color: #fff; display: flex; align-items: center; justify-content: space-between; padding: 0 24px; }
header h1 { font-size: 15px; font-weight: 500; letter-spacing: 0.01em; }
.header-right { display: flex; align-items: center; gap: 12px; }
.header-right button { background: transparent; color: rgba(255,255,255,0.7); border: 1px solid rgba(255,255,255,0.2); padding: 5px 14px; border-radius: 4px; font-size: 13px; cursor: pointer; font-family: var(--font); }
.header-right button:hover { color: #fff; border-color: rgba(255,255,255,0.4); }

/* Sidebar */
#sidebar { background: var(--surface); border-right: 1px solid var(--border); display: flex; flex-direction: column; overflow: hidden; }
.sidebar-search { padding: 16px; }
.sidebar-search input { width: 100%; padding: 8px 12px; border: 1px solid var(--border); border-radius: 4px; font-size: 13px; font-family: var(--font); outline: none; background: var(--bg); }
.sidebar-search input:focus { border-color: var(--accent); background: var(--surface); }
.paper-count { padding: 0 16px 10px; font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.06em; font-weight: 500; }
.paper-list { flex: 1; overflow-y: auto; }
.paper-item { padding: 12px 16px; cursor: pointer; border-bottom: 1px solid var(--border); font-size: 13px; line-height: 1.5; transition: background 0.1s; }
.paper-item:hover { background: var(--bg); }
.paper-item.active { background: var(--accent-light); border-left: 3px solid var(--accent); }
.paper-item .paper-title { overflow: hidden; text-overflow: ellipsis; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; font-weight: 400; }
.paper-item .status { font-size: 11px; color: var(--muted); margin-top: 4px; display: flex; align-items: center; gap: 5px; }
.paper-item .status.ready { color: var(--accent); }
.paper-item .status.error { color: #b91c1c; }
.status-dot { width: 5px; height: 5px; border-radius: 50%; display: inline-block; }
.status-dot.ready { background: var(--accent); }
.status-dot.error { background: #b91c1c; }
.status-dot.processing { background: #b45309; animation: pulse 1.2s ease-in-out infinite; }
@keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.3; } }

/* Main */
#main { display: flex; flex-direction: column; overflow: hidden; background: var(--bg); }

/* Toolbar */
#toolbar { display: none; padding: 10px 24px; border-bottom: 1px solid var(--border); background: var(--surface); align-items: center; gap: 16px; font-size: 13px; min-height: 48px; }
#toolbar.active { display: flex; }
#toolbar .paper-display-name { font-weight: 500; flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: 14px; letter-spacing: -0.01em; }
#toolbar .paper-display-name .edit-alias { display: none; cursor: pointer; color: var(--muted); margin-left: 6px; font-size: 11px; font-weight: 400; }
#toolbar .paper-display-name:hover .edit-alias { display: inline; }
.toolbar-btn { padding: 5px 12px; border: 1px solid var(--border); background: var(--surface); border-radius: 4px; font-size: 12px; cursor: pointer; color: var(--text); font-family: var(--font); font-weight: 400; }
.toolbar-btn:hover { background: var(--bg); }
.toolbar-btn.active { background: var(--accent-light); border-color: var(--accent); color: var(--accent); }
.view-toggle { display: flex; border: 1px solid var(--border); border-radius: 4px; overflow: hidden; }
.view-toggle button { padding: 5px 14px; border: none; background: var(--surface); font-size: 12px; cursor: pointer; color: var(--text); font-family: var(--font); }
.view-toggle button.active { background: var(--accent); color: #fff; }
.view-toggle button:not(:last-child) { border-right: 1px solid var(--border); }

/* Main body */
#main-body { flex: 1; display: flex; overflow: hidden; }

/* Outline */
#outline { width: 0; overflow-y: auto; overflow-x: hidden; border-right: 1px solid var(--border); background: var(--surface); transition: width 0.15s; flex-shrink: 0; }
#outline.open { width: 240px; padding: 16px 0; }
.toc-label { padding: 0 16px 8px; font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); font-weight: 500; }
#outline .toc-item { padding: 5px 16px; font-size: 13px; line-height: 1.5; cursor: pointer; color: var(--muted); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; transition: color 0.1s; }
#outline .toc-item:hover { color: var(--text); }
#outline .toc-item.l2 { padding-left: 28px; font-size: 12px; }
#outline .toc-item.l3 { padding-left: 40px; font-size: 12px; }
#outline .toc-item.l4 { padding-left: 52px; font-size: 12px; }

/* Content */
#content { flex: 1; overflow-y: auto; padding: 32px; background: var(--surface); }
#content.empty { display: flex; align-items: center; justify-content: center; color: var(--muted); font-size: 15px; background: var(--bg); }
#content .md-view { max-width: 780px; }
#content h1 { font-size: 1.75rem; font-weight: 500; letter-spacing: -0.02em; margin: 2rem 0 1rem; color: var(--text); }
#content h2 { font-size: 1.35rem; font-weight: 500; letter-spacing: -0.01em; margin: 1.75rem 0 0.75rem; }
#content h3 { font-size: 1.1rem; font-weight: 500; margin: 1.5rem 0 0.5rem; }
#content h4 { font-size: 1rem; font-weight: 500; margin: 1.25rem 0 0.5rem; }
#content p { margin: 0.625rem 0; line-height: 1.75; }
#content pre { background: var(--bg); padding: 16px; border-radius: 4px; overflow-x: auto; font-size: 13px; margin: 1rem 0; border: 1px solid var(--border); }
#content code { font-size: 13px; background: var(--bg); padding: 2px 5px; border-radius: 3px; }
#content pre code { background: none; padding: 0; }
#content blockquote { border-left: 3px solid var(--accent); padding-left: 16px; color: var(--muted); margin: 1rem 0; }
#content table { border-collapse: collapse; margin: 1rem 0; font-size: 13px; }
#content th, #content td { border: 1px solid var(--border); padding: 10px 14px; text-align: left; }
#content th { background: var(--bg); font-weight: 500; }
#content img { max-width: 100%; border-radius: 4px; margin: 0.75rem 0; }
#pdf-frame { width: 100%; height: 100%; border: none; }

/* Chat */
#chat { display: none; border-left: 1px solid var(--border); background: var(--chat-bg); flex-direction: column; overflow: hidden; }
#app.chat-open #chat { display: flex; }
#chat-header { display: flex; align-items: center; justify-content: space-between; padding: 14px 20px; font-size: 14px; font-weight: 500; border-bottom: 1px solid var(--border); background: var(--surface); }
#chat-header .chat-context { font-weight: 400; color: var(--muted); font-size: 12px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: 220px; }
#chat-messages { flex: 1; overflow-y: auto; padding: 16px 20px; display: flex; flex-direction: column; gap: 10px; }
.msg { padding: 10px 14px; border-radius: 8px; font-size: 14px; line-height: 1.7; max-width: 95%; }
.msg.user { background: var(--user-msg); align-self: flex-end; border-bottom-right-radius: 2px; }
.msg.assistant { background: var(--ai-msg); align-self: flex-start; border: 1px solid var(--border); border-bottom-left-radius: 2px; }
.msg.assistant p { margin: 0.25rem 0; }
.msg.assistant p:first-child { margin-top: 0; }
.msg.assistant code { font-size: 12px; }
#chat-input-row { display: flex; gap: 10px; padding: 14px 20px; border-top: 1px solid var(--border); background: var(--surface); }
#chat-input { flex: 1; padding: 10px 12px; border: 1px solid var(--border); border-radius: 4px; font-size: 14px; font-family: var(--font); resize: none; min-height: 42px; max-height: 120px; outline: none; }
#chat-input:focus { border-color: var(--accent); }
#chat-send { padding: 10px 20px; background: var(--accent); color: #fff; border: none; border-radius: 4px; cursor: pointer; font-size: 14px; font-weight: 500; font-family: var(--font); align-self: flex-end; }
#chat-send:hover { background: var(--accent-hover); }
#chat-send:disabled { opacity: 0.4; cursor: not-allowed; }

/* Modal */
#alias-modal { display: none; position: fixed; inset: 0; background: rgba(0,18,10,0.3); z-index: 50; align-items: center; justify-content: center; }
#alias-modal.open { display: flex; }
#alias-modal .modal-box { background: var(--surface); padding: 2rem; border-radius: 8px; width: 420px; max-width: 90vw; border: 1px solid var(--border); }
#alias-modal .modal-title { font-size: 16px; font-weight: 500; margin-bottom: 4px; }
#alias-modal .modal-desc { font-size: 13px; color: var(--muted); }
#alias-modal input { width: 100%; padding: 10px 12px; border: 1px solid var(--border); border-radius: 4px; font-size: 14px; margin: 16px 0; font-family: var(--font); outline: none; }
#alias-modal input:focus { border-color: var(--accent); }
#alias-modal .modal-actions { display: flex; gap: 10px; justify-content: flex-end; }
#alias-modal button { padding: 8px 18px; border-radius: 4px; font-size: 13px; cursor: pointer; font-family: var(--font); font-weight: 500; }
#alias-modal .btn-save { background: var(--accent); color: #fff; border: none; }
#alias-modal .btn-save:hover { background: var(--accent-hover); }
#alias-modal .btn-cancel { background: var(--surface); border: 1px solid var(--border); color: var(--text); }
</style>
</head>
<body>

<div id="auth-modal">
  <div class="auth-box">
    <h2>Paper Intelligence</h2>
    <p>Enter your API token to continue.</p>
    <input type="password" id="auth-token" placeholder="Token" autofocus
      onkeydown="if(event.key==='Enter')login()">
    <button onclick="login()">Continue</button>
  </div>
</div>

<div id="app">
  <header>
    <h1>Paper Intelligence</h1>
    <div class="header-right">
      <button onclick="logout()">Log out</button>
    </div>
  </header>

  <div id="sidebar">
    <div class="sidebar-search"><input type="text" id="paper-search" placeholder="Filter papers..."></div>
    <div class="paper-count" id="paper-count"></div>
    <div class="paper-list" id="paper-list"></div>
  </div>

  <div id="main">
    <div id="toolbar">
      <button class="toolbar-btn" id="outline-toggle" onclick="toggleOutline()">Outline</button>
      <span class="paper-display-name" id="paper-display-name">
        <span id="display-name-text"></span>
        <span class="edit-alias" onclick="openAliasModal()">rename</span>
      </span>
      <div class="view-toggle" id="view-toggle" style="display:none">
        <button class="active" onclick="setView('md')">Markdown</button>
        <button onclick="setView('pdf')">PDF</button>
      </div>
      <button class="toolbar-btn" id="chat-toggle" onclick="toggleChat()">Chat</button>
    </div>
    <div id="main-body">
      <div id="outline"></div>
      <div id="content" class="empty">Select a paper to begin</div>
    </div>
  </div>

  <div id="chat">
    <div id="chat-header">
      <span>Chat</span>
      <span class="chat-context" id="chat-context-label"></span>
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
    <div class="modal-title">Rename paper</div>
    <div class="modal-desc">Set a custom display name. Leave empty to use the auto-detected title.</div>
    <input type="text" id="alias-input" placeholder="Display name"
      onkeydown="if(event.key==='Enter')saveAlias()">
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
const CHAT_MODEL = 'claude-opus-4-6';

function headers() { return { 'Authorization': 'Bearer ' + TOKEN, 'Content-Type': 'application/json' }; }

function displayName(p) {
  return p.alias || p.title || p.name.replace(/_/g, ' ').replace(/-/g, ' ');
}

function statusInfo(s) {
  const map = { uploading: 'Uploading', converting: 'Converting', indexing: 'Indexing', embedding: 'Embedding', ready: 'Ready', error: 'Error' };
  return { label: map[s] || s, cls: s === 'ready' ? 'ready' : s === 'error' ? 'error' : 'processing' };
}

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

function fuzzyMatch(text, query) {
  let qi = 0;
  let score = 0;
  let lastMatch = -1;
  const tl = text.toLowerCase();
  const ql = query.toLowerCase();
  for (let ti = 0; ti < tl.length && qi < ql.length; ti++) {
    if (tl[ti] === ql[qi]) {
      score += (ti === lastMatch + 1) ? 2 : 1; // bonus for consecutive
      if (ti === 0 || tl[ti - 1] === ' ' || tl[ti - 1] === '-' || tl[ti - 1] === '_') score += 3; // word boundary bonus
      lastMatch = ti;
      qi++;
    }
  }
  return qi === ql.length ? score : 0;
}

function renderPaperList(filter) {
  const f = (filter || document.getElementById('paper-search').value).trim();
  let filtered;
  if (!f) {
    filtered = papers.slice();
  } else {
    filtered = papers
      .map(p => {
        const s1 = fuzzyMatch(displayName(p), f);
        const s2 = fuzzyMatch(p.name || '', f);
        return { p, score: Math.max(s1, s2) };
      })
      .filter(x => x.score > 0)
      .sort((a, b) => b.score - a.score)
      .map(x => x.p);
  }
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
  const paper = papers.find(p => p.id === id);
  activeView = (paper && paper.pdf_key) ? 'pdf' : 'md';
  renderPaperList();
  document.getElementById('toolbar').classList.add('active');
  const content = document.getElementById('content');
  content.className = '';
  content.innerHTML = '<p style="color:var(--muted);padding:32px">Loading...</p>';

  document.getElementById('display-name-text').textContent = displayName(paper);
  document.getElementById('chat-context-label').textContent = displayName(paper);

  const vt = document.getElementById('view-toggle');
  vt.style.display = paper && paper.pdf_key ? 'flex' : 'none';
  updateViewToggle();

  const res = await fetch(API + '/papers/' + id + '/content', { headers: headers() });
  if (!res.ok) { content.innerHTML = '<p style="padding:32px">Error loading paper</p>'; return; }
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
      content.innerHTML = '<p style="padding:32px;color:var(--muted)">Loading PDF...</p>';
      content.style.padding = '0';
      fetch(API + '/papers/' + activePaperId + '/pdf', {
        headers: { 'Authorization': 'Bearer ' + TOKEN }
      }).then(r => r.blob()).then(blob => {
        const url = URL.createObjectURL(blob);
        content.innerHTML = '<iframe id="pdf-frame" src="' + url + '#navpanes=0"></iframe>';
      }).catch(() => {
        content.innerHTML = '<p style="padding:32px;color:var(--muted)">Error loading PDF</p>';
      });
    } else {
      content.innerHTML = '<p style="padding:32px;color:var(--muted)">No PDF available</p>';
    }
  } else {
    content.style.padding = '32px';
    content.innerHTML = '<div class="md-view">' + marked.parse(data.markdown || '') + '</div>';
    content.querySelectorAll('h1,h2,h3,h4,h5,h6').forEach((el, i) => { el.id = 'heading-' + i; });
  }
}

function renderOutline(hdrs) {
  const outline = document.getElementById('outline');
  if (!hdrs.length) { outline.innerHTML = ''; return; }
  outline.innerHTML = '<div class="toc-label">Contents</div>' + hdrs.map((h, i) => {
    const lvl = Math.min(h.level, 4);
    return '<div class="toc-item l' + lvl + '" onclick="scrollToHeading(' + i + ')" title="' + escapeHtml(h.text) + '">' + escapeHtml(h.text) + '</div>';
  }).join('');
}

function scrollToHeading(i) {
  const el = document.getElementById('heading-' + i);
  if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function toggleOutline() {
  document.getElementById('outline').classList.toggle('open');
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
        model: CHAT_MODEL,
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
