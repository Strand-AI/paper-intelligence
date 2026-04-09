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
  --chat-bg: #f3f4f6; --user-msg: #dbeafe; --ai-msg: #fff;
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

/* Layout */
#app { display: none; height: 100vh; grid-template-columns: 280px 1fr; grid-template-rows: 48px 1fr; }
#app.active { display: grid; }
header { grid-column: 1 / -1; background: var(--header); color: #fff; display: flex; align-items: center; justify-content: space-between; padding: 0 1rem; }
header h1 { font-size: 15px; font-weight: 600; letter-spacing: -0.01em; }
.header-right { display: flex; align-items: center; gap: 0.75rem; }
.header-right select { background: rgba(255,255,255,0.1); color: #fff; border: 1px solid rgba(255,255,255,0.2); padding: 4px 8px; border-radius: 6px; font-size: 13px; }
.header-right button { background: rgba(255,255,255,0.1); color: #fff; border: 1px solid rgba(255,255,255,0.2); padding: 4px 12px; border-radius: 6px; font-size: 13px; cursor: pointer; }

/* Sidebar */
#sidebar { background: var(--sidebar); border-right: 1px solid var(--border); overflow-y: auto; display: flex; flex-direction: column; }
.sidebar-search { padding: 0.75rem; border-bottom: 1px solid var(--border); }
.sidebar-search input { width: 100%; padding: 8px 12px; border: 1px solid var(--border); border-radius: 6px; font-size: 13px; }
.paper-list { flex: 1; overflow-y: auto; }
.paper-item { padding: 10px 16px; cursor: pointer; border-bottom: 1px solid var(--border); font-size: 13px; line-height: 1.4; }
.paper-item:hover { background: var(--bg); }
.paper-item.active { background: #eff6ff; border-left: 3px solid var(--accent); }
.paper-item .status { font-size: 11px; color: var(--muted); margin-top: 2px; }
.paper-item .status.ready { color: #10b981; }

/* Main */
#main { display: flex; flex-direction: column; overflow: hidden; }
#content { flex: 1; overflow-y: auto; padding: 2rem; max-width: 900px; }
#content.empty { display: flex; align-items: center; justify-content: center; color: var(--muted); font-size: 15px; max-width: none; }
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

/* Chat */
#chat { border-top: 1px solid var(--border); background: var(--chat-bg); display: flex; flex-direction: column; height: 300px; transition: height 0.2s; }
#chat.collapsed { height: 42px; }
#chat-header { display: flex; align-items: center; justify-content: space-between; padding: 8px 16px; cursor: pointer; font-size: 13px; font-weight: 600; user-select: none; }
#chat-header span:last-child { color: var(--muted); font-weight: 400; }
#chat-messages { flex: 1; overflow-y: auto; padding: 12px 16px; display: flex; flex-direction: column; gap: 8px; }
#chat.collapsed #chat-messages, #chat.collapsed #chat-input-row { display: none; }
.msg { padding: 8px 12px; border-radius: 8px; font-size: 14px; line-height: 1.6; max-width: 85%; white-space: pre-wrap; }
.msg.user { background: var(--user-msg); align-self: flex-end; }
.msg.assistant { background: var(--ai-msg); align-self: flex-start; border: 1px solid var(--border); }
.msg.assistant p { margin: 0.25rem 0; }
.msg.assistant p:first-child { margin-top: 0; }
#chat-input-row { display: flex; gap: 8px; padding: 8px 16px 12px; }
#chat-input { flex: 1; padding: 8px 12px; border: 1px solid var(--border); border-radius: 8px; font-size: 14px; font-family: var(--font); resize: none; }
#chat-send { padding: 8px 20px; background: var(--accent); color: #fff; border: none; border-radius: 8px; cursor: pointer; font-size: 14px; }
#chat-send:disabled { opacity: 0.5; cursor: not-allowed; }

.paper-count { padding: 8px 16px; font-size: 12px; color: var(--muted); border-bottom: 1px solid var(--border); }
</style>
</head>
<body>

<div id="auth-modal">
  <div class="auth-box">
    <h2>Paper Intelligence</h2>
    <p style="color:#6b7280;margin-bottom:1rem;font-size:14px">Enter your API token to continue.</p>
    <input type="password" id="auth-token" placeholder="Bearer token" autofocus>
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
    <div id="content" class="empty">Select a paper from the sidebar</div>
    <div id="chat" class="collapsed">
      <div id="chat-header" onclick="toggleChat()">
        <span>Chat</span>
        <span id="chat-context-label">Select a paper to chat</span>
      </div>
      <div id="chat-messages"></div>
      <div id="chat-input-row">
        <textarea id="chat-input" rows="1" placeholder="Ask about this paper..." onkeydown="if(event.key==='Enter'&&!event.shiftKey){event.preventDefault();sendChat()}"></textarea>
        <button id="chat-send" onclick="sendChat()">Send</button>
      </div>
    </div>
  </div>
</div>

<script>
let TOKEN = localStorage.getItem('pi_token') || '';
let papers = [];
let activePaperId = null;
let chatHistory = [];

const API = '';

function headers() {
  return { 'Authorization': 'Bearer ' + TOKEN, 'Content-Type': 'application/json' };
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

function logout() {
  TOKEN = '';
  localStorage.removeItem('pi_token');
  location.reload();
}

async function init() {
  const res = await fetch(API + '/papers', { headers: headers() });
  if (res.status === 401) { logout(); return; }
  const data = await res.json();
  papers = data.papers || [];
  renderPaperList();
}

function renderPaperList(filter = '') {
  const list = document.getElementById('paper-list');
  const f = filter.toLowerCase();
  const filtered = papers.filter(p => !f || p.name.toLowerCase().includes(f));
  document.getElementById('paper-count').textContent = filtered.length + ' papers';
  list.innerHTML = filtered.map(p =>
    '<div class="paper-item' + (p.id === activePaperId ? ' active' : '') + '" onclick="selectPaper(\\'' + p.id + '\\')">' +
      '<div>' + p.name.replace(/_/g, ' ').replace(/-/g, ' ') + '</div>' +
      '<div class="status ' + p.status + '">' + p.status + (p.chunk_count ? ' · ' + p.chunk_count + ' chunks' : '') + '</div>' +
    '</div>'
  ).join('');
}

document.getElementById('paper-search').addEventListener('input', e => renderPaperList(e.target.value));

async function selectPaper(id) {
  activePaperId = id;
  renderPaperList(document.getElementById('paper-search').value);
  const content = document.getElementById('content');
  content.className = '';
  content.innerHTML = '<p style="color:#6b7280">Loading...</p>';

  const res = await fetch(API + '/papers/' + id + '/content', { headers: headers() });
  if (!res.ok) { content.innerHTML = '<p>Error loading paper</p>'; return; }
  const data = await res.json();
  content.innerHTML = marked.parse(data.markdown || '');

  // Update chat context
  const paper = papers.find(p => p.id === id);
  document.getElementById('chat-context-label').textContent = paper ? paper.name.replace(/_/g, ' ') : '';
  chatHistory = [];
  document.getElementById('chat-messages').innerHTML = '';
  document.getElementById('chat').classList.remove('collapsed');
}

function toggleChat() {
  document.getElementById('chat').classList.toggle('collapsed');
}

async function sendChat() {
  const input = document.getElementById('chat-input');
  const msg = input.value.trim();
  if (!msg) return;
  input.value = '';

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
      method: 'POST',
      headers: headers(),
      body: JSON.stringify({
        message: msg,
        paper_ids: activePaperId ? [activePaperId] : [],
        model: document.getElementById('model-select').value,
        history: chatHistory.slice(0, -1),
      }),
    });

    if (!res.ok) {
      const err = await res.json();
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
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}
<\/script>
</body>
</html>`;
