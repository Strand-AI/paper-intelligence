# Paper Intelligence v1 Architecture Plan

## Vision

Transform paper-intelligence from a local MCP tool into a full-stack research platform with:
- **Hosted PDF processing** using Reducto for high-quality parsing
- **Cloud library** with vector search across all your papers
- **Team collaboration** with shared libraries and deduplication
- **Offline-capable desktop app** for reading and searching downloaded papers
- **MCP integration** that bridges local and cloud capabilities

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                           Users                                      │
├──────────────┬──────────────────────┬───────────────────────────────┤
│   Web App    │    Desktop App       │         MCP Server            │
│   (React)    │    (Electron)        │    (existing + API client)    │
│              │    ├─ Local SQLite   │                               │
│              │    ├─ Local Vectors  │                               │
│              │    └─ PDF Viewer     │                               │
└──────┬───────┴──────────┬───────────┴───────────────┬───────────────┘
       │                  │                           │
       │                  │ (sync)                    │
       ▼                  ▼                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         API Gateway                                  │
│                    (Auth, Rate Limiting)                            │
└─────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       Backend Services                               │
├─────────────────┬─────────────────┬─────────────────────────────────┤
│  PDF Processor  │  Search Service │      User/Team Service          │
│  (Reducto)      │  (Pinecone)     │      (Auth, Libraries)          │
└────────┬────────┴────────┬────────┴─────────────────────────────────┘
         │                 │
         ▼                 ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────────────────┐
│   Blob Storage  │ │    Pinecone     │ │      PostgreSQL             │
│   (S3/R2)       │ │   (Vectors)     │ │   (Users, Teams, Papers)    │
└─────────────────┘ └─────────────────┘ └─────────────────────────────┘
```

---

## Core Features

### 1. PDF Processing Pipeline

**Flow:**
1. User uploads PDF or provides URL
2. Compute file hash (SHA-256)
3. Check if hash exists in global index → **deduplicate**
4. If new: Send to Reducto for parsing
5. Extract text, figures, tables, equations
6. Generate embeddings (OpenAI ada-002 or similar)
7. Store in Pinecone with metadata
8. Store PDF blob and parsed markdown in S3/R2

**Deduplication Strategy:**
- Global content-addressed storage keyed by file hash
- Papers are stored once, referenced by many users/teams
- User libraries contain references, not copies
- Saves storage, compute, and provides instant access to already-indexed papers

### 2. User & Team System

**User Model:**
```
User
├── id (uuid)
├── email
├── name
├── auth_provider (google, github, email)
├── created_at
└── tier (free, pro, team, enterprise)

Team
├── id (uuid)
├── name
├── owner_id → User
├── members[] → User (with role: admin, member, viewer)
├── shared_library_id → Library
└── settings (permissions, defaults)

Library
├── id (uuid)
├── owner_type (user | team)
├── owner_id
└── papers[] → PaperReference

PaperReference
├── library_id
├── paper_id (global paper by hash)
├── added_at
├── added_by → User
├── tags[]
├── notes
└── collections[]
```

**Tiers (tentative):**
| Tier | Papers | Teams | Offline | Price |
|------|--------|-------|---------|-------|
| Free | 50 | - | 10 | $0 |
| Pro | Unlimited | - | Unlimited | TBD |
| Team | Unlimited | Yes | Unlimited | TBD/seat |
| Enterprise | Unlimited | Yes + SSO | Unlimited | Custom |

### 3. Web Application

**Stack:** React + Vite + TailwindCSS

**Pages:**
- `/` - Landing page
- `/login` - Auth (Google, GitHub, Email magic link)
- `/library` - Personal paper library
  - Grid/list view of papers
  - Search bar (vector + keyword)
  - Filters (tags, collections, date added)
  - Upload button (drag & drop, URL input)
- `/library/:id` - Paper detail view
  - Rendered markdown with figures
  - AI chat about the paper
  - Notes, highlights, tags
- `/team/:id` - Team library (same as personal but shared)
- `/settings` - Account, API keys, team management

### 4. Desktop Application (Electron)

**Core Capabilities:**
- Full web app functionality
- **Offline mode** for downloaded papers:
  - Local SQLite database for metadata
  - Local vector store (sqlite-vss or LanceDB) for embeddings
  - Downloaded PDFs + parsed markdown stored locally
- PDF reader with annotations
- Sync engine: pull updates when online, queue uploads for later

**Sync Logic:**
```
On startup:
  1. Check connectivity
  2. If online: sync library metadata, download new papers marked for offline
  3. Load local data regardless

On search:
  1. Always search local vectors for downloaded papers
  2. If online: also search cloud library
  3. Merge and deduplicate results

On upload (offline):
  1. Queue PDF in local processing queue
  2. When online: upload and process via server
  3. Update local index when complete
```

### 5. MCP Server Updates

**New Configuration:**
```json
{
  "mcpServers": {
    "paper-intelligence": {
      "command": "uvx",
      "args": ["paper-intelligence"],
      "env": {
        "PAPER_INTELLIGENCE_API_KEY": "pi_xxxxx",
        "PAPER_INTELLIGENCE_LOCAL_PATH": "~/Documents/papers"
      }
    }
  }
}
```

**Enhanced Tools:**

| Tool | Description |
|------|-------------|
| `search` | Hybrid search: local papers + cloud library (if API key) |
| `get_paper_info` | Get metadata, processing status |
| `upload_paper` | Upload PDF to cloud library |
| `download_paper` | Download paper for offline access |
| `list_library` | List papers in library (local + cloud) |
| `ask_paper` | RAG Q&A about a specific paper |

**Behavior:**
- Without API key: works exactly as today (local only)
- With API key:
  - Searches both local and cloud
  - Can upload/download papers
  - Deduplicates results by hash

---

## Backend Services

### API Endpoints

```
POST   /auth/login          # OAuth or magic link
POST   /auth/refresh        # Refresh token
DELETE /auth/logout         # Invalidate session

GET    /user/me             # Current user profile
PATCH  /user/me             # Update profile
GET    /user/api-keys       # List API keys
POST   /user/api-keys       # Create API key
DELETE /user/api-keys/:id   # Revoke API key

GET    /library             # List papers in personal library
POST   /library/papers      # Add paper (upload or URL)
GET    /library/papers/:id  # Get paper details
DELETE /library/papers/:id  # Remove from library
PATCH  /library/papers/:id  # Update tags, notes, etc.

POST   /search              # Vector + keyword search
POST   /search/paper/:id    # Search within a specific paper

GET    /teams               # List user's teams
POST   /teams               # Create team
GET    /teams/:id           # Team details
PATCH  /teams/:id           # Update team
DELETE /teams/:id           # Delete team
POST   /teams/:id/members   # Add member
DELETE /teams/:id/members/:uid  # Remove member
GET    /teams/:id/library   # Team library

POST   /papers/process      # Internal: trigger processing
GET    /papers/:hash        # Get paper by content hash
GET    /papers/:hash/download  # Download PDF or markdown
```

### Processing Queue

Use a job queue (BullMQ, Celery, or similar) for PDF processing:

```
Job: ProcessPaper
├── paper_id
├── source_url or blob_ref
├── requested_by (user_id)
├── status: pending | processing | completed | failed
├── progress: 0-100
├── reducto_job_id
└── error_message (if failed)
```

Webhook from Reducto triggers completion handler:
1. Store parsed content in S3
2. Generate embeddings
3. Upsert to Pinecone
4. Update paper status in DB
5. Notify user (websocket or email)

---

## Data Storage

### PostgreSQL Schema (simplified)

```sql
-- Users and auth
CREATE TABLE users (
  id UUID PRIMARY KEY,
  email TEXT UNIQUE NOT NULL,
  name TEXT,
  avatar_url TEXT,
  tier TEXT DEFAULT 'free',
  created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE api_keys (
  id UUID PRIMARY KEY,
  user_id UUID REFERENCES users(id),
  key_hash TEXT NOT NULL,  -- store hashed, prefix visible
  name TEXT,
  last_used_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Teams
CREATE TABLE teams (
  id UUID PRIMARY KEY,
  name TEXT NOT NULL,
  owner_id UUID REFERENCES users(id),
  created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE team_members (
  team_id UUID REFERENCES teams(id),
  user_id UUID REFERENCES users(id),
  role TEXT DEFAULT 'member',  -- admin, member, viewer
  joined_at TIMESTAMPTZ DEFAULT NOW(),
  PRIMARY KEY (team_id, user_id)
);

-- Papers (global, deduplicated)
CREATE TABLE papers (
  id UUID PRIMARY KEY,
  content_hash TEXT UNIQUE NOT NULL,  -- SHA-256 of PDF
  title TEXT,
  authors TEXT[],
  abstract TEXT,
  source_url TEXT,
  arxiv_id TEXT,
  doi TEXT,
  blob_key TEXT,        -- S3 key for PDF
  markdown_key TEXT,    -- S3 key for parsed markdown
  pinecone_namespace TEXT,
  status TEXT DEFAULT 'pending',
  processed_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Library references
CREATE TABLE library_papers (
  id UUID PRIMARY KEY,
  library_type TEXT NOT NULL,  -- 'user' or 'team'
  library_owner_id UUID NOT NULL,  -- user_id or team_id
  paper_id UUID REFERENCES papers(id),
  added_by UUID REFERENCES users(id),
  tags TEXT[],
  notes TEXT,
  is_offline BOOLEAN DEFAULT FALSE,  -- marked for offline
  added_at TIMESTAMPTZ DEFAULT NOW(),
  UNIQUE (library_type, library_owner_id, paper_id)
);
```

### Pinecone Structure

**Index:** `paper-intelligence`

**Namespaces:** One per paper (by content_hash)

**Vector Metadata:**
```json
{
  "paper_hash": "abc123...",
  "chunk_index": 42,
  "page_number": 7,
  "section": "Methods",
  "content_type": "text|figure_caption|table|equation",
  "line_start": 150,
  "line_end": 175
}
```

---

## Future Enhancements (v2+)

### Auto-indexing Sources
- **arXiv**: Monitor RSS feeds for new papers in selected categories
- **PubMed**: Track specific journals or keywords
- **Semantic Scholar**: Follow authors or topics
- Users can "subscribe" to sources, auto-add matching papers

### AI Features
- Paper summarization on upload
- Related paper recommendations
- Citation graph visualization
- "Explain this paper" chat mode
- Cross-paper synthesis ("What do these 5 papers say about X?")

### Collaboration
- Shared annotations and highlights
- Comments on specific passages
- Reading lists and paper clubs
- Activity feed for team libraries

### Integrations
- Zotero import/export
- Mendeley import
- BibTeX export
- Notion/Obsidian sync
- Slack notifications for team activity

---

## Implementation Phases

### Phase 1: Backend Foundation
- [ ] Set up API server (FastAPI or similar)
- [ ] PostgreSQL schema and migrations
- [ ] Auth system (OAuth + API keys)
- [ ] Reducto integration for PDF processing
- [ ] Pinecone setup and embedding pipeline
- [ ] S3/R2 for blob storage
- [ ] Basic API endpoints (upload, search, library CRUD)

### Phase 2: Web Application
- [ ] React app scaffolding
- [ ] Auth flow (login, signup, OAuth)
- [ ] Library view (list papers, search)
- [ ] Paper upload (file + URL)
- [ ] Paper detail view
- [ ] Basic team support

### Phase 3: MCP Server v2
- [ ] Add API key configuration
- [ ] Implement cloud search alongside local
- [ ] Upload/download paper tools
- [ ] Hybrid search with result merging

### Phase 4: Desktop App
- [ ] Electron app shell
- [ ] Embed web app
- [ ] Local SQLite + vector store setup
- [ ] Offline paper download and sync
- [ ] Local-first search implementation

### Phase 5: Polish & Launch
- [ ] Onboarding flow
- [ ] Pricing page and Stripe integration
- [ ] Documentation and guides
- [ ] Beta launch to early users

---

## Open Questions

1. **Hosting**: Vercel/Railway for API? Cloudflare Workers? Self-hosted?
2. **Reducto pricing**: Need to estimate cost per paper for pricing model
3. **Embedding model**: OpenAI ada-002, or open-source (e5, bge)?
4. **Mobile**: Native apps eventually, or PWA sufficient?
5. **PDF viewer**: Build custom or use existing (pdf.js, react-pdf)?

---

## Tech Stack Summary

| Component | Technology |
|-----------|------------|
| API Server | FastAPI (Python) or Hono (TypeScript) |
| Database | PostgreSQL (Neon or Supabase) |
| Vector DB | Pinecone |
| Blob Storage | Cloudflare R2 or AWS S3 |
| PDF Parsing | Reducto |
| Embeddings | OpenAI ada-002 |
| Auth | Clerk or Auth.js |
| Web App | React + Vite + TailwindCSS |
| Desktop App | Electron + Local SQLite + LanceDB |
| MCP Server | Python (existing codebase) |
| Job Queue | BullMQ or Celery |
| Hosting | TBD |
