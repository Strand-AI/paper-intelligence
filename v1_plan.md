# Paper Intelligence v1 Architecture Plan

## Vision

Transform paper-intelligence from a local MCP tool into a full-stack research platform with:
- **Hosted PDF processing** using Reducto for high-quality parsing
- **Cloud library** with vector search across all your papers
- **Team collaboration** with shared libraries and deduplication
- **Offline-capable desktop app** for reading and searching downloaded papers
- **MCP integration** that bridges local and cloud capabilities

---

## Current Status

**Repository:** [github.com/Strand-AI/paper-intelligence-cloud](https://github.com/Strand-AI/paper-intelligence-cloud)

**Live API:** `https://paper-intel-api-793508173682.us-central1.run.app`

### Completed
- [x] GCP project setup (paper-intelligence-cloud)
- [x] Cloud SQL with PostgreSQL 15 + pgvector
- [x] Cloud Storage bucket for PDFs and markdown
- [x] Artifact Registry for Docker images
- [x] Database schema with Alembic migrations
- [x] Vertex AI embedding service (768 dimensions)
- [x] Reducto PDF processing integration
- [x] GCS storage service with content-addressed storage
- [x] Core API endpoints (upload, search, paper details)
- [x] Cloud Run deployment with Secret Manager

### In Progress
- [ ] Authentication (Auth.js with Google OAuth)
- [ ] Web application frontend
- [ ] End-to-end testing of upload/search flow

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
│                      Cloud Run API                                   │
│                 (FastAPI + Auth + Rate Limiting)                     │
└─────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       Backend Services                               │
├─────────────────┬─────────────────┬─────────────────────────────────┤
│  PDF Processor  │  Search Service │      User/Team Service          │
│  (Reducto)      │  (pgvector)     │      (Auth, Libraries)          │
└────────┬────────┴────────┬────────┴─────────────────────────────────┘
         │                 │
         ▼                 ▼
┌─────────────────┐ ┌─────────────────────────────────────────────────┐
│  Cloud Storage  │ │      Cloud SQL (PostgreSQL + pgvector)          │
│     (GCS)       │ │      (Users, Teams, Papers, Embeddings)         │
└─────────────────┘ └─────────────────────────────────────────────────┘
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
6. Generate embeddings (Vertex AI `text-embedding-004`, 768 dimensions)
7. Store embeddings in pgvector
8. Store PDF blob and parsed markdown in GCS

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

### API Endpoints (Implemented)

```
GET    /health              # Health check
GET    /docs                # Swagger UI

POST   /papers/upload       # Upload PDF for processing
GET    /papers/{id}         # Get paper details
POST   /papers/search       # Vector search across papers
GET    /papers/{id}/chunks  # Get paper chunks with embeddings
```

### API Endpoints (Planned)

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
DELETE /library/papers/:id  # Remove from library
PATCH  /library/papers/:id  # Update tags, notes, etc.

GET    /teams               # List user's teams
POST   /teams               # Create team
GET    /teams/:id           # Team details
PATCH  /teams/:id           # Update team
DELETE /teams/:id           # Delete team
POST   /teams/:id/members   # Add member
DELETE /teams/:id/members/:uid  # Remove member
GET    /teams/:id/library   # Team library
```

### Processing Flow

PDF upload triggers synchronous processing:
1. Compute content hash
2. Check for existing paper (deduplication)
3. Upload PDF to GCS
4. Send to Reducto for parsing
5. Store parsed markdown in GCS
6. Chunk content and generate embeddings
7. Store chunks with embeddings in pgvector
8. Update paper status

---

## Data Storage

### PostgreSQL Schema (Implemented)

```sql
-- Users and auth
CREATE TABLE users (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  email TEXT UNIQUE NOT NULL,
  name TEXT,
  avatar_url TEXT,
  tier TEXT DEFAULT 'free',
  created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE api_keys (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES users(id),
  key_hash TEXT NOT NULL,
  name TEXT,
  last_used_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Teams
CREATE TABLE teams (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name TEXT NOT NULL,
  owner_id UUID REFERENCES users(id),
  created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE team_members (
  team_id UUID REFERENCES teams(id),
  user_id UUID REFERENCES users(id),
  role TEXT DEFAULT 'member',
  joined_at TIMESTAMPTZ DEFAULT NOW(),
  PRIMARY KEY (team_id, user_id)
);

-- Papers (global, deduplicated)
CREATE TABLE papers (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  content_hash TEXT UNIQUE NOT NULL,
  title TEXT,
  authors TEXT[],
  abstract TEXT,
  source_url TEXT,
  arxiv_id TEXT,
  doi TEXT,
  blob_key TEXT,
  markdown_key TEXT,
  status TEXT DEFAULT 'pending',
  processed_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Paper chunks with embeddings (pgvector)
CREATE TABLE paper_chunks (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  paper_id UUID REFERENCES papers(id) ON DELETE CASCADE,
  chunk_index INTEGER NOT NULL,
  content TEXT NOT NULL,
  embedding VECTOR(768),  -- Vertex AI text-embedding-004
  page_number INTEGER,
  section TEXT,
  content_type TEXT,
  line_start INTEGER,
  line_end INTEGER,
  created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX ix_paper_chunks_paper_id ON paper_chunks(paper_id);

-- Library references
CREATE TABLE library_papers (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  library_type VARCHAR(10) NOT NULL,
  library_owner_id UUID NOT NULL,
  paper_id UUID REFERENCES papers(id),
  added_by UUID REFERENCES users(id),
  tags TEXT[],
  notes TEXT,
  is_offline BOOLEAN DEFAULT FALSE,
  added_at TIMESTAMPTZ DEFAULT NOW(),
  UNIQUE (library_type, library_owner_id, paper_id)
);
```

### pgvector Search

Vector similarity search using cosine distance:
```sql
SELECT p.*, pc.content, pc.chunk_index,
       1 - (pc.embedding <=> $1) as similarity
FROM paper_chunks pc
JOIN papers p ON pc.paper_id = p.id
ORDER BY pc.embedding <=> $1
LIMIT 10;
```

---

## GCP Resources

| Resource | Name | Details |
|----------|------|---------|
| Project | `paper-intelligence-cloud` | ID: 793508173682 |
| Cloud SQL | `paper-intel-db` | PostgreSQL 15 + pgvector, us-central1, db-f1-micro |
| Cloud Storage | `paper-intelligence-files` | For PDFs and markdown |
| Artifact Registry | `paper-intel` | Docker images, us-central1 |
| Cloud Run | `paper-intel-api` | API service |
| Secrets | `database-url`, `reducto-api-key` | Secret Manager |

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

### Phase 1: Backend Foundation ✅ COMPLETE
- [x] Set up GCP project and enable APIs
- [x] Cloud SQL with PostgreSQL + pgvector
- [x] Cloud Storage bucket
- [x] Artifact Registry for Docker images
- [x] FastAPI server with async SQLAlchemy
- [x] Database schema and Alembic migrations
- [x] Reducto integration for PDF processing
- [x] Vertex AI embedding service
- [x] GCS blob storage service
- [x] Basic API endpoints (upload, search)
- [x] Deploy to Cloud Run

### Phase 2: Web Application 🔄 IN PROGRESS
- [ ] Set up Auth.js with Google OAuth
- [ ] Create landing page
- [ ] Implement auth flow (login, signup)
- [ ] Build library view (list papers, search)
- [ ] Build paper upload UI (drag & drop, URL input)
- [ ] Build paper detail view with PDF.js
- [ ] Deploy web app to Cloud Run or Firebase Hosting

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

## Tech Stack Summary

| Component | Technology |
|-----------|------------|
| **Hosting** | **Google Cloud Platform** |
| API Server | Cloud Run (FastAPI) |
| Database | Cloud SQL (PostgreSQL 15) |
| Vector DB | pgvector (PostgreSQL extension) |
| Blob Storage | Cloud Storage (GCS) |
| PDF Parsing | Reducto |
| Embeddings | Vertex AI text-embedding-004 (768 dim) |
| Auth | Auth.js |
| Web App | React + Vite + TailwindCSS |
| Desktop App | Electron + Local SQLite + LanceDB |
| PDF Viewer | PDF.js (via react-pdf) |
| MCP Server | Python (existing codebase) |
| Secrets | Secret Manager |
| CI/CD | GitHub Actions |
