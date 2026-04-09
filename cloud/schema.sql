-- Paper Intelligence Cloud — D1 Schema

CREATE TABLE IF NOT EXISTS papers (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  title TEXT,                       -- auto-parsed from first H1
  alias TEXT,                       -- user-set display name
  pdf_key TEXT,
  markdown_key TEXT,
  markdown_text TEXT,              -- full markdown for grep search
  status TEXT NOT NULL DEFAULT 'uploading',
  error TEXT,
  page_count INTEGER,
  markdown_length INTEGER,
  chunk_count INTEGER,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  processed_at TEXT,
  version TEXT NOT NULL DEFAULT '0.2.0'
);

CREATE TABLE IF NOT EXISTS headers (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  paper_id TEXT NOT NULL,
  level INTEGER NOT NULL,
  text TEXT NOT NULL,
  line_number INTEGER NOT NULL,
  path TEXT NOT NULL,
  FOREIGN KEY (paper_id) REFERENCES papers(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS images (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  paper_id TEXT NOT NULL,
  r2_key TEXT NOT NULL,
  filename TEXT NOT NULL,
  FOREIGN KEY (paper_id) REFERENCES papers(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_headers_paper ON headers(paper_id);
CREATE INDEX IF NOT EXISTS idx_headers_line ON headers(paper_id, line_number);
CREATE INDEX IF NOT EXISTS idx_images_paper ON images(paper_id);
CREATE INDEX IF NOT EXISTS idx_papers_status ON papers(status);
CREATE INDEX IF NOT EXISTS idx_papers_name ON papers(name);
