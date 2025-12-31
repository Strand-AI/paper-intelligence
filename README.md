# Paper Intelligence MCP Server

A local MCP (Model Context Protocol) server for intelligent paper/PDF management with RAG capabilities.

## Quick Start

**Claude Code CLI:**
```bash
claude mcp add paper-intelligence -- uvx paper-intelligence
```

**VS Code:**
```bash
code --add-mcp '{"name":"paper-intelligence","command":"uvx","args":["paper-intelligence"]}'
```

## Features

- **PDF to Markdown**: Convert PDFs using [Marker](https://github.com/VikParuchuri/marker) with high accuracy
- **Header Indexing**: Extract document structure into searchable JSON
- **Semantic Search**: RAG-powered search using LlamaIndex + ChromaDB + HuggingFace embeddings
- **Hybrid Search**: Combined grep (text/regex) + semantic search
- **GPU Acceleration**: MPS (Apple Silicon) and CUDA support
- **Self-contained**: Each paper gets its own directory with all data
- **Version Tracking**: Metadata tracks which version processed each paper

## Installation

### Option 1: Install from PyPI (Recommended)

```bash
# Install with pip
pip install paper-intelligence

# Or run directly with uvx (no install needed)
uvx paper-intelligence
```

### Option 2: Install from GitHub

```bash
# Install directly from GitHub (no clone needed)
pip install "paper-intelligence @ git+https://github.com/Strand-AI/paper-intelligence.git"
```

### Option 3: Local Development

```bash
git clone https://github.com/Strand-AI/paper-intelligence.git
cd paper-intelligence

# Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install in development mode
pip install -e ".[dev]"

# Run the server
python -m paper_intelligence.server
```

## MCP Client Configuration

### Claude Code CLI

The easiest way to add the server:

```bash
claude mcp add paper-intelligence -- uvx paper-intelligence
```

Verify installation:
```bash
claude mcp list
```

### Claude Desktop

Add to your Claude Desktop config:
- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
- **Linux**: `~/.config/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "paper-intelligence": {
      "command": "uvx",
      "args": ["paper-intelligence"]
    }
  }
}
```

### VS Code

One-liner install:
```bash
code --add-mcp '{"name":"paper-intelligence","command":"uvx","args":["paper-intelligence"]}'
```

Or manually add to your User Settings (JSON) or `.vscode/mcp.json`:
```json
{
  "mcp": {
    "servers": {
      "paper-intelligence": {
        "command": "uvx",
        "args": ["paper-intelligence"]
      }
    }
  }
}
```

### Cursor

1. Go to **Settings → MCP → Add new MCP Server**
2. Select `command` type
3. Enter: `uvx paper-intelligence`

Or add to your Cursor MCP config:
```json
{
  "mcpServers": {
    "paper-intelligence": {
      "command": "uvx",
      "args": ["paper-intelligence"]
    }
  }
}
```

### Windsurf

Add to your Windsurf MCP configuration:
```json
{
  "mcpServers": {
    "paper-intelligence": {
      "command": "uvx",
      "args": ["paper-intelligence"]
    }
  }
}
```

## Output Structure

For `~/Downloads/paper.pdf`, creates `~/Downloads/paper/`:
```
paper/
├── paper.md        # Converted markdown
├── metadata.json   # Processing version and info
├── index.json      # Header hierarchy (for search context)
├── chroma/         # Embeddings database
└── images/         # Extracted images (if any)
```

## MCP Tools

### `process_paper`
Full pipeline: Convert PDF, index headers, and create embeddings.

```python
process_paper(
    pdf_path="~/Downloads/paper.pdf",
    use_llm=False,      # Set True for enhanced accuracy
    chunk_size=512,
    chunk_overlap=50
)
# Returns: output_dir, markdown_path, images_dir (if images extracted), image_count
```

### `convert_pdf`
Convert a PDF file to Markdown.

```python
convert_pdf(
    pdf_path="~/Downloads/paper.pdf",
    output_dir=None,  # Defaults to ~/Downloads/paper/
    use_llm=False
)
# Returns: markdown_path, images_dir (if images extracted), image_count
```

### `index_markdown`
Extract header hierarchy into searchable JSON.

```python
index_markdown(
    markdown_path="~/Downloads/paper/paper.md"
)
```

### `embed_document`
Create embeddings for semantic search.

```python
embed_document(
    markdown_path="~/Downloads/paper/paper.md",
    chunk_size=512,
    chunk_overlap=50
)
```

### `search`
Unified search with grep and/or RAG.

```python
search(
    query="transformer attention mechanism",
    paper_dirs=["~/Downloads/paper1", "~/Downloads/paper2"],
    mode="hybrid",  # "grep", "rag", or "hybrid"
    top_k=5
)
```

### `get_paper_info`
Check processing status of a paper directory.

```python
get_paper_info("~/Downloads/paper")
# Returns: has_markdown, has_index, has_embeddings, has_images,
#          images_dir, image_files, image_count,
#          version info, metadata
```

## Extracted Images

When PDFs contain images (figures, diagrams, etc.), they are automatically extracted to an `images/` subdirectory. The agent using this MCP server can:

1. Check `get_paper_info()` to see if images exist and get the `images_dir` path
2. Access individual image files listed in `image_files`
3. Reference images from the converted markdown (images are linked in the `.md` file)

## Version Compatibility

Each processed paper directory includes a `metadata.json` file tracking:

- `paper_intelligence_version`: Version used for processing
- `processed_at`: Timestamp of processing
- `source_pdf`: Original PDF filename
- `steps_completed`: Which processing steps were run

When accessing papers, `get_paper_info()` checks version compatibility and warns if re-processing might be beneficial.

## How Search Uses index.json

The `index.json` file stores the header hierarchy extracted from the markdown. When you search:

1. **Grep search**: Uses `index.json` to provide header context for matches (e.g., "Methods > Data Collection")
2. **RAG search**: Returns semantic matches from the embedded chunks

The index enables fast header lookups without re-parsing the markdown on each search.

## Technical Stack

- **MCP**: Official Python SDK with FastMCP
- **PDF Conversion**: marker-pdf
- **Embeddings**: LlamaIndex + HuggingFace (BAAI/bge-small-en-v1.5)
- **Vector Store**: ChromaDB (persistent, local per-paper)
- **GPU**: PyTorch with MPS (Apple Silicon) or CUDA support

## Development

```bash
pip install -e ".[dev]"
pytest
```

To use your local development version with MCP clients, replace `uvx paper-intelligence` with:
```bash
python -m paper_intelligence.server
```

## Debugging

Use the MCP Inspector to debug the server:

```bash
npx @modelcontextprotocol/inspector uvx paper-intelligence
```

## Troubleshooting

**Server not starting?**
- Ensure Python 3.11+ is installed
- Try `uvx paper-intelligence` directly to see error messages
- Check that all dependencies installed correctly

**Windows encoding issues?**
Add to your config:
```json
"env": {
  "PYTHONIOENCODING": "utf-8"
}
```

**Claude Desktop not detecting changes?**
Claude Desktop only reads configuration on startup. Fully restart the app after config changes.

## License

MIT
