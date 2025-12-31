# Paper Intelligence MCP Server

A local MCP (Model Context Protocol) server for intelligent paper/PDF management with RAG capabilities.

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

### Claude Desktop

Add to your Claude Desktop config (`~/.config/claude/claude_desktop_config.json` on macOS/Linux or `%APPDATA%\Claude\claude_desktop_config.json` on Windows):

**Using uvx (recommended after PyPI publish):**
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

**Using local install:**
```json
{
  "mcpServers": {
    "paper-intelligence": {
      "command": "/path/to/paper-intelligence/.venv/bin/python",
      "args": ["-m", "paper_intelligence.server"]
    }
  }
}
```

### Claude Code

Add to your Claude Code config (`~/.claude.json`):

**Using uvx (recommended after PyPI publish):**
```json
{
  "mcpServers": {
    "paper-intelligence": {
      "type": "stdio",
      "command": "uvx",
      "args": ["paper-intelligence"]
    }
  }
}
```

**Using local install:**
```json
{
  "mcpServers": {
    "paper-intelligence": {
      "type": "stdio",
      "command": "/path/to/paper-intelligence/.venv/bin/python",
      "args": ["-m", "paper_intelligence.server"],
      "cwd": "/path/to/paper-intelligence"
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

## License

MIT
