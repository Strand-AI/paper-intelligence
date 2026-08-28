# Paper Intelligence

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![MCP](https://img.shields.io/badge/MCP-Compatible-green.svg)](https://modelcontextprotocol.io)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![uv](https://img.shields.io/badge/uv-Package%20Manager-purple.svg)](https://github.com/astral-sh/uv)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-orange.svg)](https://www.trychroma.com/)
[![LlamaIndex](https://img.shields.io/badge/LlamaIndex-RAG-red.svg)](https://www.llamaindex.ai/)

A local MCP server for intelligent paper/PDF management. Convert PDFs to markdown, then search them with hybrid grep + semantic search. Designed for **token efficiency**: search first, read only what you need.

## 🚀 Quick Start

### 1. Install UV (one-time setup)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Add to Your MCP Client

**Claude Code CLI:**
```bash
claude mcp add paper-intelligence -- uvx paper-intelligence@latest
```

**VS Code:**
```bash
code --add-mcp '{"name":"paper-intelligence","command":"uvx","args":["paper-intelligence@latest"]}'
```

That's it! `uvx` handles everything automatically. Using `@latest` ensures you always get the newest version.

## 🔌 MCP Client Integration

<details>
<summary><strong>Claude Desktop</strong></summary>

Add to your Claude Desktop config:
- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
- **Linux**: `~/.config/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "paper-intelligence": {
      "command": "uvx",
      "args": ["paper-intelligence@latest"]
    }
  }
}
```

</details>

<details>
<summary><strong>Cursor</strong></summary>

1. Go to **Settings → MCP → Add new MCP Server**
2. Select `command` type
3. Enter: `uvx paper-intelligence@latest`

Or add to `~/.cursor/mcp.json`:
```json
{
  "mcpServers": {
    "paper-intelligence": {
      "command": "uvx",
      "args": ["paper-intelligence@latest"]
    }
  }
}
```

</details>

<details>
<summary><strong>Windsurf / Other MCP Clients</strong></summary>

Any MCP-compatible client can use paper-intelligence:

```json
{
  "mcpServers": {
    "paper-intelligence": {
      "command": "uvx",
      "args": ["paper-intelligence@latest"]
    }
  }
}
```

</details>

## ✨ Features

- **PDF to Markdown** — High-accuracy conversion using [Marker](https://github.com/VikParuchuri/marker)
- **Hybrid Search** — Combined grep (exact/regex) + semantic RAG search
- **Token Efficient** — Search papers instead of reading entire documents
- **GPU Acceleration** — MPS (Apple Silicon) and CUDA support
- **Self-Contained** — Each paper gets its own directory with all data
- **Header Context** — Search results show document structure (e.g., "Methods > Data Collection")

## 📖 MCP Tools

### `search`

Search one or more explicitly requested PDFs, processed paper directories, or library
directories with `grep`, `rag`, or `hybrid` mode. Direct PDF searches never inspect
sibling files or directories.

**Parameters:**
- `query` (string): Text, regex, or semantic query
- `sources` (array): PDF paths, paper directories, or an explicitly selected library directory
- `mode` (string, optional): `"grep"`, `"rag"`, or `"hybrid"` (default: hybrid)
- `top_k` (integer, optional): Number of results (default: 5)
- `regex` (boolean, optional): Treat the grep query as a regex (default: false)

A new or incomplete PDF is converted, indexed, and embedded in the background because
this normally takes **1–3 minutes**, longer than the roughly 30-second deadline used by
many MCP clients. The first call returns promptly:

```json
{
  "success": true,
  "status": "processing",
  "message": "First-use processing ... is continuing in the background.",
  "processing": [{
    "paper_dir": "/path/to/paper",
    "retry_after_seconds": 30,
    "next_step": "Call get_paper_info ..."
  }]
}
```

Processing continues after that response. Poll `get_paper_info` using `paper_dir`; when
it reports `status: "ready"`, retry the original search. Already-processed grep searches
do not initialize the semantic model. RAG and hybrid searches initialize it when needed.
Search no longer performs an unconditional remote-library sync, which previously allowed
a local query to block for up to five minutes.

### `get_paper_info`

Check a paper's processing state without loading the embedding model. Pass either the
paper directory returned by `search` or the original PDF path.

Statuses are `queued`, `processing`, `ready`, `incomplete`, or `failed`. Failed responses
include the background job's error message. A ready response includes artifact presence,
metadata, and a lightweight local chunk count when available.

## 📊 Example Output

### Search Result

```json
{
  "source": "attention-is-all-you-need.md",
  "line_number": 142,
  "header_path": "Model Architecture > Attention",
  "content": "An attention function can be described as mapping a query and a set of key-value pairs to an output...",
  "score": 0.89
}
```

## 🎯 Typical Workflow

1. **Process a paper:**
   > Process the PDF at ~/Downloads/transformer-paper.pdf

2. **Search across papers:**
   > Search for "positional encoding" in my papers

3. **Read specific sections:**
   > Show me the Methods section from the transformer paper

The agent reads search results (a few hundred tokens) instead of entire papers (tens of thousands of tokens).

## 🛠️ Installation Options

<details>
<summary><strong>Install from PyPI</strong></summary>

```bash
# Install with pip
pip install paper-intelligence

# Or run directly with uvx (no install needed)
uvx paper-intelligence@latest
```

</details>

<details>
<summary><strong>Install from GitHub</strong></summary>

```bash
pip install "paper-intelligence @ git+https://github.com/Strand-AI/paper-intelligence.git"
```

</details>

<details>
<summary><strong>Local Development</strong></summary>

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

**Development MCP config:**
```json
{
  "mcpServers": {
    "paper-intelligence": {
      "command": "python",
      "args": ["-m", "paper_intelligence.server"],
      "cwd": "/path/to/paper-intelligence"
    }
  }
}
```

**Run tests:**
```bash
# Unit tests (fast)
pytest tests/test_markdown_parser.py

# Integration tests (slow, requires ML models)
pytest tests/test_integration.py -v
```

</details>

## 🔧 Debugging

Use the MCP Inspector to debug the server:

```bash
npx @modelcontextprotocol/inspector uvx paper-intelligence@latest
```

## 🆘 Troubleshooting

<details>
<summary><strong>Server not starting?</strong></summary>

- Ensure Python 3.11+ is installed
- Try `uvx paper-intelligence@latest` directly to see error messages
- Check that all dependencies installed correctly

</details>

<details>
<summary><strong>Windows encoding issues?</strong></summary>

Add to your MCP config:
```json
"env": {
  "PYTHONIOENCODING": "utf-8"
}
```

</details>

<details>
<summary><strong>Claude Desktop not detecting changes?</strong></summary>

Claude Desktop only reads configuration on startup. Fully restart the app after config changes.

</details>

## 🏗️ Technical Stack

| Component | Technology |
|-----------|------------|
| MCP Server | Official Python SDK with FastMCP |
| PDF Conversion | [marker-pdf](https://github.com/VikParuchuri/marker) |
| Embeddings | LlamaIndex + HuggingFace (BAAI/bge-small-en-v1.5) |
| Vector Store | ChromaDB (persistent, local per-paper) |
| GPU Support | PyTorch with MPS (Apple) or CUDA |

## 🙏 Acknowledgments

- [Marker](https://github.com/VikParuchuri/marker) for excellent PDF conversion
- [LlamaIndex](https://www.llamaindex.ai/) for the RAG framework
- [ChromaDB](https://www.trychroma.com/) for the vector database
- [FastMCP](https://github.com/modelcontextprotocol/python-sdk) for the MCP server framework

## 📄 License

MIT — see [LICENSE](LICENSE) for details.
