# PaperMill
AI-powered academic writing assistant with a local-first RAG pipeline and built-in content safety scanning against prompt injection and data exfiltration.

## Features

- **PDF & Image Ingestion** — Parse PDFs with heading-aware chunking (PyMuPDF), describe images via Florence-2 (caption + OCR), embed everything with Jina Embeddings v3. DOIs are auto-extracted from PDF metadata and first-page text.
- **Vector Search** — LanceDB-backed semantic search across all ingested documents
- **RAG Chat** — Ask questions about your references with source citations and persistent chat history
- **Section Drafting** — Draft and rewrite paper sections against a 41-section outline
- **Bibliography** — Auto-fetch metadata from CrossRef by DOI with retry logic, APA formatting, DOI validation
- **Export** — Generate .docx files with academic formatting, page breaks between chapters, auto-generated References chapter with APA hanging indent, versioned snapshots
- **Content Safety Scanner** — Defense-in-depth against prompt injection and exfiltration in ingested documents
- **Startup Validation** — Config checks at launch (directory writability, Ollama reachability, API key presence)
- **Dual Interface** — CLI REPL and Streamlit web UI with runtime LLM provider switching

## Stack

| Component | Technology |
|---|---|
| Embeddings | [Jina Embeddings v3](https://huggingface.co/jinaai/jina-embeddings-v3) (1024-dim, local GPU) |
| Image Captioning | [Florence-2-large](https://huggingface.co/microsoft/Florence-2-large) (local GPU) |
| Vector DB | [LanceDB](https://lancedb.com/) (local, serverless) |
| PDF Parsing | [PyMuPDF](https://pymupdf.readthedocs.io/) |
| LLM (default) | [Ollama](https://ollama.com/) — any local model |
| LLM (alt) | Claude Sonnet / Opus via Anthropic API |
| Doc Export | python-docx (Times New Roman 12pt, 1.5 spacing, 1" margins) |
| Web UI | [Streamlit](https://streamlit.io/) |
| Safety Scanner | Regex + heuristic + LLM classifier (pluggable backends) |

## Quick Start

### Prerequisites

- Python 3.11+
- CUDA-capable GPU (for local embeddings and Florence-2)
- [Ollama](https://ollama.com/) running locally (default LLM provider)

### Install

```bash
git clone https://github.com/cr231521/PaperMill.git
cd PaperMill
pip install -r requirements.txt
```

### Pull an Ollama model

```bash
ollama pull dolphin-llama3
```

### Run

**CLI:**
```bash
python run_cli.py
```

**Streamlit:**
```bash
python run_streamlit.py
```

## Usage

### CLI Commands

```
/ingest [path]           Ingest a PDF (default: all in data/pdfs/)
/ingest-images [path]    Ingest images (default: all in data/images/)
/sources                 List ingested sources
/delete-source <name>    Delete a source from the vector DB
/outline                 Show paper outline with draft status
/draft <id>              Draft a section (e.g. /draft ch1)
/rewrite <id>            Rewrite a section with the polish model
/show <id>               Show a section draft
/export [id]             Export full paper or a section to .docx
/version [label]         Save a versioned snapshot
/versions                List all versions
/diff <v1> <v2>          Compare two versions
/ref-add <DOI>           Add a reference by DOI (auto-fetches metadata)
/refs                    List all bibliography entries
/ref-remove <key>        Remove a reference
/scan-rules              List active scanner rules
/allowlist-add <path>    Add file to scanner allowlist
/allowlist-remove <hash> Remove hash from allowlist
/quarantine-release      List/release quarantined files
/clear-history           Clear persistent chat history
```

Free text input triggers RAG-powered chat with source citations. Chat history persists across restarts.

### Streamlit UI

The web interface provides the same functionality with:
- LLM provider switcher (Ollama / Claude) in the sidebar
- File upload for PDFs and images
- Section drafting and rewriting
- Bibliography management
- Content scanner dashboard (rules, reports, quarantine)
- Performance stats after each generation

## Content Safety Scanner

The scanner protects the RAG pipeline against prompt injection and data exfiltration hidden in ingested documents.

### Defense Points

| Gate | When | Method |
|---|---|---|
| **Ingestion (PDF)** | Before embedding | Structural scan + metadata scan + per-chunk content scan |
| **Ingestion (Image)** | Before embedding | OCR/caption divergence check + content scan |
| **Retrieval** | Before LLM context | Regex-only scan on retrieved chunks |
| **Chat input** | Before query | Regex-only scan with chat scope (warnings, not hard block) |

### How It Works

1. **RegexBackend** — 15 patterns detecting role hijacking, system prompt leaks, delimiter injection, instruction overrides, and exfiltration attempts. Supports evasion-resistant "super-cleaned" matching for spaced-out text (`i g n o r e` -> `ignore`)
2. **HeuristicBackend** — Scores text on imperative command density and second-person pronoun density (0.0-1.0 suspicion score)
3. **OllamaClassifierBackend** — LLM-based classifier, triggered when heuristic score exceeds threshold. **Fail-closed**: blocks on parse errors or connection failures

Blocked files are quarantined and a Markdown report is generated. Files can be released from quarantine via CLI or Streamlit UI.

### Scanner Rules

Rules are stored in `data/scanner_rules.json` — inspectable and editable without code changes:

```json
{
  "id": "role_hijack_01",
  "category": "role_hijack",
  "severity": "high",
  "scope": "all",
  "pattern": "ignore (all |any )?(previous|prior|above) (instructions|prompts|rules|context)",
  "description": "Attempts to override system instructions"
}
```

### Extensibility

The scanner uses a pluggable `ScannerBackend` base class. Future backends (NeMo Guardrails, Llama Guard) can be added without modifying existing code.

## Project Structure

```
PaperMill/
├── config.py                  # All configuration constants
├── run_cli.py                 # CLI entry point
├── core/
│   ├── scanner.py             # Content safety scanner
│   ├── ingestion.py           # PDF ingestion pipeline
│   ├── image_ingestion.py     # Image ingestion (Florence-2)
│   ├── retrieval.py           # Vector search with safety gate
│   ├── generation.py          # LLM generation (chat, draft, rewrite)
│   ├── embedder.py            # Jina v3 embedding
│   ├── db.py                  # LanceDB schema and helpers
│   ├── bibliography.py        # DOI lookup, APA formatting
│   ├── prompts.py             # System prompt and templates
│   ├── paper_structure.py     # 41-section paper outline
│   ├── docexport.py           # .docx export
│   └── versioning.py          # Snapshot versioning
├── interfaces/
│   ├── cli.py                 # CLI REPL
│   └── streamlit_app.py       # Streamlit web UI
├── data/
│   ├── pdfs/                  # Reference PDFs
│   ├── images/                # Images for ingestion
│   ├── references.json        # Bibliography entries
│   ├── scanner_rules.json     # Scanner regex patterns
│   └── scanner_allowlist.json # Allowlisted file hashes
├── reports/                   # Safety scan reports
├── quarantine/                # Blocked files
├── paper_sections/            # Section drafts (JSON)
├── versions/                  # Versioned .docx snapshots
├── tests/                     # 50 tests
└── requirements.txt
```

## Testing

```bash
python -m pytest tests/ -v
```

50 tests covering all scanner components: data structures, normalization, regex matching, heuristic scoring, LLM classifier (mocked), pipeline orchestration, PDF structural/metadata scanning, report generation, scan history, quarantine, OCR divergence, ingestion integration, retrieval gate, and chat input scanning.

## Configuration

Key settings in `config.py`:

| Setting | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `ollama` | `ollama` or `claude` |
| `OLLAMA_DRAFT_MODEL` | `dolphin-llama3:latest` | Model for drafting and chat |
| `SCANNER_LLM_ESCALATION` | `True` | Enable LLM classifier escalation |
| `SCANNER_SUSPICION_THRESHOLD` | `0.6` | Heuristic score to trigger LLM |
| `SCANNER_MAX_LLM_ESCALATIONS` | `10` | Max LLM calls per file |
| `SCANNER_DRY_RUN` | `False` | Log threats without blocking |
| `SCANNER_SCAN_CHAT_INPUT` | `True` | Scan user chat messages |
| `MAX_CONTEXT_TOKENS` | `180000` | Max tokens before prompt truncation |
| `PAPER_TITLE` | *(your thesis title)* | Title used in prompts and .docx export |

## Security

- SQL injection protection on all LanceDB WHERE clauses
- Chunk text escaped in prompt templates (fenced code blocks) to prevent prompt injection via ingested content
- Ollama error handling with clear messages on connection failures
- Token count estimation with automatic truncation to prevent context overflow
- DOI format validation before CrossRef API calls

## AI Acknowledgment

This project was co-developed with **Claude (Anthropic)** for code generation and review. At runtime, the tool uses **Ollama / Claude** for RAG-powered drafting and content safety classification. All code was reviewed, tested, and validated by the author.

**Suggested thesis disclosure:**

> Portions of this manuscript were drafted using PaperMill, a RAG tool that synthesizes content from ingested references using [Ollama/Claude]. All AI-assisted drafts were verified against primary sources and revised by the author. The AI did not generate original findings or conclusions. Source code: https://github.com/cr231521/PaperMill.

## License

[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) — Free for academic and personal use with attribution. Commercial use requires permission.
