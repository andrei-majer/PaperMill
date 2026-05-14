# PaperMill

PaperMill with AI Co-Authoring features and built-in content safety scanning against prompt injection and data exfiltration.

## Overview

The system ingests reference PDFs and images, stores them as vector embeddings in a local database, and uses retrieval-augmented generation (RAG) to draft, rewrite, and answer questions about paper sections — all grounded in your actual source material. Supports dual retrieval: vector search (embedding similarity) or tree search (LLM-based reasoning over hierarchical document structure).

PaperMill does not conduct original research or generate findings. It is a writing tool: the researcher provides the sources, the research questions, and validates the output.

Two interfaces are provided: a command-line REPL and a Streamlit web UI, both sharing the same backend.

---

## Architecture

```
  Reference PDFs / Images
       |
       v
  [Content Safety Scanner] --> regex + heuristic + LLM classifier
       |  (block or pass)
       v
  [pdfplumber Parser / Florence-2] --> heading-aware chunking / caption+OCR
       |
       v
  [Embeddings (Jina v3 / Ollama / OpenAI)] --> configurable vectors (GPU-accelerated)
       |
       v
  [LanceDB] --> local vector database (no server needed)
       |
       v
  [Vector Search + Safety Gate] --> top-k relevant chunks (regex re-scan)
       |                    OR
  [Tree Search + Safety Gate] --> LLM reasons over document structure index
       |
       v
  [Ollama / Claude / OpenAI / OpenRouter] --> source-locked generation with APA citations
       |
       v
  [python-docx] --> formatted .docx export with References chapter
```

### Stack

| Component       | Technology                          |
|-----------------|-------------------------------------|
| Embeddings      | Configurable: Jina v3 (local GPU), Ollama, or OpenAI |
| Image Captioning| Florence-2-large (local GPU, caption + OCR) |
| Vector DB       | LanceDB (local, serverless)         |
| PDF Parsing     | pdfplumber + pypdf                  |
| LLM (default)   | Ollama — any local model (auto-detected) |
| LLM (alt)       | LM Studio, Claude Sonnet/Opus, OpenAI GPT-4o, OpenRouter (Gemini, etc.) |
| Doc Export      | python-docx (configurable formatting) |
| Web UI          | Streamlit                           |
| Safety Scanner  | Regex + heuristic + LLM classifier (pluggable backends) |

### Project Structure

```
PaperMill/
├── config.py                  # Constants, paths, model config, startup validation
├── run_cli.py                 # Entry point: python run_cli.py
├── run_streamlit.py           # Entry point: python run_streamlit.py
├── requirements.txt           # Python dependencies
├── core/
│   ├── scanner.py             # Content safety scanner (regex/heuristic/LLM)
│   ├── embedder.py            # Embedding (Jina v3 / Ollama / OpenAI)
│   ├── db.py                  # LanceDB connection, schema, CRUD helpers
│   ├── ingestion.py           # PDF parse -> scan -> chunk -> embed -> store
│   ├── image_ingestion.py     # Florence-2 captioning + OCR -> embed -> store
│   ├── retrieval.py           # Vector similarity search with safety gate
│   ├── tree_index.py          # Tree-based document indexing via LLM (PageIndex)
│   ├── tree_retrieval.py      # Tree-based retrieval with LLM reasoning
│   ├── generation.py          # LLM generation: chat, draft, rewrite (multi-provider)
│   ├── prompts.py             # System prompt and templates (source-locked, anti-hallucination)
│   ├── paper_structure.py     # 37-section paper outline + JSON draft storage
│   ├── bibliography.py        # DOI lookup (CrossRef w/ retry), APA formatting
│   ├── docexport.py           # Markdown -> .docx with References chapter
│   └── versioning.py          # Snapshot versioning with delete
├── interfaces/
│   ├── cli.py                 # REPL with slash commands + persistent chat history
│   └── streamlit_app.py       # Web UI with runtime provider switching
├── data/
│   ├── pdfs/                  # Drop reference PDFs here
│   ├── images/                # Drop images here
│   ├── references.json        # Bibliography entries
│   ├── scanner_rules.json     # Scanner regex patterns (editable)
│   ├── scanner_allowlist.json # Allowlisted file hashes
│   └── tree_indexes/          # Per-PDF tree structure indexes (JSON)
├── reports/                   # Safety scan reports (Markdown)
├── quarantine/                # Blocked files
├── paper_sections/            # Section drafts stored as JSON
├── versions/                  # Timestamped .docx exports
│   └── manifest.json          # Version history metadata
└── tests/                     # Test suite
```

---

## Setup

### Prerequisites

- Python 3.11+
- GPU with CUDA support (recommended for embeddings and Florence-2; CPU fallback available)
- [Ollama](https://ollama.com/) running locally (default LLM provider), or an Anthropic API key for Claude

### Installation

```bash
git clone https://github.com/cr231521/PaperMill.git
cd PaperMill
pip install -r requirements.txt
```

### LLM Provider

**Ollama (default):**
```bash
ollama pull llama3.1
```

**Claude (alternative):**
Set the Anthropic API key as an environment variable:
```bash
export ANTHROPIC_API_KEY="sk-ant-api03-..."
```

The provider can be switched at runtime in the Streamlit sidebar.

---

## How to Use

### 1. Add Reference PDFs

Place your reference PDFs in the `data/pdfs/` folder, or upload them through the Streamlit UI (max 20 MB each, multi-file supported). These can include:

- Legislation and regulations
- Standards and guidelines
- Academic papers and journal articles
- Technical reports
- Any reference material relevant to your research

### 2. Launch the Interface

**Command Line (CLI):**

```bash
python run_cli.py
```

**Web UI (Streamlit):**

```bash
python run_streamlit.py
```

Startup validates configuration (directory writability, Ollama reachability, API key presence) and warns about any issues.

### 3. Ingest Your Sources

Before the assistant can use your references, they must be ingested (scanned, parsed, chunked, embedded, and stored).

**CLI:**
```
>>> /ingest                          # Ingest all PDFs in data/pdfs/
>>> /ingest path/to/specific.pdf     # Ingest a single PDF
>>> /ingest-images                   # Ingest all images in data/images/
>>> /sources                         # Verify what has been ingested
```

**Streamlit:** Use the sidebar file uploader, then click "Scan & ingest". Or click "Ingest all documents" / "Ingest all images" for bulk processing. Both buttons are incremental — already-ingested files are skipped automatically.

DOIs found in PDF metadata or first-page text are automatically extracted and added to the bibliography.

### 4. Choose a Retrieval Mode

PaperMill supports two retrieval modes. Toggle between them in the Streamlit chat tab or with `/mode` in the CLI.

**Vector Search (default):** Embedding similarity via LanceDB. Fast, cheap, works offline. Best for broad queries across all sources. Limitation: similarity isn't relevance — no structural understanding of documents.

**Tree Search:** An LLM builds a hierarchical table of contents for each document, then reasons over it at query time to select relevant sections. Precise, explainable, preserves document structure. Best for targeted questions against well-structured papers. Limitation: slower, costs API tokens, quality depends on model strength (cloud models produce better trees than local Ollama).

Tree indexes are built on-demand per document:
```
>>> /tree-build FAQs_on_CRA.pdf    # Build tree index for a specific PDF
>>> /tree-sources                   # List which PDFs have tree indexes
>>> /mode tree                      # Switch to tree retrieval mode
>>> /mode vector                    # Switch back to vector mode
```

Both modes return results in the same format — drafting, chat, and rewriting work identically regardless of which mode found the chunks.

### 5. Research and Ask Questions

Type any question in free text. The assistant searches your ingested references, retrieves the most relevant passages, and generates an answer with inline citations.

```
>>> What are the key findings from Smith et al. on this topic?
>>> How do the different frameworks compare in their approach?
>>> What gaps exist in the current literature?
```

Responses include citations in the format `[Source: filename, p. X]` and DOI references when bibliography entries are available. Chat history persists across restarts.

### 6. Manage Bibliography

```
>>> /ref-add 10.1000/example-doi     # Add reference by DOI (auto-fetches from CrossRef)
>>> /refs                             # List all bibliography entries
>>> /ref-remove key                   # Remove a reference
```

DOI format is validated before lookup. Transient network failures are retried automatically (3 attempts with exponential backoff).

### 7. Draft Paper Sections

The paper outline contains 37 sections covering the full structure of the paper. View the outline and draft sections one at a time.

```
>>> /outline                    # View all sections with status
>>> /draft ch2.1                # Draft a literature review subsection
>>> /show ch2.1                 # Read the generated draft
```

**Draft** generates a section from scratch: it retrieves the top 12 most relevant chunks from your ingested references, combines them with your bibliography, and sends everything to the **draft model** (e.g., Ollama, Sonnet, GPT-4o). The result is saved with status `draft`.

The system prompt enforces source-locked generation: every claim must be supported by retrieved chunks, no fabricated quotes, no decorative language. Findings are synthesized by theme with critical evaluation of conflicting sources.

### 8. Rewrite and Polish

Once a draft exists, rewrite it for improved quality using the polish model.

```
>>> /rewrite ch2.1              # Polish with Opus/polish model
>>> /show ch2.1                 # Review the improved version
```

**Rewrite** takes the existing draft text and sends it to the **polish model** (e.g., Opus, GPT-4o) along with the top 8 RAG chunks for additional context. The model improves clarity, flow, and academic rigour. The result is saved with status `review`.

### Draft vs Rewrite — Summary

| Action | What it does | Model used | Input | Output status |
|---|---|---|---|---|
| **Draft** | Generates section from scratch using RAG context | Draft model | Section title + top 12 chunks | `draft` |
| **Rewrite** | Improves existing text for clarity and rigour | Polish model | Current text + top 8 chunks | `review` |

**Safety features:**
- **Overwrite confirmation** — If a section already has content, both Draft and Rewrite show a warning and require explicit confirmation before proceeding (Streamlit UI)
- **Undo** — Every Draft or Rewrite automatically backs up the previous version. Click Undo to restore it (one level of undo per section)

### 9. Export to Word Document

Export individual sections or the full assembled paper as a formatted .docx file.

```
>>> /export ch2.1               # Export a single section
>>> /export                     # Export the full paper
```

Export formatting is configurable in the Paper & Export settings:
- Font (Times New Roman, Arial, Calibri, etc.)
- Font size (8–24pt)
- Line spacing (1.0–3.0)
- Margins (0.5–2.0 inches)
- Proper heading hierarchy and page breaks between chapters
- Auto-generated References chapter with APA hanging indent formatting

### 10. Version Control

Save snapshots of your paper at milestones and manage versions.

```
>>> /version literature-review-done    # Save a named snapshot
>>> /versions                          # List all saved versions
>>> /diff 1 2                          # Compare version 1 vs version 2
```

Versions are saved as timestamped .docx files in the `versions/` folder with metadata tracked in `manifest.json`. Versions can be downloaded or deleted from the Streamlit UI.

---

## Content Safety Scanner

The scanner protects the RAG pipeline against prompt injection and data exfiltration hidden in ingested documents.

### Defense Points

| Gate | When | Method |
|------|------|--------|
| **Ingestion (PDF)** | Before embedding | Structural scan + metadata scan + per-chunk content scan |
| **Ingestion (Image)** | Before embedding | OCR/caption divergence check + content scan |
| **Retrieval** | Before LLM context | Regex-only scan on retrieved chunks |
| **Chat input** | Before query | Regex-only scan with chat scope (warnings, not hard block) |

### How It Works

1. **RegexBackend** — 15 patterns detecting role hijacking, system prompt leaks, delimiter injection, instruction overrides, and exfiltration attempts. Supports evasion-resistant "super-cleaned" matching for spaced-out text (`i g n o r e` -> `ignore`)
2. **HeuristicBackend** — Scores text on imperative command density and second-person pronoun density (0.0-1.0 suspicion score)
3. **OllamaClassifierBackend** — LLM-based classifier, triggered when heuristic score exceeds threshold. Fail-closed: blocks on parse errors or connection failures

Blocked files are quarantined and a Markdown report is generated. Files can be released from quarantine via CLI or Streamlit UI. Reports and quarantine can be bulk-cleared from the Streamlit dashboard.

### Scanner Commands

```
>>> /scan-rules              # List active scanner rules
>>> /allowlist-add <path>    # Add file to scanner allowlist
>>> /allowlist-remove <hash> # Remove hash from allowlist
>>> /quarantine-release      # List/release quarantined files
```

---

## CLI Command Reference

| Command                | Description                                      |
|------------------------|--------------------------------------------------|
| `/ingest [path]`       | Ingest all PDFs or a specific file               |
| `/ingest-images [path]`| Ingest images (Florence-2 caption + OCR)         |
| `/sources`             | List all ingested sources (shows [tree] tag)     |
| `/delete-source <name>`| Remove a source and its chunks from the database |
| `/mode [vector\|tree]` | Show or set retrieval mode                       |
| `/tree-build <name>`   | Build tree index for a source (on-demand)        |
| `/tree-delete <name>`  | Delete tree index for a source                   |
| `/tree-sources`        | List sources with tree indexes                   |
| `/outline`             | Show paper outline with draft status per section |
| `/draft <id>`          | Draft a section (e.g., ch1, ch2.1, abstract)     |
| `/rewrite <id>`        | Rewrite/polish a section using the polish model  |
| `/show <id>`           | Display a section's current draft                |
| `/export [id]`         | Export full paper or a single section to .docx   |
| `/version [label]`     | Save a versioned snapshot with optional label    |
| `/versions`            | List all saved version snapshots                 |
| `/diff <v1> <v2>`      | Compare two versions side by side                |
| `/ref-add <DOI>`       | Add a reference by DOI (auto-fetches metadata)   |
| `/refs`                | List all bibliography entries                    |
| `/ref-remove <key>`    | Remove a reference                               |
| `/scan-rules`          | List active scanner rules                        |
| `/allowlist-add <path>`| Add file to scanner allowlist                    |
| `/allowlist-remove <h>`| Remove hash from allowlist                       |
| `/quarantine-release`  | List/release quarantined files                   |
| `/clear-history`       | Clear persistent chat history                    |
| `/help`                | Show command reference                           |
| `/quit`                | Exit the assistant                               |

Free-text input (without `/`) is treated as a research question and processed through the RAG pipeline using the current retrieval mode.

---

## Streamlit Web UI

The web interface provides the same functionality with a visual layout:

- **Sidebar:**
  - LLM provider switcher (Ollama / LM Studio / Claude / OpenAI / OpenRouter) — takes effect immediately
  - Embedding provider/model selector (local / Ollama / OpenAI) with auto re-ingest on change
  - **Write section:**
    - Paper title editing with inline save
    - Paper Sections manager (add, remove, reorder, reset)
    - Section selector with status icons, Draft / Rewrite / Undo buttons
  - Paper & Export settings (font, size, spacing, margins)
  - Export & Versioning with download and delete
  - Bibliography management (DOI add, APA display)
  - Multi-file PDF upload (max 20 MB each) with batch ingestion
  - "Ingest all documents" and "Ingest all images" buttons (incremental)
  - Image upload with single-file ingestion
  - Ingested sources list with delete buttons and tree index build/delete controls
  - Content scanner dashboard (rules, reports with clear, quarantine with clear)

- **Main Area:**
  - **Chat tab:** Conversational research interface with retrieval mode toggle (Vector/Tree Search), source citations, persistent history, and clear history button
  - **Section Viewer tab:** Read and review drafted sections with inline editor, word count, and citation audit

---

## Paper Outline

The assistant includes a pre-configured 37-section outline (customisable via the Streamlit sidebar or `core/paper_structure.py`):

| Chapter | Title | Target Words |
|---------|-------|-------------|
| Abstract | Abstract | 300 |
| Ch 1 | Introduction (5 subsections) | 2,500 |
| Ch 2 | Literature Review (4 subsections) | 5,000 |
| Ch 3 | Methodology (4 subsections) | 3,500 |
| Ch 4 | Domain Analysis (2 subsections) | 3,000 |
| Ch 5 | Toolkit Design and Development (3 subsections) | 4,000 |
| Ch 6 | Evaluation and Results (3 subsections) | 3,500 |
| Ch 7 | Discussion (3 subsections) | 2,500 |
| Ch 8 | Conclusion | 1,500 |
| | References + Appendices | — |

The outline can be customised via the **Write > Paper Sections** expander in the Streamlit sidebar. Changes are persisted to `data/settings.json`.

---

## Security

- **Content safety scanner** with defense-in-depth (regex + heuristic + LLM) at ingestion, retrieval, and chat input
- **SQL injection protection** on all LanceDB WHERE clauses via escape helper
- **Prompt injection defense** — chunk text fenced in code blocks within prompt templates
- **Token overflow protection** — automatic truncation when context exceeds model limits
- **DOI validation** before CrossRef API calls
- **Fail-closed LLM classifier** — blocks on parse/connection failures
- **Startup validation** — checks directory writability, Ollama reachability, API key presence
- **Upload size limit** — 20 MB per PDF to prevent resource exhaustion

---

## How It Works — Under the Hood

### PDF Chunking Strategy

The ingestion pipeline uses a hybrid heading-aware + fixed-size approach:

1. **Scan** — Content safety scanner checks PDF structure, metadata, and per-chunk text (regex + heuristic + LLM escalation)
2. **Parse** — pdfplumber extracts text with font-size and style metadata per text span
3. **Detect headings** — Heuristics identify section boundaries using font size (>115% of median), bold text, ALL CAPS, and numbered patterns (e.g., "2.1 Overview")
4. **Chunk within sections** — Text is split into ~800-token chunks with 200-token overlap, never crossing section boundaries
5. **Self-describing chunks** — Each chunk is prefixed with its section heading for better retrieval context
6. **DOI extraction** — DOIs found in PDF metadata or first-page text are auto-added to bibliography

### Retrieval Modes

**Vector Search (default):**
- Queries are embedded using Jina v3's `retrieval.query` task head (or configured embedding provider)
- Passages are embedded using the `retrieval.passage` task head
- Search returns the top-k most similar chunks (default: 8)
- Optional source filtering restricts results to a specific PDF
- Retrieval-time safety gate re-scans chunks (regex-only) from sources not yet verified with current rules

**Tree Search (on-demand):**
- Adapted from [VectifyAI/PageIndex](https://github.com/VectifyAI/PageIndex) (MIT License)
- Build a hierarchical tree index per document via LLM (detects TOC or generates structure)
- At query time, a single LLM call reasons over the tree to select relevant sections
- Page text is fetched directly via pdfplumber — no embedding needed
- Same safety gate as vector search (regex-only scan on retrieved content)
- Returns results in the same format as vector search — generation pipeline works unchanged
- Best with strong models (Claude, GPT-4o); local Ollama models produce coarser trees

### Generation

- Retrieved chunks are formatted with source metadata, fenced in code blocks, and injected into prompt templates
- The system prompt enforces source-locked generation: every claim must trace to a provided chunk
- Anti-hallucination constraints: no fabricated quotes, no decorative language, explicit gap statements
- Bibliography with DOIs is included in all prompts (chat, draft, rewrite)
- Dynamic provider switching — Ollama, LM Studio, Claude, OpenAI, or OpenRouter can be toggled at runtime
- Token count estimation prevents context overflow with automatic truncation

### Section Storage

Each drafted section is saved as a JSON file in `paper_sections/` with:
- Section ID, title, and full Markdown text
- Status tracking: `empty` -> `draft` -> `review` -> `final`
- Timestamp of last update

---

## Testing

```bash
python -m pytest tests/ -v
```

---

## Tips for Best Results

1. **More references = better output.** Ingest as many relevant PDFs as you can — the RAG pipeline selects the most relevant passages automatically.

2. **Be specific with questions.** Instead of "tell me about topic X", ask "What are the specific requirements described in Section 3 of Document Y?"

3. **Iterate on sections.** Draft first with `/draft`, review with `/show`, then `/rewrite` with specific instructions for improvement.

4. **Save versions at milestones.** Use `/version` after completing each chapter so you can track progress and compare changes.

5. **Add DOIs early.** Use `/ref-add` to build your bibliography — the assistant will use proper APA citations with DOIs in all generated text.

6. **Customise the outline.** Use the Write > Paper Sections expander in the Streamlit sidebar to add, remove, or reorder sections.

7. **Use Undo after Draft/Rewrite.** If a Draft or Rewrite produces poor results, click Undo to restore the previous version instantly.
