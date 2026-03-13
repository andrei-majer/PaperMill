# PaperMill

PaperMill with AI Co-Authoring features and a local-first RAG pipeline (built-in content safety scanning against prompt injection and data exfiltration).

## Overview

AI-powered academic writing assistant with a local-first RAG pipeline and defense-in-depth content safety scanning against prompt injection and data exfiltration.

The system ingests reference PDFs and images, stores them as vector embeddings in a local database, and uses retrieval-augmented generation (RAG) to draft, rewrite, and answer questions about paper sections — all grounded in your actual source material.

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
  [PyMuPDF Parser / Florence-2] --> heading-aware chunking / caption+OCR
       |
       v
  [Jina Embeddings v3] --> 1024-dim vectors (GPU-accelerated)
       |
       v
  [LanceDB] --> local vector database (no server needed)
       |
       v
  [Vector Search + Safety Gate] --> top-k relevant chunks (regex re-scan)
       |
       v
  [Ollama / Claude API] --> source-locked generation with APA citations
       |
       v
  [python-docx] --> formatted .docx export with References chapter
```

### Stack

| Component       | Technology                          |
|-----------------|-------------------------------------|
| Embeddings      | jinaai/jina-embeddings-v3 (local, GPU, 1024-dim, 8192 token context) |
| Image Captioning| Florence-2-large (local GPU, caption + OCR) |
| Vector DB       | LanceDB (local, serverless)         |
| PDF Parsing     | PyMuPDF (fitz)                      |
| LLM (default)   | Ollama — any local model (e.g. dolphin-llama3) |
| LLM (alt)       | Claude Sonnet / Opus via Anthropic API |
| Doc Export      | python-docx                         |
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
│   ├── embedder.py            # Jina v3 model loading + embed functions
│   ├── db.py                  # LanceDB connection, schema, CRUD helpers
│   ├── ingestion.py           # PDF parse -> scan -> chunk -> embed -> store
│   ├── image_ingestion.py     # Florence-2 captioning + OCR -> embed -> store
│   ├── retrieval.py           # Vector similarity search with safety gate
│   ├── generation.py          # LLM generation: chat, draft, rewrite (Ollama/Claude)
│   ├── prompts.py             # System prompt and templates (source-locked, anti-hallucination)
│   ├── paper_structure.py     # 41-section paper outline + JSON draft storage
│   ├── bibliography.py        # DOI lookup (CrossRef w/ retry), APA formatting
│   ├── docexport.py           # Markdown -> .docx with References chapter
│   └── versioning.py          # Timestamped snapshots + manifest tracking
├── interfaces/
│   ├── cli.py                 # REPL with slash commands + persistent chat history
│   └── streamlit_app.py       # Web UI with runtime provider switching
├── data/
│   ├── pdfs/                  # Drop reference PDFs here
│   ├── images/                # Drop images here
│   ├── references.json        # Bibliography entries
│   ├── scanner_rules.json     # Scanner regex patterns (editable)
│   └── scanner_allowlist.json # Allowlisted file hashes
├── reports/                   # Safety scan reports (Markdown)
├── quarantine/                # Blocked files
├── paper_sections/            # Section drafts stored as JSON
├── versions/                  # Timestamped .docx exports
│   └── manifest.json          # Version history metadata
└── tests/                     # 50 tests
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
ollama pull dolphin-llama3
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

Place your reference PDFs in the `data/pdfs/` folder. These can include:

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

**Streamlit:** Use the sidebar file uploader, then click "Ingest".

DOIs found in PDF metadata or first-page text are automatically extracted and added to the bibliography.

### 4. Research and Ask Questions

Type any question in free text. The assistant searches your ingested references, retrieves the most relevant passages, and generates an answer with inline citations.

```
>>> What are the key findings from Smith et al. on this topic?
>>> How do the different frameworks compare in their approach?
>>> What gaps exist in the current literature?
```

Responses include citations in the format `[Source: filename, p. X]` and DOI references when bibliography entries are available. Chat history persists across restarts.

### 5. Manage Bibliography

```
>>> /ref-add 10.1000/example-doi     # Add reference by DOI (auto-fetches from CrossRef)
>>> /refs                             # List all bibliography entries
>>> /ref-remove key                   # Remove a reference
```

DOI format is validated before lookup. Transient network failures are retried automatically (3 attempts with exponential backoff).

### 6. Draft Paper Sections

The paper outline contains 41 sections covering the full structure of the thesis. View the outline and draft sections one at a time.

```
>>> /outline                    # View all sections with status
>>> /draft ch2.1                # Draft a literature review subsection
>>> /show ch2.1                 # Read the generated draft
```

**Draft** generates a section from scratch: it retrieves the top 12 most relevant chunks from your ingested references, combines them with your bibliography, and sends everything to the **draft model** (e.g., Ollama, Sonnet, GPT-4o). The result is saved with status `draft`.

The system prompt enforces source-locked generation: every claim must be supported by retrieved chunks, no fabricated quotes, no decorative language. Findings are synthesized by theme with critical evaluation of conflicting sources.

### 7. Rewrite and Polish

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

### 8. Export to Word Document

Export individual sections or the full assembled paper as a formatted .docx file.

```
>>> /export ch2.1               # Export a single section
>>> /export                     # Export the full paper
```

The exported document uses:
- Times New Roman, 12pt font
- 1.5 line spacing
- 1-inch margins
- Proper heading hierarchy
- Page breaks between chapters
- Auto-generated References chapter with APA hanging indent formatting

### 9. Version Control

Save snapshots of your paper at milestones and compare versions.

```
>>> /version literature-review-done    # Save a named snapshot
>>> /versions                          # List all saved versions
>>> /diff 1 2                          # Compare version 1 vs version 2
```

Versions are saved as timestamped .docx files in the `versions/` folder with metadata tracked in `manifest.json`.

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

Blocked files are quarantined and a Markdown report is generated. Files can be released from quarantine via CLI or Streamlit UI.

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
| `/sources`             | List all ingested sources                        |
| `/delete-source <name>`| Remove a source and its chunks from the database |
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

Free-text input (without `/`) is treated as a research question and processed through the RAG pipeline.

---

## Streamlit Web UI

The web interface provides the same functionality with a visual layout:

- **Sidebar:**
  - LLM provider switcher (Ollama / Claude) — takes effect immediately
  - PDF and image upload with ingestion
  - List of ingested sources (with delete buttons)
  - Paper sections dropdown showing draft status
  - Draft, Rewrite, and Undo buttons per section (with overwrite confirmation)
  - Export to .docx with download
  - Version saving and download links
  - Bibliography management
  - Content scanner dashboard (rules, reports, quarantine)

- **Main Area:**
  - **Chat tab:** Conversational research interface with source citations, persistent history, and clear history button
  - **Section Viewer tab:** Read and review drafted sections

---

## Paper Outline

The assistant includes a pre-configured 41-section outline (customisable in `core/paper_structure.py`):

| Chapter | Title | Target Words |
|---------|-------|-------------|
| Abstract | Abstract | 300 |
| Ch 1 | Introduction (5 subsections) | 2,500 |
| Ch 2 | Literature Review (5 subsections) | 5,000 |
| Ch 3 | Methodology (5 subsections) | 3,500 |
| Ch 4 | Domain Analysis (3 subsections) | 3,000 |
| Ch 5 | Toolkit Design and Development (4 subsections) | 4,000 |
| Ch 6 | Evaluation and Results (4 subsections) | 3,500 |
| Ch 7 | Discussion (4 subsections) | 2,500 |
| Ch 8 | Conclusion | 1,500 |
| | References + Appendices | — |

The outline can be customised via the **Settings > Paper Sections Manager** in the Streamlit sidebar, or by editing `core/paper_structure.py` directly. Changes are persisted to `data/settings.json`.

---

## Security

- **Content safety scanner** with defense-in-depth (regex + heuristic + LLM) at ingestion, retrieval, and chat input
- **SQL injection protection** on all LanceDB WHERE clauses via escape helper
- **Prompt injection defense** — chunk text fenced in code blocks within prompt templates
- **Token overflow protection** — automatic truncation when context exceeds model limits
- **DOI validation** before CrossRef API calls
- **Fail-closed LLM classifier** — blocks on parse/connection failures
- **Startup validation** — checks directory writability, Ollama reachability, API key presence

---

## How It Works — Under the Hood

### PDF Chunking Strategy

The ingestion pipeline uses a hybrid heading-aware + fixed-size approach:

1. **Scan** — Content safety scanner checks PDF structure, metadata, and per-chunk text (regex + heuristic + LLM escalation)
2. **Parse** — PyMuPDF extracts text with font-size and style metadata per text span
3. **Detect headings** — Heuristics identify section boundaries using font size (>115% of median), bold text, ALL CAPS, and numbered patterns (e.g., "2.1 Overview")
4. **Chunk within sections** — Text is split into ~800-token chunks with 200-token overlap, never crossing section boundaries
5. **Self-describing chunks** — Each chunk is prefixed with its section heading for better retrieval context
6. **DOI extraction** — DOIs found in PDF metadata or first-page text are auto-added to bibliography

### Vector Search

- Queries are embedded using Jina v3's `retrieval.query` task head
- Passages are embedded using the `retrieval.passage` task head
- Search returns the top-k most similar chunks (default: 8)
- Optional source filtering restricts results to a specific PDF
- Retrieval-time safety gate re-scans chunks (regex-only) from sources not yet verified with current rules

### Generation

- Retrieved chunks are formatted with source metadata, fenced in code blocks, and injected into prompt templates
- The system prompt enforces source-locked generation: every claim must trace to a provided chunk
- Anti-hallucination constraints: no fabricated quotes, no decorative language, explicit gap statements
- Bibliography with DOIs is included in all prompts (chat, draft, rewrite)
- Dynamic provider switching — Ollama or Claude can be toggled at runtime
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

50 tests covering: data structures, text normalization, regex matching, heuristic scoring, LLM classifier (mocked), pipeline orchestration, PDF structural/metadata scanning, report generation, scan history, quarantine, OCR divergence, ingestion integration, retrieval gate, chat input scanning, and database safety operations.

---

## Tips for Best Results

1. **More references = better output.** Ingest as many relevant PDFs as you can — the RAG pipeline selects the most relevant passages automatically.

2. **Be specific with questions.** Instead of "tell me about topic X", ask "What are the specific requirements described in Section 3 of Document Y?"

3. **Iterate on sections.** Draft first with `/draft`, review with `/show`, then `/rewrite` with specific instructions for improvement.

4. **Save versions at milestones.** Use `/version` after completing each chapter so you can track progress and compare changes.

5. **Add DOIs early.** Use `/ref-add` to build your bibliography — the assistant will use proper APA citations with DOIs in all generated text.

6. **Customise the outline.** Use the Settings panel in the Streamlit sidebar to add, remove, or reorder sections — or edit `core/paper_structure.py` directly.

7. **Use Undo after Draft/Rewrite.** If a Draft or Rewrite produces poor results, click Undo to restore the previous version instantly.
