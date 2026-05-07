<div align="center">

# 📝 PaperMill

### AI Co-Authoring for Academic Research Papers

**Built-in content safety scanning against prompt injection and data exfiltration**

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io/)

</div>

---

## 🔍 The Problem

Writing academic papers requires synthesizing dozens of reference sources into structured, citation-grounded prose. Manual literature reviews are slow. Existing AI tools either hallucinate freely or can't ground their output in your actual sources.

## 💡 The Solution

PaperMill ingests your reference PDFs and images, stores them as vector embeddings in a local database, and uses **retrieval-augmented generation (RAG)** to draft, rewrite, and answer questions about paper sections — all grounded in your actual source material.

Every claim is **source-locked** to ingested references. Conflicting sources are flagged for critical evaluation. Gaps are stated explicitly rather than filled with fabrication. The output follows scholarly conventions — APA citations, formal academic register, thematic synthesis, and a configurable paper outline from Abstract through Conclusion.

> **PaperMill does not conduct original research or generate findings.** It is a writing tool: the researcher provides the sources, the research questions, and validates the output.

---

## 🌲 Dual Retrieval: Vector Search + Tree Search

PaperMill offers two retrieval modes, toggled in the UI or CLI:

| | Vector Search (default) | Tree Search |
|---|---|---|
| **How** | Embedding similarity via LanceDB | LLM reasons over hierarchical document structure |
| **Speed** | Fast (local, no LLM calls) | Slower (LLM calls at index + query time) |
| **Cost** | Free with local embeddings | API tokens per query |
| **Best for** | Broad search across all sources | Precise questions on structured papers |
| **Offline** | ✅ Works with Ollama | ⚠️ Quality depends on model strength |

**Vector Search** chunks your PDFs into ~800-token overlapping blocks and embeds them. At query time, your question is embedded and LanceDB finds the most similar chunks via cosine distance.

**Tree Search** builds a hierarchical table of contents per document — sections, subsections, page ranges, and summaries. At query time, the LLM reasons over this structure to select the most relevant sections, then fetches their full page text directly. Adapted from [VectifyAI/PageIndex](https://github.com/VectifyAI/PageIndex) (MIT License).

Both modes return results in the same format — the generation pipeline works identically regardless of which mode found the chunks.

---

## ⚡ Core Features

- **📄 PDF & Image Ingestion** — Heading-aware chunking (pdfplumber), Florence-2 caption + OCR, configurable embedding providers (Jina v3 local, Ollama, OpenAI), auto DOI extraction
- **🌲 Dual Retrieval** — Vector search or Tree search, toggle in UI or CLI
- **💬 RAG Chat** — Ask questions with source citations and persistent chat history
- **✍️ Section Drafting & Rewriting** — Draft from scratch with RAG context, polish with a separate model, overwrite confirmation and one-step undo
- **📚 Bibliography** — Auto-fetch metadata from CrossRef by DOI, APA formatting
- **📊 Citation Audit** — Flag unresolved placeholders and missing bibliography entries per section
- **📎 Export** — .docx with configurable academic formatting, page breaks, auto-generated References chapter with APA hanging indent
- **🛡️ Content Safety Scanner** — Defense-in-depth against prompt injection and exfiltration (regex + heuristic + LLM classifier)
- **🔄 5 LLM Providers** — Ollama, LM Studio, Claude, OpenAI, OpenRouter — switch at runtime. Any OpenAI-compatible API (e.g. DeepSeek V4) works via the `openai` provider by setting `OPENAI_BASE_URL`
- **🖥️ Dual Interface** — CLI REPL and Streamlit web UI

---

## 🏗️ Architecture

```
  Reference PDFs / Images
       │
       ▼
  [Content Safety Scanner] ──→ regex + heuristic + LLM classifier
       │  (block or pass)
       ▼
  [pdfplumber / Florence-2] ──→ heading-aware chunking / caption+OCR
       │
       ▼
  [Embeddings (Jina v3 / Ollama / OpenAI)] ──→ configurable vectors
       │
       ▼
  [LanceDB] ──→ local vector database (no server needed)
       │
       ├── Vector Search ──→ top-k similar chunks (cosine distance)
       │        OR
       └── Tree Search ───→ LLM reasons over document structure index
                │
                ▼
  [Ollama / Claude / OpenAI / OpenRouter] ──→ source-locked generation
       │
       ▼
  [python-docx] ──→ formatted .docx with References chapter
```

---

## 🚀 Quick Start

### 1. Install

```bash
git clone https://github.com/andrei-majer/PaperMill.git
cd PaperMill
pip install -r requirements.txt
```

### 2. Pull an Ollama model

```bash
ollama pull llama3.1
ollama pull gemma4:e4b
```

### 3. Run

```bash
python run_streamlit.py    # Web UI → http://localhost:8501
python run_cli.py          # CLI mode
```

<details>
<summary><b>Prerequisites</b></summary>

- Python 3.11+
- GPU for local embeddings and Florence-2 (CUDA on Windows/Linux, Metal on Apple Silicon)
- [Ollama](https://ollama.com/) running locally (default LLM provider)

</details>

<details>
<summary><b>Apple Silicon (Mac Mini M3)</b></summary>

Runs without code changes. Ollama and LM Studio both have native ARM builds with Metal acceleration.

| Config | RAM estimate |
|---|---|
| OS + Streamlit + LanceDB | ~2–3 GB |
| Jina v3 embeddings (local) | ~600 MB |
| Ollama llama3.1:8b Q4 | ~4.7 GB |
| Ollama gemma4:e4b Q4 | ~7–8 GB |

On 16 GB unified memory, `llama3.1:8b` (draft) + `gemma4:e4b` (polish) is workable but tight. For comfortable headroom, use a cloud provider (Claude, OpenAI, OpenRouter) which requires no local model RAM.

</details>

---

## 📖 Usage

### CLI Commands

```
/ingest [path]           Ingest a PDF (default: all in data/pdfs/)
/ingest-images [path]    Ingest images (default: all in data/images/)
/sources                 List ingested sources (shows [tree] tag)
/delete-source <name>    Delete a source from the vector DB
/mode [vector|tree]      Show or set retrieval mode
/tree-build <name>       Build tree index for a source (on-demand)
/tree-delete <name>      Delete tree index for a source
/tree-sources            List sources with tree indexes
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

Free text input triggers RAG-powered chat using the current retrieval mode.

### Draft vs Rewrite

| Action | What it does | Model | Input | Output |
|---|---|---|---|---|
| **Draft** | Generate from scratch using RAG context | Draft (Sonnet, GPT-4o, Ollama) | Section title + top 12 chunks | `draft` |
| **Rewrite** | Improve clarity, flow, and academic rigour | Polish (Opus, GPT-4o, Ollama) | Current text + top 8 chunks | `review` |

Both include **overwrite confirmation** and **one-step undo**.

### Streamlit UI

The web interface provides:

- **Sidebar:** LLM provider switcher, embedding settings, paper sections manager, export settings, bibliography, source management with tree index controls, content scanner dashboard
- **Chat tab:** Research chat with retrieval mode toggle (Vector/Tree), source citations, persistent history
- **Section Viewer:** Read/edit drafted sections, word count progress, citation audit, single section export

---

## 🛡️ Content Safety Scanner

Defense-in-depth against prompt injection and data exfiltration in ingested documents.

| Gate | When | Method |
|---|---|---|
| **Ingestion (PDF)** | Before embedding | Structure + metadata + per-chunk content scan |
| **Ingestion (Image)** | Before embedding | OCR/caption divergence check + content scan |
| **Retrieval** | Before LLM context | Regex-only scan on retrieved chunks |
| **Chat input** | Before query | Regex-only scan (warnings, not hard block) |

**Three backends:** RegexBackend (15 evasion-resistant patterns), HeuristicBackend (imperative/pronoun density scoring), OllamaClassifierBackend (LLM escalation, fail-closed). Pluggable via `ScannerBackend` base class.

<details>
<summary><b>Scanner rules example</b></summary>

Rules stored in `data/scanner_rules.json` — inspectable and editable:

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

</details>

---

## 🗂️ Stack

| Component | Technology |
|---|---|
| Embeddings | [Jina v3](https://huggingface.co/jinaai/jina-embeddings-v3) (local GPU), Ollama, or OpenAI |
| Image Captioning | [Florence-2-large](https://huggingface.co/microsoft/Florence-2-large) (local GPU) |
| Vector DB | [LanceDB](https://lancedb.com/) (local, serverless) |
| PDF Parsing | [pdfplumber](https://github.com/jsvine/pdfplumber) + [pypdf](https://github.com/py-pdf/pypdf) |
| LLM | [Ollama](https://ollama.com/), [LM Studio](https://lmstudio.ai/), Claude, OpenAI, OpenRouter |
| Doc Export | python-docx |
| Web UI | [Streamlit](https://streamlit.io/) |
| Safety Scanner | Regex + heuristic + LLM classifier |

---

## 📁 Project Structure

```
PaperMill/
├── config.py                  # All configuration constants
├── run_cli.py                 # CLI entry point
├── run_streamlit.py           # Streamlit entry point
├── core/
│   ├── scanner.py             # Content safety scanner
│   ├── ingestion.py           # PDF ingestion pipeline
│   ├── image_ingestion.py     # Image ingestion (Florence-2)
│   ├── retrieval.py           # Vector search with safety gate
│   ├── tree_index.py          # Tree-based document indexing (PageIndex)
│   ├── tree_retrieval.py      # Tree-based retrieval with LLM reasoning
│   ├── generation.py          # LLM generation (chat, draft, rewrite)
│   ├── embedder.py            # Embedding (Jina v3 / Ollama / OpenAI)
│   ├── db.py                  # LanceDB schema and helpers
│   ├── bibliography.py        # DOI lookup, APA formatting
│   ├── prompts.py             # System prompt and templates
│   ├── paper_structure.py     # Paper outline management
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
│   ├── scanner_allowlist.json # Allowlisted file hashes
│   └── tree_indexes/          # Per-PDF tree structure indexes
├── reports/                   # Safety scan reports
├── quarantine/                # Blocked files
├── paper_sections/            # Section drafts (JSON)
├── versions/                  # Versioned .docx snapshots
└── tests/                     # Test suite
```

---

## ⚙️ Configuration

<details>
<summary><b>Key settings in config.py</b></summary>

| Setting | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `ollama` | `ollama`, `lmstudio`, `claude`, `openai`, or `openrouter` |
| `OLLAMA_DRAFT_MODEL` | `llama3.1:latest` | Ollama model for drafting and chat |
| `OLLAMA_POLISH_MODEL` | `gemma4:e4b` | Ollama model for rewriting |
| `LMSTUDIO_URL` | `http://localhost:1234/v1` | LM Studio API endpoint |
| `OPENAI_BASE_URL` | *(empty)* | Override OpenAI base URL — point at any OpenAI-compatible API (e.g. `https://api.deepseek.com/v1` for DeepSeek V4; same key/model vars, no code changes) |
| `OPENAI_DRAFT_MODEL` | `gpt-5.4-mini` | OpenAI model for drafting (e.g. `deepseek-v4-flash`) |
| `OPENROUTER_DRAFT_MODEL` | `google/gemini-2.5-flash-preview` | OpenRouter model for drafting |
| `EMBEDDING_PROVIDER` | `local` | `local`, `ollama`, or `openai` |
| `SCANNER_LLM_ESCALATION` | `True` | Enable LLM classifier escalation |
| `SCANNER_SUSPICION_THRESHOLD` | `0.6` | Heuristic score to trigger LLM |
| `SCANNER_DRY_RUN` | `False` | Log threats without blocking |
| `MAX_CONTEXT_TOKENS` | `180000` | Max tokens before prompt truncation |
| `PAPER_TITLE` | *(your paper title)* | Title used in prompts and .docx export |

</details>

---

## 🧪 Testing

```bash
python -m pytest tests/ -v
```

---

## 🔒 Security

- SQL injection protection on all LanceDB WHERE clauses
- Chunk text escaped in prompt templates (fenced code blocks)
- Token count estimation with automatic truncation
- DOI format validation before CrossRef API calls
- Fail-closed LLM classifier — blocks on parse/connection failures
- Upload size limit — 20 MB per PDF

---

## 🤖 AI Acknowledgment

This project was co-developed with **Claude (Anthropic)** for code generation and review. At runtime, the tool uses **Ollama / Claude** for RAG-powered drafting and content safety classification. All code was reviewed, tested, and validated by the author.

**Suggested paper disclosure:**

> Portions of this manuscript were drafted using PaperMill, a RAG tool that synthesizes content from ingested references using [Ollama/Claude]. All AI-assisted drafts were verified against primary sources and revised by the author. The AI did not generate original findings or conclusions. Source code: https://github.com/andrei-majer/PaperMill.

---

## ⭐ Support

Leave a star 🌟 if you find PaperMill useful for your research!

### Citation

If you use PaperMill in your work, please cite:

```
Andrei Majer, "PaperMill: AI Co-Authoring for Academic Research Papers", 2026.
https://github.com/andrei-majer/PaperMill
```

```bibtex
@software{PaperMill,
  author = {Andrei Majer},
  title = {PaperMill: AI Co-Authoring for Academic Research Papers},
  year = {2026},
  url = {https://github.com/andrei-majer/PaperMill},
  license = {CC-BY-NC-4.0}
}
```

---

## 📄 License

[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) — Free for academic and personal use with attribution. Commercial use requires permission.

---

<div align="center">

© 2026 Andrei Majer

[![GitHub](https://img.shields.io/badge/GitHub-andrei--majer-181717?logo=github)](https://github.com/andrei-majer/PaperMill) [![LinkedIn](https://img.shields.io/badge/LinkedIn-Andrei%20Majer-0A66C2?logo=linkedin)](https://www.linkedin.com/in/andrei-majer/)

</div>
