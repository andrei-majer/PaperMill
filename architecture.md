# PaperMill Architecture

## How PaperMill works with Claude + Local LanceDB

```
Your PDFs/Images
       │
       ▼
 ┌─────────────────────┐
 │  Content Scanner     │  ← runs locally (regex + heuristic + Ollama LLM)
 │  (block or pass)     │
 └────────┬────────────┘
          ▼
 ┌─────────────────────┐
 │  Jina Embeddings v3  │  ← runs locally on your GPU
 │  (1024-dim vectors)  │     no API calls, no data leaves your machine
 └────────┬────────────┘
          ▼
 ┌─────────────────────┐
 │  LanceDB (local)     │  ← serverless, file-based DB at ~/.claude/lancedb/paper-assistant/
 │  stores all vectors   │     no network, no cloud — just files on disk
 └────────┬────────────┘
          │
    ══════╪══════════  you ask a question  ══════════
          │
          ├──── Vector Mode ────┐     ├──── Tree Mode ──────┐
          ▼                     │     ▼                     │
 ┌─────────────────────┐       │  ┌──────────────────────┐ │
 │  Vector Search       │       │  │  Tree Search          │ │
 │  top-8 similar chunks│       │  │  LLM reasons over     │ │
 │  (embedding cosine)  │       │  │  document structure    │ │
 └────────┬────────────┘       │  └────────┬─────────────┘ │
          │                     │           │               │
          └─────────────────────┘           │               │
          │  ← user picks mode              │               │
          ├─────────────────────────────────┘               │
          │                                                 │
          │  chunks + system prompt + bibliography
          ▼
 ┌─────────────────────┐
 │  LLM Provider ☁️     │  ← ONLY this step goes to the cloud (unless using Ollama)
 │  Claude / OpenAI /   │     sends: system prompt + retrieved chunks + your question
 │  OpenRouter / Ollama  │     receives: source-locked response with citations
 └────────┬────────────┘
          ▼
     Your answer with [Source: file, pp. X-Y] citations
```

## What stays local (never leaves your machine)

- All PDFs and images
- All embeddings and vector search
- LanceDB database (`~/.claude/lancedb/paper-assistant/`)
- Content safety scanning
- Tree indexes (`data/tree_indexes/`)
- Section drafts, versions, bibliography

## What goes to the cloud API (Claude / OpenAI / OpenRouter)

- The system prompt (academic writing instructions)
- The top-8 retrieved text chunks relevant to your query
- Your question
- Your bibliography entries

The cloud API only sees the *relevant snippets* from your documents (not the full PDFs), and only when you actively ask a question or draft a section. The embedding, indexing, search, and storage are entirely local.

When using **Ollama**, everything stays local — no data leaves your machine at all.

## LLM Providers

| Provider | Draft Model | Polish Model | Data Location |
|---|---|---|---|
| Ollama (default) | llama3.1:latest | gemma4:e4b | 100% local |
| Claude | claude-sonnet-4-6 | claude-opus-4-7 | Chunks sent to Anthropic API |
| OpenAI | gpt-5.4-mini | gpt-5.5 | Chunks sent to OpenAI API |
| OpenRouter | gemini-2.5-flash-preview | gemini-2.5-pro-preview | Chunks sent via OpenRouter API |

## Data Flow per Operation

### Ingestion (100% local)
```
PDF → Content Scanner → pdfplumber parse → heading-aware chunking (800 tok, 200 overlap)
    → Jina v3 embed (GPU) → LanceDB store
    → DOI auto-extract → CrossRef metadata fetch → bibliography
```

### Image Ingestion (100% local)
```
Image → Florence-2 caption + OCR (GPU) → OCR divergence check
      → Content Scanner → Jina v3 embed → LanceDB store
```

### Chat / Draft / Rewrite — Vector Mode (local search + cloud generation)
```
Query → Jina v3 embed query (local GPU)
      → LanceDB vector search top-k (local)
      → Retrieval safety gate regex scan (local)
      → Format chunks + bibliography + system prompt
      → LLM provider API call (cloud, unless Ollama)
      → Response with source citations
```

### Chat / Draft / Rewrite — Tree Mode (LLM search + cloud generation)
```
Query → Load tree index JSON (local)
      → LLM call: select relevant sections from tree structure
      → pdfplumber: fetch page text for selected sections (local)
      → Retrieval safety gate regex scan (local)
      → Format chunks + bibliography + system prompt
      → LLM provider API call (cloud, unless Ollama)
      → Response with source citations
```

### Tree Index Building (on-demand, per document)
```
PDF → pdfplumber page extraction (local)
    → LLM call: detect TOC or generate structure
    → LLM call(s): subdivide large sections (>15 pages)
    → Save tree JSON to data/tree_indexes/ (local)
```

### Export (100% local)
```
Section drafts (JSON) → python-docx formatting → .docx with References chapter
```
