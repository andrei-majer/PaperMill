# PaperMill — Security Architecture

Defense-in-depth security across ingestion, retrieval, generation, and user interaction layers.

---

## Content Safety Scanner

Multi-backend scanner protecting the RAG pipeline against prompt injection and data exfiltration hidden in ingested documents.

### Backends

| Backend | Method | Trigger | Confidence |
|---|---|---|---|
| **RegexBackend** | 15+ patterns across 5 threat categories | Always runs first | 1.0 (direct), 0.9 (super-cleaned) |
| **HeuristicBackend** | Imperative density + pronoun density scoring | After regex, if no high-severity hit | 0.0–1.0 suspicion score |
| **OllamaClassifierBackend** | LLM-based threat classification via Ollama | When heuristic score > 0.6 threshold | Fail-closed on parse/connection errors |

### Threat Categories (scanner_rules.json)

| Category | Patterns | Examples |
|---|---|---|
| `role_hijack` | 5 | "ignore previous instructions", "you are now", "forget your instructions" |
| `system_prompt_leak` | 2 | "print your system prompt", "what are your instructions" |
| `delimiter_injection` | 3 | ChatML (`<\|im_start\|>`), Llama (`[INST]`), generic role delimiters |
| `instruction_override` | 2 | "IMPORTANT: you must", "do not follow the previous" |
| `exfiltration` | 4 | URL exfiltration, credential exfiltration, webhook/callback references |

Rules are stored in `data/scanner_rules.json` — inspectable and editable without code changes. Each rule has an `id`, `category`, `severity`, `scope` (all/document/chat), `pattern`, and `description`.

### Evasion Resistance

| Technique | Defense |
|---|---|
| Zero-width characters (U+200B, U+FEFF, etc.) | 13 Unicode zero-width chars stripped before matching |
| Unicode tricks | NFKC normalization applied to all text |
| Spaced-out text (`i g n o r e`) | Super-cleaned matching removes all non-alphanumeric chars for high-severity rules |
| Case variation | All regex compiled with `re.IGNORECASE` |
| Whitespace padding | Whitespace collapsed before matching |

### Scan Points

| Gate | When | Method | Behavior |
|---|---|---|---|
| **PDF Ingestion** | Before embedding | Structure + metadata + per-chunk content scan | Block + quarantine |
| **Image Ingestion** | Before embedding | OCR/caption divergence + content scan | Block + quarantine |
| **Retrieval** | Before LLM context | Regex-only scan on returned chunks | Filter + flag in DB |
| **Chat Input** | Before query processing | Regex-only scan, chat scope | Warn (not block) |

### Scan Flow & Budget

1. Regex backend runs on all text (always)
2. High-severity regex hits short-circuit further scanning
3. Heuristic backend runs if no high-severity found
4. LLM escalation triggers when heuristic score exceeds `SCANNER_SUSPICION_THRESHOLD` (default: 0.6)
5. LLM escalation budget: max `SCANNER_MAX_LLM_ESCALATIONS` (default: 10) calls per file
6. `SCANNER_DRY_RUN` mode: scan runs but files are ingested anyway (default: off — fail-closed)

### Scan History & Caching

- Stored in `reports/scan_history.json`, keyed by SHA-256 file hash
- Tracks: filename, first/last seen, result (passed/blocked/released), pattern version, report path
- If current pattern version matches cached version and result is "passed", re-scan is skipped
- Thread-safe writes via `FileLock`

---

## PDF Structure Scanning

Recursive analysis of PDF internal objects before content extraction.

### Dangerous Keys Detected

| Key | Risk |
|---|---|
| `/JS`, `/JavaScript` | Embedded JavaScript execution |
| `/OpenAction` | Auto-execute on open |
| `/AA` | Additional actions (triggers on events) |
| `/EmbeddedFile` | Hidden embedded files |
| `/Launch` | External application launch |

### Additional Checks

- **Encrypted PDFs** flagged as unsafe (cannot be scanned)
- **Metadata scanning**: author, title, subject, keywords, creator, producer fields scanned through regex backend
- Recursive object traversal with visited-set tracking to prevent infinite loops

---

## Quarantine System

Files that fail content scanning are quarantined and reported.

### Flow

1. File fails scan → moved to `quarantine/` directory
2. Filename collision handling (appends counter)
3. Markdown report generated in `reports/`
4. Scan history updated with "blocked" result

### Management

| Action | Effect |
|---|---|
| **Release** | Moves file back to source directory, auto-adds to allowlist, updates history to "released" |
| **Clear quarantine** | Deletes all quarantined files, removes "blocked" entries from scan history |
| **Clear reports** | Deletes all scan report files |

### Allowlist

- File hash (SHA-256) based bypass
- Stores: filename, added date, reason per hash
- FileLock-protected read/write
- Allowlisted files skip all scanning but still get ingested normally
- Add/remove via CLI (`/allowlist-add`, `/allowlist-remove`) or Streamlit UI

---

## Prompt Injection Defenses

### Source-Locked Generation

The system prompt (`core/prompts.py`) enforces:

- **No external sources**: "Do NOT use any external sources, web searches, or outside knowledge"
- **Source locking**: "Only use information provided in the retrieved chunks"
- **Citation requirement**: Every factual claim must be followed by its source identifier
- **No fabrication**: "NEVER fabricate, invent, or paraphrase-as-quote text that does not appear verbatim"
- **Hallucination check**: "Internally verify that no 'common knowledge' has leaked into the response"
- **Gap statement**: When no relevant source exists, state the gap explicitly

### Chunk Fencing

Retrieved chunks are wrapped in Markdown code blocks before injection into prompts:

- Backticks in chunk text escaped (```` ``` ```` → `~~~`)
- Each chunk wrapped in ` ```text ... ``` ` fencing
- Source attribution headers separate from chunk content
- Prevents chunk text from being interpreted as prompt instructions

### Chat Input Scanning

- Regex-only scan with "chat" scope before processing user messages
- Rules scoped to "document" (e.g., exfiltration patterns) do not trigger in chat
- Behavior: **warn** the user, do not block the message
- Configurable via `SCANNER_SCAN_CHAT_INPUT` flag (default: enabled)

---

## Retrieval Safety Gate

Re-scans chunks at retrieval time before they reach the LLM.

### How It Works

1. After vector search returns candidate chunks, check if source PDF was scanned with current pattern version
2. If source is up-to-date, skip re-scan
3. If not, run regex-only scan on each chunk
4. Failed chunks are:
   - Excluded from the response
   - Tagged with `safety_flag = "flagged"` in LanceDB
   - Logged to `reports/retrieval_flags.log` with timestamp, query, source, chunk index, and threat IDs
5. Flagged chunks are filtered in future searches via WHERE clause: `safety_flag = '' OR safety_flag IS NULL`

---

## Input Validation

### Filename Sanitization

Applied to all uploaded files (`interfaces/streamlit_app.py`):

| Check | Defense |
|---|---|
| Path component stripping | `Path(name).name` + `PurePosixPath(name).name` |
| Null bytes | `re.sub(r'[\x00/\\]', '', name)` |
| Leading dots (hidden files) | `name.lstrip('.')` |
| Unsafe Windows chars | `re.sub(r'[<>:"\|?*]', '_', name)` |
| Directory traversal | Path separators removed |

### File Upload Limits

| Type | Limit | Accepted Formats |
|---|---|---|
| PDF | 20 MB per file | `.pdf` |
| Image | Streamlit default | `.png`, `.jpg`, `.jpeg`, `.gif`, `.webp`, `.bmp` |

### Duplicate Detection

- **By filename + size**: Same name, same size → skip saving
- **By filename, different size**: Overwrite with warning
- **By size match across directory**: Warn "may be a duplicate"
- **By database check**: Already ingested in vector DB → skip re-ingestion

### DOI Validation

- Pattern: `^10\.\d{4,}/\S+$` — validated before any CrossRef API call
- URL prefix stripping (`https://doi.org/`, `doi:`) before validation
- Retry logic: 3 attempts with exponential backoff (1s, 2s, 4s), 15-second timeout

### Section ID Validation

- Pattern: `^[a-zA-Z0-9._-]+$` — alphanumeric plus safe chars only
- Empty string check
- `..` path traversal explicitly blocked

---

## Database Safety (LanceDB)

### SQL Injection Protection

All WHERE clauses use `_sql_escape()` which escapes single quotes (`'` → `''`):

- Source filtering in vector search
- Chunk flagging operations
- Source deletion operations

### Safety Flag Column

- `safety_flag` field in chunk records
- Set to `"flagged"` when chunk fails retrieval-time scan
- Filtered during search: `WHERE safety_flag = '' OR safety_flag IS NULL`
- Prevents flagged content from reaching the LLM

### Schema Migration

- Missing columns (`safety_flag`, `source_type`) added automatically on first connection
- Run-once flag prevents duplicate migrations

---

## Token & Context Overflow Protection

### Estimation & Truncation

- Token estimation: ~4 characters per token
- Budget: `MAX_CONTEXT_TOKENS` (default: 180,000)
- If `estimated_prompt_tokens + max_tokens > MAX_CONTEXT_TOKENS`:
  - Walk messages in reverse (preserve most recent)
  - Truncate message content to fit budget
  - Log warning

### Chunk Size Limits

- `CHUNK_SIZE_TOKENS = 800` — max tokens per chunk at ingestion
- `CHUNK_OVERLAP_TOKENS = 200` — overlap between consecutive chunks

---

## Image Security

### OCR/Caption Divergence Detection

Detects hidden text in images by comparing OCR extraction length to caption length:

- Threshold ratio: `SCANNER_OCR_DIVERGENCE_RATIO` (default: 5.0x)
- Minimum OCR length: `SCANNER_OCR_DIVERGENCE_MIN_CHARS` (default: 100)
- If OCR text is >5x longer than caption AND >100 chars: flagged as `ocr_caption_divergence` (medium severity, 0.7 confidence)
- Purpose: detect steganographic or hidden text attacks

### Image Processing

- Florence-2 generates both detailed caption and OCR extraction
- Combined description scanned through content safety scanner
- Same quarantine flow as PDFs if blocked

---

## API Key Handling

| Aspect | Implementation |
|---|---|
| Storage | `data/settings.json` (plaintext) with environment variable fallback — **never commit this file** (it is `.gitignore`d) |
| UI masking | All key inputs use `type="password"` in Streamlit |
| Keys managed | `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `OPENROUTER_API_KEY`, `EMBEDDING_API_KEY` |
| Startup check | Warning logged if provider is selected but key is missing |

---

## Startup Validation

Performed at application launch (`config.validate_config()`):

| Check | Action on Failure |
|---|---|
| Directory writability (LanceDB, reports, quarantine, sections, versions) | Log warning |
| API key presence (for selected provider) | Log warning |
| Ollama reachability (`/api/tags` endpoint, 3s timeout) | Log warning |
| Model name sanity (non-empty draft/polish model names) | Log warning |

All checks are non-blocking — warnings are logged but the application starts regardless.

---

## Scan Reports

Generated for every scanned file (passed or blocked), stored in `reports/` as Markdown:

### Report Contents

- **Header**: Filename, date, SHA-256 hash, result (PASSED/BLOCKED), pattern version
- **Threats section** (if any): Pattern name, severity, confidence, location, category, matched text
- **Summary table**: Scan types run, chunks scanned, LLM escalations, threat counts by severity

### Retrieval Flag Log

Separate log at `reports/retrieval_flags.log`:
- Format: `[ISO-timestamp] query={query} source={pdf} chunk_index={idx} threats=[{threat_ids}]`
- Logged when chunks fail regex scan at retrieval time

---

## File Integrity

### SHA-256 Hashing

- All files hashed at ingestion (8192-byte chunks)
- Hash used for: scan history keying, allowlist lookup, duplicate detection
- Deterministic chunk IDs: `SHA-256("{source_pdf}:{chunk_index}")[:16]`

---

## Configuration Reference

| Setting | Default | Description |
|---|---|---|
| `SCANNER_LLM_ESCALATION` | `True` | Enable LLM classifier escalation |
| `SCANNER_SUSPICION_THRESHOLD` | `0.6` | Heuristic score to trigger LLM |
| `SCANNER_MAX_LLM_ESCALATIONS` | `10` | Max LLM calls per file |
| `SCANNER_DRY_RUN` | `False` | Log threats without blocking (fail-closed when False) |
| `SCANNER_SCAN_CHAT_INPUT` | `True` | Scan user chat messages |
| `SCANNER_OCR_DIVERGENCE_RATIO` | `5.0` | OCR/caption length ratio threshold |
| `SCANNER_OCR_DIVERGENCE_MIN_CHARS` | `100` | Minimum OCR length to trigger divergence check |
| `MAX_CONTEXT_TOKENS` | `180,000` | Max tokens before prompt truncation |
| `CHUNK_SIZE_TOKENS` | `800` | Max tokens per chunk |
| `CHUNK_OVERLAP_TOKENS` | `200` | Overlap between chunks |
