"""Configuration constants for PaperMill — AI-Powered Academic Writing Assistant."""

import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
PDF_DIR = DATA_DIR / "pdfs"
LANCEDB_DIR = Path.home() / ".claude" / "lancedb" / "paper-assistant"
VERSIONS_DIR = PROJECT_ROOT / "versions"
SECTIONS_DIR = PROJECT_ROOT / "paper_sections"
MANIFEST_PATH = VERSIONS_DIR / "manifest.json"
CHAT_HISTORY_PATH = DATA_DIR / "chat_history.json"

# ── Paper ─────────────────────────────────────────────────────────────────
PAPER_TITLE = os.environ.get(
    "PAPER_TITLE",
    "Developing a Cyber Resilience Act compliance toolkit for SMEs",
)

# Ensure directories exist
for d in (PDF_DIR, LANCEDB_DIR, VERSIONS_DIR, SECTIONS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ── LanceDB ────────────────────────────────────────────────────────────────
LANCE_TABLE_NAME = "chunks"

# ── Embeddings ─────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "jinaai/jina-embeddings-v3"
EMBEDDING_DIM = 1024
MAX_TOKEN_CTX = 8192

# ── Chunking ──────────────────────────────────────────────────────────────
CHUNK_SIZE_TOKENS = 800
CHUNK_OVERLAP_TOKENS = 200

# ── Retrieval ─────────────────────────────────────────────────────────────
DEFAULT_TOP_K = 8

# ── LLM Provider ─────────────────────────────────────────────────────────
# "claude" or "ollama"
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "ollama")

# ── Claude API ────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
CLAUDE_DRAFT_MODEL = "claude-sonnet-4-6"
CLAUDE_POLISH_MODEL = "claude-opus-4-6"

# ── Ollama ────────────────────────────────────────────────────────────────
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_DRAFT_MODEL = os.environ.get("OLLAMA_DRAFT_MODEL", "dolphin-llama3:latest")
OLLAMA_POLISH_MODEL = os.environ.get("OLLAMA_POLISH_MODEL", "dolphin-llama3:latest")

# ── Resolved models (based on provider) ──────────────────────────────────
if LLM_PROVIDER == "ollama":
    DRAFT_MODEL = OLLAMA_DRAFT_MODEL
    POLISH_MODEL = OLLAMA_POLISH_MODEL
else:
    DRAFT_MODEL = CLAUDE_DRAFT_MODEL
    POLISH_MODEL = CLAUDE_POLISH_MODEL

MAX_CONTEXT_TOKENS = 180_000  # leave headroom below 200k

# ── Document Export ───────────────────────────────────────────────────────
DOCX_FONT = "Times New Roman"
DOCX_FONT_SIZE_PT = 12
DOCX_LINE_SPACING = 1.5
DOCX_MARGIN_INCHES = 1.0

# ── Content Scanner ──────────────────────────────────────────────────────
REPORTS_DIR = PROJECT_ROOT / "reports"
QUARANTINE_DIR = PROJECT_ROOT / "quarantine"
SCAN_HISTORY_PATH = REPORTS_DIR / "scan_history.json"
SCANNER_RULES_PATH = DATA_DIR / "scanner_rules.json"
SCANNER_ALLOWLIST_PATH = DATA_DIR / "scanner_allowlist.json"
SCANNER_BACKENDS = ["regex", "ollama"]        # future: "nemo", "llama_guard"
SCANNER_LLM_ESCALATION = True
SCANNER_SUSPICION_THRESHOLD = 0.6
SCANNER_MAX_LLM_ESCALATIONS = 10
SCANNER_DRY_RUN = False
SCANNER_SCAN_CHAT_INPUT = True
SCANNER_OCR_DIVERGENCE_RATIO = 5.0
SCANNER_OCR_DIVERGENCE_MIN_CHARS = 100

# Ensure scanner directories exist
for d in (REPORTS_DIR, QUARANTINE_DIR):
    d.mkdir(parents=True, exist_ok=True)


# ── Startup Validation ────────────────────────────────────────────────────

def validate_config() -> None:
    """Check key config values at startup and log warnings for problems.

    Never raises — safe to call unconditionally from any entry point.
    """
    import logging
    import tempfile
    import urllib.request
    import urllib.error

    log = logging.getLogger(__name__)

    # 1. Directory writability
    _writable_dirs = {
        "LANCEDB_DIR": LANCEDB_DIR,
        "REPORTS_DIR": REPORTS_DIR,
        "QUARANTINE_DIR": QUARANTINE_DIR,
        "SECTIONS_DIR": SECTIONS_DIR,
        "VERSIONS_DIR": VERSIONS_DIR,
    }
    for name, directory in _writable_dirs.items():
        try:
            directory.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=directory, delete=True):
                pass
        except Exception as exc:
            log.warning("%s (%s) is not writable: %s", name, directory, exc)

    # 2. API key warning
    if LLM_PROVIDER == "claude" and not ANTHROPIC_API_KEY:
        log.warning("LLM_PROVIDER is 'claude' but ANTHROPIC_API_KEY is not set")

    # 3. Ollama reachability
    if LLM_PROVIDER == "ollama":
        try:
            url = OLLAMA_URL.rstrip("/") + "/api/tags"
            urllib.request.urlopen(url, timeout=3)
        except Exception:
            log.warning(
                "Ollama not reachable at %s — is it running?", OLLAMA_URL
            )

        # 4. Model name sanity
        if not OLLAMA_DRAFT_MODEL:
            log.warning("OLLAMA_DRAFT_MODEL is empty")
        if not OLLAMA_POLISH_MODEL:
            log.warning("OLLAMA_POLISH_MODEL is empty")
