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

# ── User Settings (persisted) ─────────────────────────────────────────────
SETTINGS_PATH = DATA_DIR / "settings.json"


def _load_settings() -> dict:
    """Load user-overridden settings from settings.json."""
    try:
        if SETTINGS_PATH.exists():
            import json
            return json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def save_settings(updates: dict) -> None:
    """Merge *updates* into the persisted settings file and refresh in-memory values."""
    import json
    current = _load_settings()
    current.update(updates)
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_PATH.write_text(json.dumps(current, indent=2, ensure_ascii=False), encoding="utf-8")
    _reload_settings()


_user_settings = _load_settings()


def _reload_settings() -> None:
    """Refresh module-level config values from the settings file."""
    global _user_settings, PAPER_TITLE, DOCX_FONT, DOCX_FONT_SIZE_PT
    global DOCX_LINE_SPACING, DOCX_MARGIN_INCHES
    _user_settings = _load_settings()
    PAPER_TITLE = _user_settings.get(
        "paper_title",
        os.environ.get(
            "PAPER_TITLE",
            "Developing a Cyber Resilience Act compliance toolkit for SMEs",
        ),
    )
    DOCX_FONT = _user_settings.get("docx_font", "Times New Roman")
    DOCX_FONT_SIZE_PT = _user_settings.get("docx_font_size", 12)
    DOCX_LINE_SPACING = _user_settings.get("docx_line_spacing", 1.5)
    DOCX_MARGIN_INCHES = _user_settings.get("docx_margin_inches", 1.0)


# ── Paper ─────────────────────────────────────────────────────────────────
PAPER_TITLE = _user_settings.get(
    "paper_title",
    os.environ.get(
        "PAPER_TITLE",
        "Developing a Cyber Resilience Act compliance toolkit for SMEs",
    ),
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
# "ollama", "claude", "openai", or "openrouter"
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "ollama")

# ── Claude API ────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
CLAUDE_DRAFT_MODEL = "claude-sonnet-4-6"
CLAUDE_POLISH_MODEL = "claude-opus-4-6"

# ── Ollama ────────────────────────────────────────────────────────────────
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_DRAFT_MODEL = os.environ.get("OLLAMA_DRAFT_MODEL", "llama3.1:latest")
OLLAMA_POLISH_MODEL = os.environ.get("OLLAMA_POLISH_MODEL", "gemma3:12b")

# ── LM Studio ─────────────────────────────────────────────────────────────
LMSTUDIO_URL = os.environ.get("LMSTUDIO_URL", "http://localhost:1234/v1")
LMSTUDIO_DRAFT_MODEL = os.environ.get("LMSTUDIO_DRAFT_MODEL", "")
LMSTUDIO_POLISH_MODEL = os.environ.get("LMSTUDIO_POLISH_MODEL", "")

# ── OpenAI (also works with HuggingFace Inference Endpoints) ─────────────
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "")  # e.g. https://xyz.endpoints.huggingface.cloud/v1
OPENAI_DRAFT_MODEL = os.environ.get("OPENAI_DRAFT_MODEL", "gpt-4o")
OPENAI_POLISH_MODEL = os.environ.get("OPENAI_POLISH_MODEL", "gpt-4o")

# ── OpenRouter ────────────────────────────────────────────────────────────
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_DRAFT_MODEL = os.environ.get("OPENROUTER_DRAFT_MODEL", "google/gemini-2.5-flash-preview")
OPENROUTER_POLISH_MODEL = os.environ.get("OPENROUTER_POLISH_MODEL", "google/gemini-2.5-pro-preview")

# ── Resolved models (based on provider) ──────────────────────────────────
_PROVIDER_MODELS = {
    "ollama": (OLLAMA_DRAFT_MODEL, OLLAMA_POLISH_MODEL),
    "lmstudio": (LMSTUDIO_DRAFT_MODEL, LMSTUDIO_POLISH_MODEL),
    "claude": (CLAUDE_DRAFT_MODEL, CLAUDE_POLISH_MODEL),
    "openai": (OPENAI_DRAFT_MODEL, OPENAI_POLISH_MODEL),
    "openrouter": (OPENROUTER_DRAFT_MODEL, OPENROUTER_POLISH_MODEL),
}
DRAFT_MODEL, POLISH_MODEL = _PROVIDER_MODELS.get(LLM_PROVIDER, (OLLAMA_DRAFT_MODEL, OLLAMA_POLISH_MODEL))

MAX_CONTEXT_TOKENS = 180_000  # leave headroom below 200k

# ── Document Export ───────────────────────────────────────────────────────
DOCX_FONT = _user_settings.get("docx_font", "Times New Roman")
DOCX_FONT_SIZE_PT = _user_settings.get("docx_font_size", 12)
DOCX_LINE_SPACING = _user_settings.get("docx_line_spacing", 1.5)
DOCX_MARGIN_INCHES = _user_settings.get("docx_margin_inches", 1.0)

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

    # 2. API key warnings
    if LLM_PROVIDER == "claude" and not ANTHROPIC_API_KEY:
        log.warning("LLM_PROVIDER is 'claude' but ANTHROPIC_API_KEY is not set")
    if LLM_PROVIDER == "openai" and not OPENAI_API_KEY:
        log.warning("LLM_PROVIDER is 'openai' but OPENAI_API_KEY is not set")
    if LLM_PROVIDER == "openrouter" and not OPENROUTER_API_KEY:
        log.warning("LLM_PROVIDER is 'openrouter' but OPENROUTER_API_KEY is not set")

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
