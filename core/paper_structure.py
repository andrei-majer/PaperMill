"""Paper outline and section draft management."""

import json
import re
from pathlib import Path
from config import SECTIONS_DIR, _load_settings, save_settings

# Only allow alphanumeric, dots, hyphens, underscores in section IDs
_SAFE_ID_PATTERN = re.compile(r'^[a-zA-Z0-9._-]+$')


def _validate_section_id(section_id: str) -> str:
    """Validate and sanitize a section ID to prevent path traversal."""
    section_id = section_id.strip()
    if not section_id:
        raise ValueError("Section ID cannot be empty")
    if not _SAFE_ID_PATTERN.match(section_id):
        raise ValueError(f"Invalid section ID '{section_id}': only letters, numbers, dots, hyphens, underscores allowed")
    if '..' in section_id:
        raise ValueError(f"Invalid section ID '{section_id}': path traversal not allowed")
    return section_id

# ── Default Paper Outline ────────────────────────────────────────────────
# Each entry: (section_id, title, suggested_word_count)
_DEFAULT_OUTLINE: list[tuple[str, str, int]] = [
    ("abstract", "Abstract", 300),
    ("ch1", "Chapter 1 — Introduction", 2500),
    ("ch1.1", "1.1 Background and Context", 800),
    ("ch1.2", "1.2 Problem Statement", 600),
    ("ch1.3", "1.3 Research Objectives and Questions", 500),
    ("ch1.4", "1.4 Scope and Limitations", 400),
    ("ch1.5", "1.5 Structure of the Paper", 300),
    ("ch2", "Chapter 2 — Literature Review", 5000),
    ("ch2.1", "2.1 The Cyber Resilience Act: Legislative Overview", 1200),
    ("ch2.2", "2.2 SME Cybersecurity Landscape", 1000),
    ("ch2.3", "2.3 Compliance Frameworks and Standards", 1000),
    ("ch2.4", "2.4 Existing Compliance Toolkits and Gaps", 1000),
    ("ch2.5", "2.5 Summary of Literature Gaps", 500),
    ("ch3", "Chapter 3 — Methodology", 3500),
    ("ch3.1", "3.1 Research Design", 800),
    ("ch3.2", "3.2 Data Collection Methods", 800),
    ("ch3.3", "3.3 Toolkit Development Approach", 800),
    ("ch3.4", "3.4 Evaluation Framework", 600),
    ("ch3.5", "3.5 Ethical Considerations", 400),
    ("ch4", "Chapter 4 — The Cyber Resilience Act: Analysis", 3000),
    ("ch4.1", "4.1 Key Requirements and Obligations", 1000),
    ("ch4.2", "4.2 Impact Assessment for SMEs", 1000),
    ("ch4.3", "4.3 Compliance Challenges", 800),
    ("ch5", "Chapter 5 — Toolkit Design and Development", 4000),
    ("ch5.1", "5.1 Requirements Analysis", 800),
    ("ch5.2", "5.2 Toolkit Architecture", 1000),
    ("ch5.3", "5.3 Component Design", 1200),
    ("ch5.4", "5.4 Implementation Details", 800),
    ("ch6", "Chapter 6 — Evaluation and Results", 3500),
    ("ch6.1", "6.1 Evaluation Methodology", 800),
    ("ch6.2", "6.2 Results and Analysis", 1200),
    ("ch6.3", "6.3 Comparison with Existing Solutions", 800),
    ("ch6.4", "6.4 Limitations", 500),
    ("ch7", "Chapter 7 — Discussion", 2500),
    ("ch7.1", "7.1 Interpretation of Findings", 800),
    ("ch7.2", "7.2 Implications for SMEs", 700),
    ("ch7.3", "7.3 Contributions to Knowledge", 500),
    ("ch7.4", "7.4 Future Research Directions", 400),
    ("ch8", "Chapter 8 — Conclusion", 1500),
    ("references", "References", 0),
    ("appendices", "Appendices", 0),
]


def _load_outline() -> list[tuple[str, str, int]]:
    """Load outline from settings, falling back to default."""
    settings = _load_settings()
    saved = settings.get("paper_outline")
    if saved and isinstance(saved, list):
        return [(s["id"], s["title"], s["words"]) for s in saved]
    return list(_DEFAULT_OUTLINE)


def save_outline(outline: list[tuple[str, str, int]]) -> None:
    """Persist the paper outline to settings."""
    serialized = [{"id": sid, "title": title, "words": words} for sid, title, words in outline]
    save_settings({"paper_outline": serialized})


def add_section(section_id: str, title: str, words: int, after: str | None = None) -> None:
    """Add a section to the outline. Inserts after *after* if given, else appends."""
    section_id = _validate_section_id(section_id)
    if after:
        after = _validate_section_id(after)
    outline = _load_outline()
    # Prevent duplicates
    if any(sid == section_id for sid, _, _ in outline):
        raise ValueError(f"Section '{section_id}' already exists")
    entry = (section_id, title, words)
    if after:
        idx = next((i for i, (sid, _, _) in enumerate(outline) if sid == after), None)
        if idx is not None:
            outline.insert(idx + 1, entry)
        else:
            outline.append(entry)
    else:
        outline.append(entry)
    save_outline(outline)
    _rebuild_index()


def remove_section(section_id: str) -> None:
    """Remove a section from the outline."""
    section_id = _validate_section_id(section_id)
    outline = _load_outline()
    outline = [(sid, t, w) for sid, t, w in outline if sid != section_id]
    save_outline(outline)
    _rebuild_index()


def move_section(section_id: str, direction: str) -> None:
    """Move a section up or down in the outline."""
    section_id = _validate_section_id(section_id)
    outline = _load_outline()
    idx = next((i for i, (sid, _, _) in enumerate(outline) if sid == section_id), None)
    if idx is None:
        return
    if direction == "up" and idx > 0:
        outline[idx], outline[idx - 1] = outline[idx - 1], outline[idx]
    elif direction == "down" and idx < len(outline) - 1:
        outline[idx], outline[idx + 1] = outline[idx + 1], outline[idx]
    save_outline(outline)
    _rebuild_index()


def reset_outline() -> None:
    """Reset outline to the built-in default."""
    save_outline(list(_DEFAULT_OUTLINE))
    _rebuild_index()


# ── Live outline (always read from settings) ─────────────────────────────
PAPER_OUTLINE: list[tuple[str, str, int]] = _load_outline()

# O(1) lookup dict keyed by section_id → (title, target_words)
_OUTLINE_INDEX: dict[str, tuple[str, int]] = {
    sid: (title, words) for sid, title, words in PAPER_OUTLINE
}


def _rebuild_index() -> None:
    """Rebuild PAPER_OUTLINE and _OUTLINE_INDEX from disk after mutations."""
    global PAPER_OUTLINE, _OUTLINE_INDEX
    PAPER_OUTLINE = _load_outline()
    _OUTLINE_INDEX = {sid: (title, words) for sid, title, words in PAPER_OUTLINE}


# ── Section Draft Storage ─────────────────────────────────────────────────

def _section_path(section_id: str) -> Path:
    section_id = _validate_section_id(section_id)
    return SECTIONS_DIR / f"{section_id}.json"


def load_section(section_id: str) -> dict | None:
    """Load a section draft. Returns dict with keys: id, title, text, status, updated_at."""
    path = _section_path(section_id)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _undo_path(section_id: str) -> Path:
    section_id = _validate_section_id(section_id)
    return SECTIONS_DIR / f"{section_id}.undo.json"


def save_section(section_id: str, title: str, text: str, status: str = "draft") -> None:
    """Save or update a section draft. Backs up the previous version for undo."""
    from datetime import datetime, timezone

    # Back up current version before overwriting
    current_path = _section_path(section_id)
    if current_path.exists():
        _undo_path(section_id).write_text(
            current_path.read_text(encoding="utf-8"), encoding="utf-8"
        )

    data = {
        "id": section_id,
        "title": title,
        "text": text,
        "status": status,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    current_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def undo_section(section_id: str) -> bool:
    """Restore the previous version of a section. Returns True if successful."""
    undo = _undo_path(section_id)
    if not undo.exists():
        return False
    current = _section_path(section_id)
    current.write_text(undo.read_text(encoding="utf-8"), encoding="utf-8")
    undo.unlink()
    return True


def has_undo(section_id: str) -> bool:
    """Check if an undo backup exists for a section."""
    return _undo_path(section_id).exists()


def list_sections_status() -> list[dict]:
    """Return outline with draft status for each section."""
    result = []
    for sid, title, words in _load_outline():
        section = load_section(sid)
        status = section["status"] if section else "empty"
        result.append({
            "id": sid,
            "title": title,
            "target_words": words,
            "status": status,
        })
    return result


def get_section_title(section_id: str) -> str:
    """Get the title for a section ID from the outline."""
    entry = _OUTLINE_INDEX.get(section_id)
    return entry[0] if entry is not None else section_id


def get_section_target_words(section_id: str) -> int:
    """Get the target word count for a section."""
    entry = _OUTLINE_INDEX.get(section_id)
    return entry[1] if entry is not None else 1000
