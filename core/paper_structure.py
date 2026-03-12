"""Paper outline and section draft management."""

import json
from pathlib import Path
from config import SECTIONS_DIR

# ── Paper Outline ─────────────────────────────────────────────────────────
# Each entry: (section_id, title, suggested_word_count)
PAPER_OUTLINE: list[tuple[str, str, int]] = [
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

# O(1) lookup dict keyed by section_id → (title, target_words)
_OUTLINE_INDEX: dict[str, tuple[str, int]] = {
    sid: (title, words) for sid, title, words in PAPER_OUTLINE
}


# ── Section Draft Storage ─────────────────────────────────────────────────

def _section_path(section_id: str) -> Path:
    return SECTIONS_DIR / f"{section_id}.json"


def load_section(section_id: str) -> dict | None:
    """Load a section draft. Returns dict with keys: id, title, text, status, updated_at."""
    path = _section_path(section_id)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def save_section(section_id: str, title: str, text: str, status: str = "draft") -> None:
    """Save or update a section draft."""
    from datetime import datetime, timezone
    data = {
        "id": section_id,
        "title": title,
        "text": text,
        "status": status,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    _section_path(section_id).write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def list_sections_status() -> list[dict]:
    """Return outline with draft status for each section."""
    result = []
    for sid, title, words in PAPER_OUTLINE:
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
