"""Version management — timestamped .docx exports with manifest tracking."""

import json
import difflib
from datetime import datetime, timezone
from pathlib import Path

from config import VERSIONS_DIR, MANIFEST_PATH
from core.docexport import export_full_paper
from core.paper_structure import list_sections_status


def _load_manifest() -> list[dict]:
    """Load the version manifest."""
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return []


def _save_manifest(manifest: list[dict]) -> None:
    """Save the version manifest."""
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def save_version(label: str = "") -> dict:
    """Export the current paper state as a versioned .docx and record in manifest."""
    now = datetime.now(timezone.utc)
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    label_slug = label.replace(" ", "-").lower() if label else "snapshot"
    filename = f"paper_{timestamp}_{label_slug}.docx"
    output_path = VERSIONS_DIR / filename

    export_full_paper(output_path)

    # Collect section statuses
    statuses = list_sections_status()
    drafted = sum(1 for s in statuses if s["status"] != "empty")

    entry = {
        "version": len(_load_manifest()) + 1,
        "label": label or "snapshot",
        "filename": filename,
        "timestamp": now.isoformat(),
        "sections_drafted": drafted,
        "total_sections": len(statuses),
    }

    manifest = _load_manifest()
    manifest.append(entry)
    _save_manifest(manifest)

    return entry


def list_versions() -> list[dict]:
    """Return all version entries from the manifest."""
    return _load_manifest()


def compare_versions(v1: int, v2: int) -> str:
    """Compare two versions by reading their .docx files and diffing the text.

    v1, v2 are 1-based version numbers from the manifest.
    Returns a unified diff string.
    """
    manifest = _load_manifest()

    def _find(num: int) -> Path:
        for entry in manifest:
            if entry["version"] == num:
                return VERSIONS_DIR / entry["filename"]
        raise ValueError(f"Version {num} not found in manifest.")

    from docx import Document

    def _extract_text(path: Path) -> list[str]:
        doc = Document(str(path))
        return [p.text for p in doc.paragraphs if p.text.strip()]

    text1 = _extract_text(_find(v1))
    text2 = _extract_text(_find(v2))

    diff = difflib.unified_diff(
        text1, text2,
        fromfile=f"v{v1}", tofile=f"v{v2}",
        lineterm="",
    )
    return "\n".join(diff)
