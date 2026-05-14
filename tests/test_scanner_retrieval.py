"""Tests for retrieval-time scanner gate."""
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_log_retrieval_flag(tmp_path, monkeypatch):
    import config
    monkeypatch.setattr(config, "REPORTS_DIR", tmp_path)

    from core.retrieval import _log_retrieval_flag
    from core.scanner import Threat

    threat = Threat("role_hijack_01", "role_hijack", "ignore instructions", "chunk:3", "high", 1.0)
    chunk = {"source_pdf": "evil.pdf", "chunk_index": 3}
    _log_retrieval_flag("test query", chunk, [threat])

    log_path = tmp_path / "retrieval_flags.log"
    assert log_path.exists()
    content = log_path.read_text()
    assert "evil.pdf" in content
    assert "role_hijack_01" in content
