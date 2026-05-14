"""Tests for reingest_all flow."""
import sys
from pathlib import Path
from io import BytesIO
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from reportlab.pdfgen import canvas


def _make_pdf(path: Path, text: str) -> Path:
    """Create a simple PDF with text content."""
    buf = BytesIO()
    c = canvas.Canvas(buf)
    c.drawString(72, 720, text)
    c.save()
    buf.seek(0)
    path.write_bytes(buf.getvalue())
    return path


def test_reingest_wipes_and_reingests(tmp_path, monkeypatch):
    """reingest_all wipes the table and re-ingests all PDFs."""
    import config
    import core.db as db_module

    lance_dir = tmp_path / "lancedb"
    lance_dir.mkdir()
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    settings_path = tmp_path / "settings.json"

    monkeypatch.setattr(config, "LANCEDB_DIR", lance_dir)
    monkeypatch.setattr(config, "PDF_DIR", pdf_dir)
    monkeypatch.setattr(config, "REPORTS_DIR", reports_dir)
    monkeypatch.setattr(config, "SETTINGS_PATH", settings_path)
    monkeypatch.setattr(config, "EMBEDDING_PROVIDER", "local")
    monkeypatch.setattr(config, "EMBEDDING_MODEL", "test-model")
    monkeypatch.setattr(config, "LAST_EMBEDDING_MODEL", "")
    monkeypatch.setattr(db_module, "LANCEDB_DIR", lance_dir)
    monkeypatch.setattr(db_module, "_db", None)
    monkeypatch.setattr(db_module, "_migrated", False)

    # Create test PDFs
    _make_pdf(pdf_dir / "paper1.pdf", "The Cyber Resilience Act establishes cybersecurity requirements for digital products in the EU market.")
    _make_pdf(pdf_dir / "paper2.pdf", "This paper examines the intersection of machine learning and network security in modern enterprise environments.")

    # Mock embedder to avoid loading real model
    mock_vecs = [[0.1] * 1024] * 10  # enough for any chunk count
    with patch("core.embedder.detect_and_save_dim", return_value=1024):
        with patch("core.embedder.embed_passages", return_value=mock_vecs[:2]):
            from core.ingestion import reingest_all
            result = reingest_all()

    assert result["ingested"] >= 1
    assert result["total_chunks"] >= 1
    assert result["failed"] == [] or len(result["failed"]) == 0


def test_reingest_progress_callback(tmp_path, monkeypatch):
    """Progress callback is called for each PDF."""
    import config
    import core.db as db_module

    lance_dir = tmp_path / "lancedb"
    lance_dir.mkdir()
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    settings_path = tmp_path / "settings.json"

    monkeypatch.setattr(config, "LANCEDB_DIR", lance_dir)
    monkeypatch.setattr(config, "PDF_DIR", pdf_dir)
    monkeypatch.setattr(config, "REPORTS_DIR", reports_dir)
    monkeypatch.setattr(config, "SETTINGS_PATH", settings_path)
    monkeypatch.setattr(config, "EMBEDDING_PROVIDER", "local")
    monkeypatch.setattr(config, "EMBEDDING_MODEL", "test-model")
    monkeypatch.setattr(config, "LAST_EMBEDDING_MODEL", "")
    monkeypatch.setattr(db_module, "LANCEDB_DIR", lance_dir)
    monkeypatch.setattr(db_module, "_db", None)
    monkeypatch.setattr(db_module, "_migrated", False)

    _make_pdf(pdf_dir / "a.pdf", "Content for testing progress callback functionality.")
    _make_pdf(pdf_dir / "b.pdf", "More content for second document in progress test.")

    progress_calls = []

    def track_progress(current, total, filename):
        progress_calls.append((current, total, filename))

    mock_vecs = [[0.1] * 1024] * 10
    with patch("core.embedder.detect_and_save_dim", return_value=1024):
        with patch("core.embedder.embed_passages", return_value=mock_vecs[:2]):
            from core.ingestion import reingest_all
            reingest_all(progress_callback=track_progress)

    assert len(progress_calls) == 2
    assert progress_calls[0][1] == 2  # total
    assert progress_calls[1][1] == 2


def test_reingest_failed_pdfs_dont_block_others(tmp_path, monkeypatch):
    """A failing PDF doesn't prevent other PDFs from being ingested."""
    import config
    import core.db as db_module

    lance_dir = tmp_path / "lancedb"
    lance_dir.mkdir()
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    settings_path = tmp_path / "settings.json"

    monkeypatch.setattr(config, "LANCEDB_DIR", lance_dir)
    monkeypatch.setattr(config, "PDF_DIR", pdf_dir)
    monkeypatch.setattr(config, "REPORTS_DIR", reports_dir)
    monkeypatch.setattr(config, "SETTINGS_PATH", settings_path)
    monkeypatch.setattr(config, "EMBEDDING_PROVIDER", "local")
    monkeypatch.setattr(config, "EMBEDDING_MODEL", "test-model")
    monkeypatch.setattr(config, "LAST_EMBEDDING_MODEL", "")
    monkeypatch.setattr(db_module, "LANCEDB_DIR", lance_dir)
    monkeypatch.setattr(db_module, "_db", None)
    monkeypatch.setattr(db_module, "_migrated", False)

    # Create one valid PDF and one corrupt file
    _make_pdf(pdf_dir / "good.pdf", "Valid academic content about cybersecurity frameworks.")
    (pdf_dir / "bad.pdf").write_bytes(b"not a pdf at all")

    mock_vecs = [[0.1] * 1024] * 10
    with patch("core.embedder.detect_and_save_dim", return_value=1024):
        with patch("core.embedder.embed_passages", return_value=mock_vecs[:2]):
            from core.ingestion import reingest_all
            result = reingest_all()

    # bad.pdf should fail but good.pdf should succeed
    assert len(result["failed"]) >= 1
    assert any("bad.pdf" in f for f in result["failed"])


def test_reingest_updates_last_embedding_model(tmp_path, monkeypatch):
    """After reingest, last_embedding_model is updated in memory and on disk."""
    import config
    import core.db as db_module

    lance_dir = tmp_path / "lancedb"
    lance_dir.mkdir()
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    settings_path = tmp_path / "settings.json"

    monkeypatch.setattr(config, "LANCEDB_DIR", lance_dir)
    monkeypatch.setattr(config, "PDF_DIR", pdf_dir)
    monkeypatch.setattr(config, "REPORTS_DIR", reports_dir)
    monkeypatch.setattr(config, "SETTINGS_PATH", settings_path)
    monkeypatch.setattr(config, "EMBEDDING_PROVIDER", "ollama")
    monkeypatch.setattr(config, "EMBEDDING_MODEL", "nomic-embed-text")
    monkeypatch.setattr(config, "LAST_EMBEDDING_MODEL", "local:old-model")
    monkeypatch.setattr(db_module, "LANCEDB_DIR", lance_dir)
    monkeypatch.setattr(db_module, "_db", None)
    monkeypatch.setattr(db_module, "_migrated", False)

    # No PDFs needed — just test the model key update
    with patch("core.embedder.detect_and_save_dim", return_value=768):
        from core.ingestion import reingest_all
        reingest_all()

    # In-memory update
    assert config.LAST_EMBEDDING_MODEL == "ollama:nomic-embed-text"

    # On-disk update
    import json
    saved = json.loads(settings_path.read_text(encoding="utf-8"))
    assert saved["last_embedding_model"] == "ollama:nomic-embed-text"
