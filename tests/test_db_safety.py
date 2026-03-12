"""Tests for safety_flag DB operations."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_chunk_record_has_safety_flag():
    from core.db import ChunkRecord
    fields = ChunkRecord.model_fields
    assert "safety_flag" in fields


def test_tag_chunk_flagged_updates_record(monkeypatch):
    from unittest.mock import MagicMock
    import core.db as db_module

    mock_table = MagicMock()
    monkeypatch.setattr(db_module, "get_or_create_table", lambda: mock_table)

    from core.db import tag_chunk_flagged
    tag_chunk_flagged("test.pdf", 3)
    mock_table.update.assert_called_once_with(
        where="source_pdf = 'test.pdf' AND chunk_index = 3",
        values={"safety_flag": "flagged"},
    )
