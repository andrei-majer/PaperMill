"""Tests for dynamic ChunkRecord schema."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_make_chunk_record_default_dim():
    from core.db import make_chunk_record
    Record = make_chunk_record(1024)
    assert hasattr(Record, 'model_fields')
    assert 'vector' in Record.model_fields


def test_make_chunk_record_different_dim():
    from core.db import make_chunk_record
    Record384 = make_chunk_record(384)
    Record1024 = make_chunk_record(1024)
    # Verify actual vector dimensions via Arrow schema
    schema384 = Record384.to_arrow_schema()
    schema1024 = Record1024.to_arrow_schema()
    assert schema384.field("vector").type.list_size == 384
    assert schema1024.field("vector").type.list_size == 1024


def test_dimension_mismatch_detected(tmp_path, monkeypatch):
    """If existing table has different dim than config, wipe_table is needed."""
    import config
    import core.db as db_module
    lance_dir = tmp_path / "lancedb"
    lance_dir.mkdir()
    monkeypatch.setattr(config, "LANCEDB_DIR", lance_dir)
    monkeypatch.setattr(db_module, "LANCEDB_DIR", lance_dir)
    monkeypatch.setattr(db_module, "_db", None)
    monkeypatch.setattr(db_module, "_migrated", False)

    # Create table with 1024 dim
    monkeypatch.setattr(config, "EMBEDDING_DIM", 1024)
    table = db_module.get_or_create_table()
    assert config.LANCE_TABLE_NAME in db_module.get_db().table_names()

    # Change config to 384 dim
    monkeypatch.setattr(config, "EMBEDDING_DIM", 384)
    monkeypatch.setattr(db_module, "_migrated", False)

    # The table still exists with old dim - verify the schema has 1024-dim vectors
    existing_table = db_module.get_db().open_table(config.LANCE_TABLE_NAME)
    vec_field = existing_table.schema.field("vector")
    assert vec_field.type.list_size == 1024  # old dim, mismatch with config's 384

    # After wipe + recreate, new table should have 384
    db_module.wipe_table()
    new_table = db_module.get_or_create_table()
    new_vec_field = new_table.schema.field("vector")
    assert new_vec_field.type.list_size == 384


def test_wipe_table(tmp_path, monkeypatch):
    import config
    import core.db as db_module
    lance_dir = tmp_path / "lancedb"
    lance_dir.mkdir()
    monkeypatch.setattr(config, "LANCEDB_DIR", lance_dir)
    monkeypatch.setattr(db_module, "LANCEDB_DIR", lance_dir)
    monkeypatch.setattr(db_module, "_db", None)
    monkeypatch.setattr(db_module, "_migrated", False)

    table = db_module.get_or_create_table()
    assert config.LANCE_TABLE_NAME in db_module.get_db().table_names()

    db_module.wipe_table()
    assert config.LANCE_TABLE_NAME not in db_module.get_db().table_names()

    table2 = db_module.get_or_create_table()
    assert config.LANCE_TABLE_NAME in db_module.get_db().table_names()
