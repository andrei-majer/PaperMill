"""LanceDB connection and table helpers."""

import lancedb
from lancedb.pydantic import LanceModel, Vector
import config
from config import LANCEDB_DIR, LANCE_TABLE_NAME


def _sql_escape(s: str) -> str:
    """Escape single quotes in a string for use in SQL WHERE clauses."""
    return s.replace("'", "''")

_db: lancedb.DBConnection | None = None
_migrated: bool = False


def make_chunk_record(dim: int) -> type:
    """Factory that returns a LanceModel subclass with the given vector dimension."""
    class _ChunkRecord(LanceModel):
        id: str
        vector: Vector(dim)
        text: str
        source_pdf: str
        page_start: int
        page_end: int
        chunk_index: int
        section_hint: str = ""
        ingested_at: str
        safety_flag: str = ""
        source_type: str = "pdf"
    return _ChunkRecord


ChunkRecord = make_chunk_record(config.EMBEDDING_DIM)


def get_db() -> lancedb.DBConnection:
    """Return a (cached) connection to the local LanceDB instance."""
    global _db
    if _db is None:
        _db = lancedb.connect(str(LANCEDB_DIR))
    return _db


def _migrate_table(table: lancedb.table.Table) -> lancedb.table.Table:
    """Add missing columns to an existing table to match ChunkRecord schema."""
    import pyarrow as pa

    existing_names = set(table.schema.names)
    expected_fields = {
        "safety_flag": pa.field("safety_flag", pa.utf8()),
        "source_type": pa.field("source_type", pa.utf8()),
    }
    defaults = {"safety_flag": "", "source_type": "pdf"}

    missing = {k: v for k, v in expected_fields.items() if k not in existing_names}
    if not missing:
        return table

    # Add missing columns via update with default values
    df = table.to_pandas()
    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default

    # Recreate table with full schema
    db = get_db()
    db.drop_table(LANCE_TABLE_NAME)
    return db.create_table(LANCE_TABLE_NAME, data=df)


def get_or_create_table() -> lancedb.table.Table:
    """Open the chunks table, creating it if it doesn't exist.

    Automatically migrates old tables that are missing newer columns.
    """
    global ChunkRecord
    ChunkRecord = make_chunk_record(config.EMBEDDING_DIM)
    db = get_db()
    existing = db.list_tables().tables
    if LANCE_TABLE_NAME in existing:
        table = db.open_table(LANCE_TABLE_NAME)
        global _migrated
        if not _migrated:
            table = _migrate_table(table)
            _migrated = True
        return table
    return db.create_table(LANCE_TABLE_NAME, schema=ChunkRecord)


def wipe_table() -> None:
    """Drop the chunks table entirely and reset migration state."""
    global _migrated
    db = get_db()
    if LANCE_TABLE_NAME in db.list_tables().tables:
        db.drop_table(LANCE_TABLE_NAME)
    _migrated = False


def list_sources() -> list[str]:
    """Return a sorted list of unique source_pdf values in the table."""
    table = get_or_create_table()
    try:
        df = table.to_pandas()
        if df.empty:
            return []
        return sorted(df["source_pdf"].unique().tolist())
    except Exception:
        return []


def tag_chunk_flagged(source_pdf: str, chunk_index: int) -> None:
    """Mark a specific chunk as flagged by the content scanner."""
    table = get_or_create_table()
    if "safety_flag" not in table.schema.names:
        return  # Legacy table without safety_flag column — skip silently
    table.update(
        where=f"source_pdf = '{_sql_escape(source_pdf)}' AND chunk_index = {chunk_index}",
        values={"safety_flag": "flagged"},
    )


def delete_source(source_name: str) -> int:
    """Delete all chunks from a given source PDF. Returns count deleted."""
    table = get_or_create_table()
    escaped = _sql_escape(source_name)
    count_before = table.count_rows(filter=f"source_pdf = '{escaped}'")
    if count_before == 0:
        return 0
    table.delete(where=f"source_pdf = '{escaped}'")
    return count_before
