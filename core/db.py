"""LanceDB connection and table helpers."""

import lancedb
from lancedb.pydantic import LanceModel, Vector
from config import LANCEDB_DIR, LANCE_TABLE_NAME, EMBEDDING_DIM


def _sql_escape(s: str) -> str:
    """Escape single quotes in a string for use in SQL WHERE clauses."""
    return s.replace("'", "''")

_db: lancedb.DBConnection | None = None


class ChunkRecord(LanceModel):
    """Schema for a document chunk stored in LanceDB."""
    id: str                          # sha256(source_pdf + chunk_index)
    vector: Vector(1024)             # jina-embeddings-v3 output
    text: str                        # chunk text
    source_pdf: str                  # filename
    page_start: int
    page_end: int
    chunk_index: int
    section_hint: str = ""           # detected heading
    ingested_at: str                 # ISO timestamp
    safety_flag: str = ""            # "flagged" if scanner blocked, "" otherwise
    source_type: str = "pdf"         # "pdf" or "image"


def get_db() -> lancedb.DBConnection:
    """Return a (cached) connection to the local LanceDB instance."""
    global _db
    if _db is None:
        _db = lancedb.connect(str(LANCEDB_DIR))
    return _db


def get_or_create_table() -> lancedb.table.Table:
    """Open the chunks table, creating it if it doesn't exist."""
    db = get_db()
    existing = db.table_names()
    if LANCE_TABLE_NAME in existing:
        return db.open_table(LANCE_TABLE_NAME)
    return db.create_table(LANCE_TABLE_NAME, schema=ChunkRecord)


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
