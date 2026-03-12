"""LanceDB connection and table helpers."""

import lancedb
from lancedb.pydantic import LanceModel, Vector
from config import LANCEDB_DIR, LANCE_TABLE_NAME, EMBEDDING_DIM

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
    table.update(
        where=f"source_pdf = '{source_pdf}' AND chunk_index = {chunk_index}",
        values={"safety_flag": "flagged"},
    )


def delete_source(source_name: str) -> int:
    """Delete all chunks from a given source PDF. Returns count deleted."""
    table = get_or_create_table()
    df = table.to_pandas()
    mask = df["source_pdf"] == source_name
    count = int(mask.sum())
    if count == 0:
        return 0
    remaining = df[~mask]
    # Recreate the table with remaining rows
    db = get_db()
    db.drop_table(LANCE_TABLE_NAME)
    if remaining.empty:
        db.create_table(LANCE_TABLE_NAME, schema=ChunkRecord)
    else:
        db.create_table(LANCE_TABLE_NAME, data=remaining)
    return count
