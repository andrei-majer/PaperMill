"""Vector search over the LanceDB chunk store."""

import logging
from datetime import datetime, timezone

import config
from core.embedder import embed_query
from core.db import get_or_create_table, tag_chunk_flagged
from core.scanner import scan_text, Threat

logger = logging.getLogger(__name__)


def _log_retrieval_flag(query: str, chunk: dict, threats: list[Threat]) -> None:
    """Append a retrieval-flag entry to the retrieval_flags.log file."""
    log_path = config.REPORTS_DIR / "retrieval_flags.log"
    now = datetime.now(timezone.utc).isoformat()
    threat_ids = ", ".join(t.pattern_name for t in threats)
    line = (
        f"[{now}] query={query!r} source={chunk.get('source_pdf', '')} "
        f"chunk_index={chunk.get('chunk_index', '')} threats=[{threat_ids}]\n"
    )
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line)


def search(
    query: str,
    top_k: int = 8,
    source_filter: str | None = None,
) -> list[dict]:
    """Search for chunks relevant to a query.

    Args:
        query: Natural language query.
        top_k: Number of results to return.
        source_filter: Optional PDF filename to restrict search to.

    Returns:
        List of dicts with keys: text, source_pdf, page_start, page_end,
        section_hint, chunk_index, _distance.
    """
    table = get_or_create_table()

    # Check if table has data
    try:
        if table.count_rows() == 0:
            return []
    except Exception:
        return []

    query_vec = embed_query(query)

    # Build search — filter out flagged chunks
    results = table.search(query_vec).limit(top_k)

    # Exclude chunks previously flagged by the scanner
    safety_filter = "safety_flag = '' OR safety_flag IS NULL"
    if source_filter:
        results = results.where(f"source_pdf = '{source_filter}' AND ({safety_filter})")
    else:
        results = results.where(safety_filter)

    df = results.to_pandas()

    if df.empty:
        return []

    chunks = []
    for _, row in df.iterrows():
        chunks.append({
            "text": row["text"],
            "source_pdf": row["source_pdf"],
            "page_start": int(row["page_start"]),
            "page_end": int(row["page_end"]),
            "section_hint": row.get("section_hint", ""),
            "chunk_index": int(row["chunk_index"]),
            "_distance": float(row.get("_distance", 0)),
        })

    # ── Retrieval-time scanner gate (regex only) ──────────────────────────
    safe_chunks = []
    for chunk in chunks:
        result = scan_text(
            chunk["text"],
            source=chunk["source_pdf"],
            location=f"chunk:{chunk['chunk_index']}",
            regex_only=True,
            scope="document",
        )
        if result.threats:
            # Tag flagged chunk in DB and log it
            try:
                tag_chunk_flagged(chunk["source_pdf"], chunk["chunk_index"])
            except Exception as e:
                logger.warning("Failed to tag chunk in DB: %s", e)
            _log_retrieval_flag(query, chunk, result.threats)
            logger.warning(
                "Retrieval-time flag: %s chunk:%s — %s",
                chunk["source_pdf"],
                chunk["chunk_index"],
                [t.pattern_name for t in result.threats],
            )
        else:
            safe_chunks.append(chunk)

    return safe_chunks
