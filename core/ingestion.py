"""PDF ingestion pipeline: parse → chunk → embed → store."""

import hashlib
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

import pdfplumber

import config
from config import CHUNK_SIZE_TOKENS, CHUNK_OVERLAP_TOKENS, save_settings
from core.embedder import embed_passages
from core.db import get_or_create_table, ChunkRecord
from core.scanner import (
    compute_file_hash,
    load_scan_history,
    update_scan_history,
    load_allowlist,
    load_rules_version,
    scan_structure,
    scan_metadata,
    extract_pdf_metadata,
    scan_text,
    generate_report,
    quarantine_file,
    ContentBlockedError,
)

logger = logging.getLogger(__name__)

_DOI_RE = re.compile(r"\b(10\.\d{4,}/\S+)")


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~0.75 tokens per word."""
    return int(len(text.split()) * 1.33)


def _extract_pages(pdf_path: Path) -> list[dict]:
    """Extract text from each page with font-size metadata for heading detection.

    Returns list of dicts: {page_num, text, blocks} where blocks have font info.
    """
    pages = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            chars = page.chars or []
            page_text_parts = []
            page_blocks = []

            # Group consecutive characters into text runs by font
            current_text = []
            current_size = None
            current_bold = False

            for ch in chars:
                size = round(ch.get("size", 12), 1)
                fontname = ch.get("fontname", "").lower()
                bold = "bold" in fontname or "black" in fontname
                text_char = ch.get("text", "")

                if size != current_size or bold != current_bold:
                    # Flush previous run
                    run_text = "".join(current_text).strip()
                    if run_text and current_size is not None:
                        page_blocks.append({
                            "text": run_text,
                            "font_size": current_size,
                            "bold": current_bold,
                        })
                        page_text_parts.append(run_text)
                    current_text = [text_char]
                    current_size = size
                    current_bold = bold
                else:
                    current_text.append(text_char)

            # Flush last run
            run_text = "".join(current_text).strip()
            if run_text and current_size is not None:
                page_blocks.append({
                    "text": run_text,
                    "font_size": current_size,
                    "bold": current_bold,
                })
                page_text_parts.append(run_text)

            pages.append({
                "page_num": page_num,
                "text": " ".join(page_text_parts),
                "blocks": page_blocks,
            })
    return pages


def _detect_heading(block: dict, median_size: float) -> bool:
    """Heuristic: block is a heading if font is significantly larger, bold, or ALL CAPS."""
    text = block["text"].strip()
    if not text or len(text) > 200:
        return False
    if block["font_size"] > median_size * 1.15:
        return True
    if block["bold"] and len(text) < 100:
        return True
    if text.isupper() and len(text) > 3 and len(text) < 100:
        return True
    if re.match(r"^\d+(\.\d+)*\s+[A-Z]", text):
        return True
    return False


def chunk_pdf(pdf_path: Path) -> list[dict]:
    """Parse a PDF and produce heading-aware, overlapping chunks.

    Returns list of dicts with keys: text, page_start, page_end, chunk_index, section_hint.
    """
    pages = _extract_pages(pdf_path)
    if not pages:
        return []

    # Compute median font size across all blocks
    all_sizes = [b["font_size"] for p in pages for b in p["blocks"]]
    if not all_sizes:
        return []
    all_sizes.sort()
    median_size = all_sizes[len(all_sizes) // 2]

    # Build sections: split on detected headings
    sections: list[dict] = []  # {heading, text, page_start, page_end}
    current_heading = ""
    current_text_parts: list[str] = []
    current_page_start = 1
    current_page_end = 1

    for page in pages:
        for block in page["blocks"]:
            if _detect_heading(block, median_size):
                # Save previous section
                if current_text_parts:
                    sections.append({
                        "heading": current_heading,
                        "text": " ".join(current_text_parts),
                        "page_start": current_page_start,
                        "page_end": current_page_end,
                    })
                current_heading = block["text"].strip()
                current_text_parts = []
                current_page_start = page["page_num"]
                current_page_end = page["page_num"]
            else:
                current_text_parts.append(block["text"])
                current_page_end = page["page_num"]

    # Don't forget the last section
    if current_text_parts:
        sections.append({
            "heading": current_heading,
            "text": " ".join(current_text_parts),
            "page_start": current_page_start,
            "page_end": current_page_end,
        })

    # Chunk within each section
    chunks = []
    chunk_index = 0
    for sec in sections:
        words = sec["text"].split()
        # Convert token limits to approximate word counts
        chunk_words = int(CHUNK_SIZE_TOKENS / 1.33)
        overlap_words = int(CHUNK_OVERLAP_TOKENS / 1.33)
        step = max(chunk_words - overlap_words, 1)

        if not words:
            continue

        start = 0
        while start < len(words):
            end = min(start + chunk_words, len(words))
            chunk_text = " ".join(words[start:end])

            # Prepend section heading for self-describing context
            if sec["heading"]:
                chunk_text = f"[{sec['heading']}]\n{chunk_text}"

            chunks.append({
                "text": chunk_text,
                "page_start": sec["page_start"],
                "page_end": sec["page_end"],
                "chunk_index": chunk_index,
                "section_hint": sec["heading"],
            })
            chunk_index += 1

            if end >= len(words):
                break
            start += step

    return chunks


def _chunk_id(source_pdf: str, chunk_index: int) -> str:
    """Deterministic ID for a chunk."""
    raw = f"{source_pdf}:{chunk_index}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _extract_doi_from_metadata(meta: dict) -> str | None:
    """Look for a DOI in PDF metadata fields.

    Checks direct 'doi' key first, then scans 'Subject' and 'Keywords'
    (and any XMP-style keys) for a DOI pattern.
    """
    # Direct key (case-insensitive search)
    for key in meta:
        if key.lower() == "doi":
            value = str(meta[key]).strip()
            if value:
                # Strip common URL prefixes
                value = value.removeprefix("https://doi.org/").removeprefix("http://doi.org/")
                if _DOI_RE.search(value):
                    return _DOI_RE.search(value).group(1)

    # Scan string values of common fields and any XMP-style keys
    scan_keys = {"subject", "keywords"}
    for key, value in meta.items():
        if key.lower() in scan_keys or key.lower().startswith("xmp"):
            match = _DOI_RE.search(str(value))
            if match:
                return match.group(1)

    return None


def _try_add_doi_to_bibliography(doi: str, filename: str) -> None:
    """Attempt to fetch DOI metadata and add it to the bibliography.

    Failures are caught and logged as warnings so they never block ingestion.
    Silently skips if the DOI is already present in the bibliography.
    """
    try:
        from core.bibliography import add_reference
        add_reference(doi=doi)
        logger.info("Auto-extracted DOI %s from %s and added to bibliography", doi, filename)
    except ValueError as exc:
        # Covers duplicate DOI, invalid format, CrossRef 404/400
        msg = str(exc)
        if "already exists" in msg:
            logger.debug("DOI %s from %s already in bibliography — skipping", doi, filename)
        else:
            logger.warning("Could not add DOI %s from %s to bibliography: %s", doi, filename, exc)
    except Exception as exc:  # network errors, etc.
        logger.warning("Failed to fetch metadata for DOI %s from %s: %s", doi, filename, exc)


def is_already_ingested(pdf_path: Path) -> bool:
    """Check if a PDF has already been ingested by looking for its name in the DB."""
    table = get_or_create_table()
    try:
        df = table.to_pandas()
        if df.empty:
            return False
        return pdf_path.name in df["source_pdf"].values
    except Exception:
        return False


def ingest_pdf(pdf_path: Path, force: bool = False) -> int:
    """Full pipeline: parse PDF → scan → chunk → embed → store in LanceDB.

    Raises ContentBlockedError if the scanner blocks the file (unless SCANNER_DRY_RUN).
    Returns number of chunks ingested.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if not force and is_already_ingested(pdf_path):
        return 0  # Already ingested

    # ── Scanner gate ─────────────────────────────────────────────────────
    file_hash = compute_file_hash(pdf_path)
    chunks = None  # will be set during scan or after
    doi_from_metadata: str | None = None  # tracks DOI found in PDF metadata

    # Check allowlist — skip scanning if file is explicitly allowed
    allowlist = load_allowlist()
    if file_hash in allowlist:
        pass  # skip scanning, fall through to ingest
    else:
        # Check scan history — skip re-scan if already scanned with current rules
        history = load_scan_history()
        current_version = load_rules_version()
        cached = history.get(file_hash, {})
        skip_scan = (
            cached.get("pattern_version") == current_version
            and cached.get("result") == "passed"
        )

        if not skip_scan:
            scan_results = []

            # 1. Structural scan
            struct_result = scan_structure(pdf_path)
            scan_results.append(struct_result)

            # 2. Metadata scan
            meta = extract_pdf_metadata(pdf_path)
            meta_result = scan_metadata(meta, pdf_path.name)
            scan_results.append(meta_result)

            # Extract DOI from metadata while we have it
            doi_from_metadata = _extract_doi_from_metadata(meta)

            # 3. Chunk content scan — reuse chunks for embedding later
            chunks = chunk_pdf(pdf_path)
            llm_escalations_remaining = config.SCANNER_MAX_LLM_ESCALATIONS
            for chunk in chunks:
                chunk_result = scan_text(
                    chunk["text"],
                    source=pdf_path.name,
                    location=f"chunk:{chunk['chunk_index']}",
                    scope="document",
                    _llm_budget=llm_escalations_remaining,
                )
                llm_escalations_remaining -= chunk_result.llm_escalations
                scan_results.append(chunk_result)

            overall_safe = all(r.is_safe for r in scan_results)

            if not overall_safe:
                report_path = generate_report(scan_results, pdf_path.name, file_hash)
                update_scan_history(file_hash, pdf_path.name, "blocked", current_version, str(report_path))

                if not config.SCANNER_DRY_RUN:
                    quarantine_file(pdf_path)
                    raise ContentBlockedError(
                        f"PDF blocked by content scanner: {pdf_path.name}",
                        report_path=report_path,
                    )
            else:
                update_scan_history(file_hash, pdf_path.name, "passed", current_version)

        else:
            # Scan was skipped (cached pass) — still extract metadata for DOI
            meta = extract_pdf_metadata(pdf_path)
            doi_from_metadata = _extract_doi_from_metadata(meta)

    # ── Embed + store pipeline ────────────────────────────────────────────

    # Reuse chunks from scan phase, or parse now if scan was skipped
    if chunks is None:
        chunks = chunk_pdf(pdf_path)
    if not chunks:
        return 0

    # ── DOI auto-extraction ───────────────────────────────────────────────

    if doi_from_metadata:
        _try_add_doi_to_bibliography(doi_from_metadata, pdf_path.name)
    else:
        # Fallback: scan first-page text for a DOI pattern
        first_page_text = chunks[0]["text"]
        match = _DOI_RE.search(first_page_text)
        if match:
            _try_add_doi_to_bibliography(match.group(1), pdf_path.name)

    # Embed
    texts = [c["text"] for c in chunks]
    vectors = embed_passages(texts)

    # Build records
    now = datetime.now(timezone.utc).isoformat()
    records = []
    for chunk, vector in zip(chunks, vectors):
        records.append({
            "id": _chunk_id(pdf_path.name, chunk["chunk_index"]),
            "vector": vector,
            "text": chunk["text"],
            "source_pdf": pdf_path.name,
            "page_start": chunk["page_start"],
            "page_end": chunk["page_end"],
            "chunk_index": chunk["chunk_index"],
            "section_hint": chunk["section_hint"],
            "ingested_at": now,
            "safety_flag": "",
            "source_type": "pdf",
        })

    # Store
    table = get_or_create_table()
    table.add(records)

    # Track which embedding model built these vectors
    _model_key = f"{config.EMBEDDING_PROVIDER}:{config.EMBEDDING_MODEL}"
    if config.LAST_EMBEDDING_MODEL != _model_key:
        save_settings({"last_embedding_model": _model_key})
        config.LAST_EMBEDDING_MODEL = _model_key

    return len(records)


def reingest_all(progress_callback=None) -> dict:
    """Wipe vector DB and re-ingest all PDFs from data/pdfs/.

    Args:
        progress_callback: Optional callable(current, total, filename) for progress.

    Returns dict with keys: ingested, failed, total_chunks.
    """
    from core.db import wipe_table
    from core.embedder import detect_and_save_dim

    # Detect and persist embedding dimension before wiping
    detect_and_save_dim()

    wipe_table()

    pdf_files = sorted(config.PDF_DIR.glob("*.pdf"))
    total = len(pdf_files)
    ingested = 0
    total_chunks = 0
    failed = []

    for i, pdf_path in enumerate(pdf_files):
        if progress_callback:
            progress_callback(i, total, pdf_path.name)
        try:
            count = ingest_pdf(pdf_path, force=True)
            if count > 0:
                ingested += 1
                total_chunks += count
        except Exception as e:
            failed.append(f"{pdf_path.name}: {e}")
            logger.warning("Re-ingestion failed for %s: %s", pdf_path.name, e)

    # Update last_embedding_model to current
    model_key = f"{config.EMBEDDING_PROVIDER}:{config.EMBEDDING_MODEL}"
    save_settings({"last_embedding_model": model_key})
    config.LAST_EMBEDDING_MODEL = model_key

    return {"ingested": ingested, "failed": failed, "total_chunks": total_chunks}
