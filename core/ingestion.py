"""PDF ingestion pipeline: parse → chunk → embed → store."""

import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path

import fitz  # PyMuPDF

import config
from config import CHUNK_SIZE_TOKENS, CHUNK_OVERLAP_TOKENS
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


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~0.75 tokens per word."""
    return int(len(text.split()) * 1.33)


def _extract_pages(pdf_path: Path) -> list[dict]:
    """Extract text from each page with font-size metadata for heading detection.

    Returns list of dicts: {page_num, text, blocks} where blocks have font info.
    """
    doc = fitz.open(str(pdf_path))
    pages = []
    for page_num, page in enumerate(doc, 1):
        blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
        page_text_parts = []
        page_blocks = []

        for block in blocks:
            if block.get("type") != 0:  # text blocks only
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "").strip()
                    if not text:
                        continue
                    font_size = span.get("size", 12)
                    flags = span.get("flags", 0)
                    is_bold = bool(flags & 2**4)  # bit 4 = bold
                    page_blocks.append({
                        "text": text,
                        "font_size": font_size,
                        "bold": is_bold,
                    })
                    page_text_parts.append(text)

        pages.append({
            "page_num": page_num,
            "text": " ".join(page_text_parts),
            "blocks": page_blocks,
        })
    doc.close()
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

            # 3. Chunk content scan (scan each chunk text)
            chunks_for_scan = chunk_pdf(pdf_path)
            for chunk in chunks_for_scan:
                chunk_result = scan_text(
                    chunk["text"],
                    source=pdf_path.name,
                    location=f"chunk:{chunk['chunk_index']}",
                    scope="document",
                )
                scan_results.append(chunk_result)

            overall_safe = all(r.is_safe for r in scan_results)
            result_str = "passed" if overall_safe else "blocked"

            if not overall_safe:
                report_path = generate_report(scan_results, pdf_path.name, file_hash)
                update_scan_history(file_hash, pdf_path.name, result_str, current_version, str(report_path))

                if not config.SCANNER_DRY_RUN:
                    quarantine_file(pdf_path)
                    raise ContentBlockedError(
                        f"PDF blocked by content scanner: {pdf_path.name}",
                        report_path=report_path,
                    )
            else:
                report_path = generate_report(scan_results, pdf_path.name, file_hash)
                update_scan_history(file_hash, pdf_path.name, result_str, current_version, str(report_path))

    # ── Embed + store pipeline ────────────────────────────────────────────

    # Parse and chunk
    chunks = chunk_pdf(pdf_path)
    if not chunks:
        return 0

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
        })

    # Store
    table = get_or_create_table()
    table.add(records)
    return len(records)
