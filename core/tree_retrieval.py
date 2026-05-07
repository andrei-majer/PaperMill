"""Tree-based retrieval — LLM reasons over document structure to find relevant sections.

Returns results in the same format as vector retrieval (core/retrieval.py) so
the generation pipeline works unchanged.
"""

import json
import logging

import pdfplumber

import config
from core.generation import _generate, _get_model
from core.tree_index import load_tree_index, list_tree_indexed_sources
from core.scanner import scan_text

logger = logging.getLogger(__name__)


def _format_tree_for_prompt(tree: dict, indent: int = 0) -> str:
    """Format a tree structure as readable text for the LLM prompt."""
    lines = []
    for section in tree.get("sections", []):
        _format_section(section, lines, indent)
    return "\n".join(lines)


def _format_section(section: dict, lines: list[str], indent: int = 0) -> None:
    """Recursively format a section and its children."""
    prefix = "  " * indent
    page_range = f"pp. {section.get('start_page', '?')}-{section.get('end_page', '?')}"
    summary = section.get("summary", "")
    title = section.get("title", "Untitled")

    line = f"{prefix}- {title} ({page_range})"
    if summary:
        line += f" — {summary}"
    lines.append(line)

    for child in section.get("children", []):
        _format_section(child, lines, indent + 1)


def _extract_pages_text(pdf_path: str, start_page: int, end_page: int) -> str:
    """Extract text from a range of pages in a PDF."""
    path = config.PDF_DIR / pdf_path
    if not path.exists():
        logger.warning("PDF not found for page extraction: %s", pdf_path)
        return ""

    texts = []
    with pdfplumber.open(str(path)) as pdf:
        for i in range(start_page - 1, min(end_page, len(pdf.pages))):
            text = pdf.pages[i].extract_text() or ""
            texts.append(text)
    return "\n\n".join(texts)


def _build_section_path(section: dict, parent_path: str = "") -> str:
    """Build a breadcrumb path like 'Methods > Data Collection'."""
    title = section.get("title", "Untitled")
    if parent_path:
        return f"{parent_path} > {title}"
    return title


def tree_search(
    query: str,
    source_filter: str | None = None,
) -> list[dict]:
    """Search for relevant content using tree-based retrieval.

    Uses LLM reasoning over document structure to select relevant sections,
    then fetches page text for those sections.

    Args:
        query: Natural language query.
        source_filter: Optional PDF filename to restrict search to.

    Returns:
        List of dicts matching the vector search format:
        {text, source_pdf, page_start, page_end, section_hint, chunk_index}
    """
    from core.prompts import TREE_RETRIEVAL_PROMPT

    # Gather tree indexes for available documents
    indexed_sources = list_tree_indexed_sources()
    if not indexed_sources:
        return []

    if source_filter:
        if source_filter not in indexed_sources:
            return []
        indexed_sources = [source_filter]

    # Load trees and format for the prompt
    structures_parts = []
    trees = {}
    for pdf_name in indexed_sources:
        tree = load_tree_index(pdf_name)
        if tree:
            trees[pdf_name] = tree
            formatted = _format_tree_for_prompt(tree)
            structures_parts.append(f"**{pdf_name}** ({tree.get('total_pages', '?')} pages):\n{formatted}")

    if not trees:
        return []

    structures_text = "\n\n".join(structures_parts)

    # Single LLM call: ask which sections are relevant
    prompt = TREE_RETRIEVAL_PROMPT.format(
        query=query,
        structures=structures_text,
    )

    system = "You are a research assistant. Follow instructions precisely and return only valid JSON."
    messages = [{"role": "user", "content": prompt}]
    response_text, _stats = _generate(_get_model("draft"), system, messages, max_tokens=2048)

    # Parse the LLM's selection
    selections = _parse_selections(response_text)
    if not selections:
        return []

    # Fetch page text for selected sections
    chunks = []
    chunk_index = 0

    for selection in selections:
        pdf_name = selection.get("document", "")
        if pdf_name not in trees:
            continue

        for section in selection.get("sections", []):
            start_page = section.get("start_page", 1)
            end_page = section.get("end_page", start_page)
            title = section.get("title", "")

            # Extract page text
            text = _extract_pages_text(pdf_name, start_page, end_page)
            if not text.strip():
                continue

            # Build section path hint
            section_hint = title
            if start_page != end_page:
                section_hint += f" (pp. {start_page}-{end_page})"

            chunks.append({
                "text": text,
                "source_pdf": pdf_name,
                "page_start": start_page,
                "page_end": end_page,
                "section_hint": section_hint,
                "chunk_index": chunk_index,
            })
            chunk_index += 1

    # Safety scan — same as vector retrieval
    safe_chunks = _safety_scan(chunks, query)

    return safe_chunks


def _parse_selections(response_text: str) -> list[dict]:
    """Parse the LLM's section selections from the response."""
    text = response_text.strip()

    # Strip markdown code fences
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)

    try:
        result = json.loads(text)
        return result.get("selections", [])
    except json.JSONDecodeError as e:
        logger.warning("Failed to parse tree retrieval response: %s", e)
        return []


def _safety_scan(chunks: list[dict], query: str) -> list[dict]:
    """Run retrieval-time safety scan on tree-retrieved chunks.

    Reuses the same regex-only scan as vector retrieval.
    """
    from core.scanner import load_scan_history, load_rules_version

    try:
        history = load_scan_history()
        current_version = load_rules_version()
        up_to_date_sources = {
            v["filename"] for v in history.values()
            if v.get("pattern_version") == current_version and v.get("result") == "passed"
        }
    except Exception:
        up_to_date_sources = set()

    safe = []
    for chunk in chunks:
        if chunk["source_pdf"] in up_to_date_sources:
            safe.append(chunk)
            continue

        result = scan_text(
            chunk["text"],
            source=chunk["source_pdf"],
            location=f"tree_chunk:{chunk['chunk_index']}",
            regex_only=True,
            scope="document",
        )
        if result.threats:
            logger.warning(
                "Tree retrieval flag: %s pp.%d-%d — %s",
                chunk["source_pdf"],
                chunk["page_start"],
                chunk["page_end"],
                [t.pattern_name for t in result.threats],
            )
        else:
            safe.append(chunk)

    return safe
