"""Tree-based document indexing — builds hierarchical structure from PDFs via LLM.

Adapted from VectifyAI/PageIndex (MIT License).
Uses the existing PaperMill LLM dispatch instead of litellm.
"""

import json
import logging
from pathlib import Path

import pdfplumber

import config
from core.generation import _generate, _get_model

logger = logging.getLogger(__name__)

# Maximum pages to check for a TOC
_TOC_SCAN_PAGES = 10
# Sections larger than this get subdivided
_SUBDIVISION_THRESHOLD = 15


def _tree_index_path(pdf_name: str) -> Path:
    """Return the path to the tree index JSON for a given PDF filename."""
    return config.TREE_INDEX_DIR / f"{pdf_name}_structure.json"


def has_tree_index(pdf_name: str) -> bool:
    """Check if a tree index exists for the given PDF."""
    return _tree_index_path(pdf_name).exists()


def load_tree_index(pdf_name: str) -> dict | None:
    """Load a tree index from disk. Returns None if not found."""
    path = _tree_index_path(pdf_name)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("Failed to load tree index for %s: %s", pdf_name, e)
        return None


def delete_tree_index(pdf_name: str) -> None:
    """Delete the tree index for a given PDF."""
    path = _tree_index_path(pdf_name)
    if path.exists():
        path.unlink()
        logger.info("Deleted tree index for %s", pdf_name)


def estimate_llm_calls(pdf_path: Path) -> int:
    """Estimate the number of LLM calls needed to build a tree index.

    Returns approximate count: 1 for TOC detection + 1 for structure generation
    + 1 per large section that needs subdivision.
    """
    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            total_pages = len(pdf.pages)
    except Exception:
        return 3  # safe default

    # Base: TOC check + structure generation
    calls = 2
    # Estimate subdivisions: roughly 1 per 15 pages beyond the first 15
    if total_pages > _SUBDIVISION_THRESHOLD:
        calls += (total_pages - _SUBDIVISION_THRESHOLD) // _SUBDIVISION_THRESHOLD + 1
    return calls


def _extract_page_texts(pdf_path: Path) -> list[dict]:
    """Extract text from each page of a PDF.

    Returns list of {page_num: int, text: str}.
    """
    pages = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for i, page in enumerate(pdf.pages, 1):
            text = page.extract_text() or ""
            pages.append({"page_num": i, "text": text})
    return pages


def _call_llm(prompt: str) -> str:
    """Send a prompt to the configured LLM and return the response text."""
    from core.prompts import get_prompt
    system = "You are a document analysis assistant. Follow instructions precisely and return only valid JSON."
    messages = [{"role": "user", "content": prompt}]
    text, _stats = _generate(_get_model("draft"), system, messages, max_tokens=4096)
    return text.strip()


def _parse_json_response(text: str) -> dict:
    """Parse a JSON response from the LLM, handling common formatting issues."""
    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last lines (fences)
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)
    return json.loads(text)


def _detect_toc(pages: list[dict]) -> dict | None:
    """Check the first pages for a table of contents via LLM.

    Returns parsed TOC structure or None if no TOC found.
    """
    from core.prompts import TREE_TOC_DETECTION_PROMPT

    scan_pages = pages[:_TOC_SCAN_PAGES]
    pages_text = "\n\n".join(
        f"--- Page {p['page_num']} ---\n{p['text'][:2000]}"
        for p in scan_pages
    )

    prompt = TREE_TOC_DETECTION_PROMPT.format(pages_text=pages_text)
    response = _call_llm(prompt)

    try:
        result = _parse_json_response(response)
        if result.get("has_toc") and result.get("sections"):
            return result
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning("Failed to parse TOC detection response: %s", e)

    return None


def _generate_structure(pages: list[dict]) -> list[dict]:
    """Generate a hierarchical structure for the document via LLM."""
    from core.prompts import TREE_STRUCTURE_GENERATION_PROMPT

    # Summarize pages for the LLM (truncate long pages to keep within context)
    pages_text = "\n\n".join(
        f"--- Page {p['page_num']} ---\n{p['text'][:1000]}"
        for p in pages
    )

    prompt = TREE_STRUCTURE_GENERATION_PROMPT.format(
        total_pages=len(pages),
        pages_text=pages_text,
    )
    response = _call_llm(prompt)

    try:
        result = _parse_json_response(response)
        return result.get("sections", [])
    except (json.JSONDecodeError, KeyError) as e:
        logger.error("Failed to parse structure generation response: %s", e)
        # Fallback: single-node tree covering the whole document
        return [{
            "title": "Full Document",
            "summary": "Unable to determine document structure.",
            "start_page": 1,
            "end_page": len(pages),
            "children": [],
        }]


def _subdivide_section(section: dict, pages: list[dict]) -> list[dict]:
    """Subdivide a large section into children via LLM."""
    from core.prompts import TREE_SECTION_SUBDIVISION_PROMPT

    start = section["start_page"]
    end = section["end_page"]
    page_count = end - start + 1

    # Get page texts for this section
    section_pages = [p for p in pages if start <= p["page_num"] <= end]
    pages_text = "\n\n".join(
        f"--- Page {p['page_num']} ---\n{p['text'][:1000]}"
        for p in section_pages
    )

    prompt = TREE_SECTION_SUBDIVISION_PROMPT.format(
        section_title=section["title"],
        start_page=start,
        end_page=end,
        page_count=page_count,
        pages_text=pages_text,
    )
    response = _call_llm(prompt)

    try:
        result = _parse_json_response(response)
        return result.get("subsections", [])
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning("Failed to subdivide section '%s': %s", section["title"], e)
        return []


def _recursively_subdivide(sections: list[dict], pages: list[dict]) -> list[dict]:
    """Walk the tree and subdivide any section exceeding the page threshold."""
    result = []
    for section in sections:
        page_span = section.get("end_page", 0) - section.get("start_page", 0) + 1

        # Recursively process existing children first
        if section.get("children"):
            section["children"] = _recursively_subdivide(section["children"], pages)
        elif page_span > _SUBDIVISION_THRESHOLD:
            # No children but too large — subdivide
            children = _subdivide_section(section, pages)
            if children:
                section["children"] = children

        result.append(section)
    return result


def build_tree_index(pdf_path: Path, progress_callback=None) -> dict:
    """Build a hierarchical tree index for a PDF document.

    Args:
        pdf_path: Path to the PDF file.
        progress_callback: Optional callable(step: str) for progress updates.

    Returns:
        The tree structure dict with keys: filename, total_pages, sections.

    Raises:
        FileNotFoundError: If the PDF doesn't exist.
        RuntimeError: If tree building fails completely.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if progress_callback:
        progress_callback("Extracting page text...")

    pages = _extract_page_texts(pdf_path)
    if not pages:
        raise RuntimeError(f"No pages found in {pdf_path.name}")

    total_pages = len(pages)
    sections = None

    # Step 1: Check for TOC
    if progress_callback:
        progress_callback("Checking for table of contents...")

    toc_result = _detect_toc(pages)
    if toc_result and toc_result.get("sections"):
        sections = toc_result["sections"]
        logger.info("Found TOC in %s with %d sections", pdf_path.name, len(sections))

        # Fill in end_page for TOC-derived sections (TOC often only has start pages)
        _fill_end_pages(sections, total_pages)
    else:
        # Step 2: Generate structure from scratch
        if progress_callback:
            progress_callback("Generating document structure...")

        sections = _generate_structure(pages)

    # Step 3: Subdivide large sections
    if progress_callback:
        progress_callback("Refining large sections...")

    sections = _recursively_subdivide(sections, pages)

    # Build the final tree
    tree = {
        "filename": pdf_path.name,
        "total_pages": total_pages,
        "sections": sections,
    }

    # Save to disk
    out_path = _tree_index_path(pdf_path.name)
    out_path.write_text(json.dumps(tree, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Saved tree index for %s (%d sections)", pdf_path.name, len(sections))

    return tree


def _fill_end_pages(sections: list[dict], total_pages: int) -> None:
    """Fill in missing end_page values for TOC-derived sections.

    TOC entries often only have start_page. This infers end_page from the
    next sibling's start_page.
    """
    for i, section in enumerate(sections):
        if "end_page" not in section or not section["end_page"]:
            if i + 1 < len(sections):
                section["end_page"] = sections[i + 1].get("start_page", total_pages) - 1
            else:
                section["end_page"] = total_pages

        # Ensure children have end_page values too
        if section.get("children"):
            parent_end = section["end_page"]
            _fill_end_pages(section["children"], parent_end)


def list_tree_indexed_sources() -> list[str]:
    """Return list of PDF filenames that have tree indexes."""
    if not config.TREE_INDEX_DIR.exists():
        return []
    return [
        p.name.removesuffix("_structure.json")
        for p in config.TREE_INDEX_DIR.glob("*_structure.json")
    ]
