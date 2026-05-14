"""Bibliography management — store, fetch, and format academic references."""

import json
import re
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path

from config import DATA_DIR

REFERENCES_PATH = DATA_DIR / "references.json"


def _load_refs() -> list[dict]:
    if REFERENCES_PATH.exists():
        return json.loads(REFERENCES_PATH.read_text(encoding="utf-8"))
    return []


def _save_refs(refs: list[dict]) -> None:
    REFERENCES_PATH.write_text(
        json.dumps(refs, indent=2, ensure_ascii=False), encoding="utf-8"
    )


_DOI_PATTERN = re.compile(r"^10\.\d{4,}/\S+$")
_CROSSREF_RETRY_DELAYS = (1, 2, 4)  # seconds between attempts


def fetch_doi_metadata(doi: str) -> dict:
    """Fetch citation metadata from CrossRef API using a DOI.

    Returns dict with: doi, title, authors, year, journal, volume, issue, pages, url.
    Validates DOI format before making any HTTP call.
    Retries up to 3 times with exponential backoff (1 s / 2 s / 4 s).
    """
    doi = doi.strip().removeprefix("https://doi.org/").removeprefix("http://doi.org/")

    if not _DOI_PATTERN.match(doi):
        raise ValueError(
            f"Invalid DOI format: {doi!r}. Expected format: 10.<registrant>/<suffix>"
        )

    url = f"https://api.crossref.org/works/{urllib.request.quote(doi, safe='')}"
    req = urllib.request.Request(url, headers={
        "User-Agent": "PaperAssistant/1.0 (mailto:noreply@example.com)",
        "Accept": "application/json",
    })

    last_error: Exception | None = None
    for attempt, delay in enumerate((*_CROSSREF_RETRY_DELAYS, None), start=1):
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            break  # success
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code in (404, 400):
                # Non-retryable: DOI not found or bad request
                raise ValueError(
                    f"CrossRef returned HTTP {exc.code} for DOI {doi!r}: {exc.reason}"
                ) from exc
            if delay is None:
                raise RuntimeError(
                    f"CrossRef request failed after {attempt - 1} retries "
                    f"(HTTP {exc.code}): {exc.reason}"
                ) from exc
        except urllib.error.URLError as exc:
            last_error = exc
            if delay is None:
                raise RuntimeError(
                    f"CrossRef request failed after {attempt - 1} retries: {exc.reason}"
                ) from exc
        time.sleep(delay)

    item = data.get("message", {})

    # Extract authors
    authors = []
    for a in item.get("author", []):
        family = a.get("family", "")
        given = a.get("given", "")
        if family:
            authors.append(f"{family}, {given[0]}." if given else family)

    # Extract year
    year = None
    for date_field in ("published-print", "published-online", "issued", "created"):
        parts = item.get(date_field, {}).get("date-parts", [[]])
        if parts and parts[0] and parts[0][0]:
            year = parts[0][0]
            break

    # Extract title
    titles = item.get("title", [])
    title = titles[0] if titles else ""

    # Extract journal
    journals = item.get("container-title", [])
    journal = journals[0] if journals else ""

    return {
        "doi": doi,
        "title": title,
        "authors": authors,
        "year": year,
        "journal": journal,
        "volume": item.get("volume", ""),
        "issue": item.get("issue", ""),
        "pages": item.get("page", ""),
        "url": f"https://doi.org/{doi}",
        "type": item.get("type", ""),
    }


def add_reference(doi: str = "", manual: dict | None = None) -> dict:
    """Add a reference by DOI (auto-fetch) or manually.

    Args:
        doi: DOI string to fetch metadata for.
        manual: Dict with keys: title, authors (list), year, journal, doi, url.

    Returns the added reference entry.
    """
    refs = _load_refs()

    if doi:
        doi = doi.strip().removeprefix("https://doi.org/").removeprefix("http://doi.org/")
        # Check for duplicates
        if any(r.get("doi", "").lower() == doi.lower() for r in refs):
            raise ValueError(f"Reference with DOI {doi} already exists.")
        entry = fetch_doi_metadata(doi)
    elif manual:
        entry = {
            "doi": manual.get("doi", ""),
            "title": manual.get("title", ""),
            "authors": manual.get("authors", []),
            "year": manual.get("year"),
            "journal": manual.get("journal", ""),
            "volume": manual.get("volume", ""),
            "issue": manual.get("issue", ""),
            "pages": manual.get("pages", ""),
            "url": manual.get("url", ""),
            "type": manual.get("type", ""),
        }
    else:
        raise ValueError("Provide either a DOI or manual entry.")

    # Generate citation key: AuthorYear
    if entry["authors"]:
        first_author_family = entry["authors"][0].split(",")[0].strip()
    else:
        first_author_family = "Unknown"
    key = f"{first_author_family}{entry.get('year', '')}"
    # Deduplicate key: try 'b'..'z' first, then numeric suffixes _2, _3, ...
    existing_keys = {r.get("key", "") for r in refs}
    if key in existing_keys:
        # Try letter suffixes b–z
        candidate = key
        for letter in "bcdefghijklmnopqrstuvwxyz":
            candidate = f"{key}{letter}"
            if candidate not in existing_keys:
                break
        else:
            # All letters exhausted — fall back to numeric suffixes
            n = 2
            while True:
                candidate = f"{key}_{n}"
                if candidate not in existing_keys:
                    break
                n += 1
        key = candidate

    entry["key"] = key
    entry["added_at"] = datetime.now(timezone.utc).isoformat()

    refs.append(entry)
    _save_refs(refs)
    return entry


def remove_reference(key_or_doi: str) -> bool:
    """Remove a reference by citation key or DOI. Returns True if found and removed."""
    refs = _load_refs()
    new_refs = [
        r for r in refs
        if r.get("key", "") != key_or_doi and r.get("doi", "") != key_or_doi
    ]
    if len(new_refs) == len(refs):
        return False
    _save_refs(new_refs)
    return True


def list_references() -> list[dict]:
    """Return all bibliography entries sorted by citation key."""
    refs = _load_refs()
    return sorted(refs, key=lambda r: r.get("key", ""))


def format_apa(ref: dict) -> str:
    """Format a reference in APA style."""
    authors_str = ", ".join(ref.get("authors", ["Unknown"]))
    year = ref.get("year", "n.d.")
    title = ref.get("title", "Untitled")
    journal = ref.get("journal", "")
    vol = ref.get("volume", "")
    issue = ref.get("issue", "")
    pages = ref.get("pages", "")
    doi = ref.get("doi", "")

    parts = [f"{authors_str} ({year}). {title}."]
    if journal:
        j = f"*{journal}*"
        if vol:
            j += f", *{vol}*"
        if issue:
            j += f"({issue})"
        if pages:
            j += f", {pages}"
        parts.append(f"{j}.")
    if doi:
        parts.append(f"https://doi.org/{doi}")

    return " ".join(parts)


def format_bibliography() -> str:
    """Format the full bibliography in APA style for inclusion in prompts."""
    refs = list_references()
    if not refs:
        return "No references in bibliography."
    lines = [format_apa(r) for r in refs]
    return "\n\n".join(lines)


def format_refs_for_prompt() -> str:
    """Format references as a compact list for inclusion in generation prompts."""
    refs = list_references()
    if not refs:
        return ""
    lines = []
    for r in refs:
        authors = ", ".join(r.get("authors", [])[:3])
        if len(r.get("authors", [])) > 3:
            authors += " et al."
        year = r.get("year", "n.d.")
        title = r.get("title", "")
        lines.append(f"[{r['key']}] {authors} ({year}). {title}.")
    return "\n".join(lines)
