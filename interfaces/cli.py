"""CLI REPL interface for the RAG Paper Writing Assistant."""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
from config import PDF_DIR, DATA_DIR, CHAT_HISTORY_PATH
from core.scanner import (
    load_rules, ContentBlockedError, load_scan_history,
    add_to_allowlist, remove_from_allowlist, quarantine_release, compute_file_hash,
)
from core.ingestion import ingest_pdf, is_already_ingested
from core.image_ingestion import ingest_image, ingest_images_dir, IMAGE_EXTENSIONS
from core.retrieval import search
from core.generation import chat, draft_section, rewrite_section
from core.paper_structure import (
    PAPER_OUTLINE,
    list_sections_status,
    load_section,
    get_section_title,
)
from core.db import list_sources, delete_source
from core.docexport import export_full_paper, export_section
from core.versioning import save_version, list_versions, compare_versions
from core.bibliography import add_reference, remove_reference, list_references, format_apa
from config import VERSIONS_DIR


# ── Chat history persistence ───────────────────────────────────────────────

def _save_chat_history(history: list[dict]) -> None:
    """Write chat history to CHAT_HISTORY_PATH as JSON."""
    try:
        CHAT_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        CHAT_HISTORY_PATH.write_text(
            json.dumps(history, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as exc:
        print(f"  [Warning] Could not save chat history: {exc}")


def _load_chat_history() -> list[dict]:
    """Load chat history from CHAT_HISTORY_PATH; return [] if missing or corrupt."""
    try:
        if CHAT_HISTORY_PATH.exists():
            data = json.loads(CHAT_HISTORY_PATH.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return data
    except Exception:
        pass
    return []


def print_help():
    """Print available commands."""
    help_text = """
Commands:
  /ingest [path]         Ingest a PDF (default: all in data/pdfs/)
  /ingest-images [path]  Ingest images (default: all in data/images/)
  /sources               List ingested sources
  /delete-source <name>  Delete a source from the vector DB
  /outline               Show paper outline with draft status
  /draft <id>            Draft a section (e.g. /draft ch1)
  /rewrite <id>          Rewrite a section with Opus
  /show <id>             Show a section draft
  /export [id]           Export full paper or a section to .docx
  /version [label]       Save a versioned snapshot
  /versions              List all versions
  /diff <v1> <v2>        Compare two versions (by number)
  /ref-add <DOI>         Add a reference by DOI (auto-fetches metadata)
  /refs                  List all bibliography entries
  /ref-remove <key>      Remove a reference by citation key or DOI
  /scan-rules            List active scanner rules
  /allowlist-add <path>  Add file to scanner allowlist
  /allowlist-remove <hash> Remove hash from allowlist
  /quarantine-release [hash] Release quarantined file (no args = list)
  /clear-history         Clear chat history (file + in-memory)
  /help                  Show this help
  /quit                  Exit

Free text → RAG-powered chat with citations.
"""
    print(help_text)


def cmd_ingest(args: str):
    """Ingest PDFs."""
    if args.strip():
        path = Path(args.strip())
        if not path.exists():
            print(f"File not found: {path}")
            return
        print(f"Ingesting {path.name}...")
        try:
            count = ingest_pdf(path, force=False)
            if count == 0:
                print(f"  Already ingested (use force to re-ingest).")
            else:
                print(f"  Ingested {count} chunks.")
        except ContentBlockedError as e:
            print(f"  BLOCKED: {e}")
            print(f"  Report: {e.report_path}")
    else:
        pdfs = list(PDF_DIR.glob("*.pdf"))
        if not pdfs:
            print(f"No PDFs found in {PDF_DIR}")
            return
        for pdf in pdfs:
            print(f"Ingesting {pdf.name}...")
            try:
                count = ingest_pdf(pdf, force=False)
                if count == 0:
                    print(f"  Already ingested.")
                else:
                    print(f"  Ingested {count} chunks.")
            except ContentBlockedError as e:
                print(f"  BLOCKED: {e}")
                print(f"  Report: {e.report_path}")


def cmd_ingest_images(args: str):
    """Ingest images."""
    images_dir = Path(args.strip()) if args.strip() else DATA_DIR / "images"
    if not images_dir.is_dir():
        print(f"Directory not found: {images_dir}")
        return
    print(f"Ingesting images from {images_dir}...")
    results = ingest_images_dir(images_dir)
    print(f"  Ingested: {results['ingested']}, Skipped: {results['skipped']}")
    for err in results["errors"]:
        print(f"  Error: {err}")


def cmd_sources():
    """List ingested sources."""
    sources = list_sources()
    if not sources:
        print("No sources ingested yet.")
        return
    print("Ingested sources:")
    for s in sources:
        print(f"  - {s}")


def cmd_delete_source(args: str):
    """Delete a source."""
    name = args.strip()
    if not name:
        print("Usage: /delete-source <filename>")
        return
    count = delete_source(name)
    if count == 0:
        print(f"Source '{name}' not found.")
    else:
        print(f"Deleted {count} chunks from '{name}'.")


def cmd_outline():
    """Show paper outline with status."""
    statuses = list_sections_status()
    print("\nPaper Outline:")
    print("-" * 60)
    for s in statuses:
        status_icon = {"empty": "  ", "draft": "D ", "review": "R ", "final": "F "}
        icon = status_icon.get(s["status"], "  ")
        words = f"({s['target_words']}w)" if s["target_words"] > 0 else ""
        print(f"  [{icon}] {s['id']:12s} {s['title']} {words}")
    print("-" * 60)
    print("  Legend: D=draft, R=review, F=final")


def cmd_draft(args: str):
    """Draft a section."""
    section_id = args.strip()
    if not section_id:
        print("Usage: /draft <section_id>")
        return
    title = get_section_title(section_id)
    print(f"Drafting: {title}")
    print("Searching for relevant references...")
    chunks = search(title, top_k=12)
    print(f"  Found {len(chunks)} relevant chunks.")
    print("Generating draft (this may take a moment)...")
    text, stats = draft_section(section_id, chunks)
    print(f"\n{'='*60}")
    print(text[:500] + "..." if len(text) > 500 else text)
    print(f"{'='*60}")
    print(f"Draft saved. {stats.tokens_per_sec} tok/s | {stats.completion_tokens} tokens | {stats.elapsed_sec}s")
    print(f"View full text with: /show {section_id}")


def cmd_rewrite(args: str):
    """Rewrite a section."""
    section_id = args.strip()
    if not section_id:
        print("Usage: /rewrite <section_id>")
        return
    section = load_section(section_id)
    if not section:
        print(f"No draft found for '{section_id}'. Draft it first with /draft {section_id}")
        return
    print(f"Rewriting: {section['title']} (using Opus for polish)...")
    chunks = search(section["title"], top_k=8)
    text, stats = rewrite_section(section_id, chunks=chunks)
    print(f"\n{'='*60}")
    print(text[:500] + "..." if len(text) > 500 else text)
    print(f"{'='*60}")
    print(f"Rewrite saved. {stats.tokens_per_sec} tok/s | {stats.completion_tokens} tokens | {stats.elapsed_sec}s")
    print(f"View full text with: /show {section_id}")


def cmd_show(args: str):
    """Show a section draft."""
    section_id = args.strip()
    if not section_id:
        print("Usage: /show <section_id>")
        return
    section = load_section(section_id)
    if not section:
        print(f"No draft for '{section_id}'.")
        return
    print(f"\n{'='*60}")
    print(f"Section: {section['title']}  [Status: {section['status']}]")
    print(f"Updated: {section['updated_at']}")
    print(f"{'='*60}")
    print(section["text"])
    print(f"{'='*60}")


def cmd_export(args: str):
    """Export to .docx."""
    section_id = args.strip()
    if section_id:
        path = VERSIONS_DIR / f"{section_id}.docx"
        try:
            export_section(section_id, path)
            print(f"Exported section to: {path}")
        except ValueError as e:
            print(f"Error: {e}")
    else:
        path = VERSIONS_DIR / "paper_current.docx"
        export_full_paper(path)
        print(f"Exported full paper to: {path}")


def cmd_version(args: str):
    """Save a version."""
    label = args.strip()
    entry = save_version(label)
    print(f"Saved version {entry['version']}: {entry['filename']}")
    print(f"  Sections drafted: {entry['sections_drafted']}/{entry['total_sections']}")


def cmd_versions():
    """List versions."""
    versions = list_versions()
    if not versions:
        print("No versions saved yet.")
        return
    print("\nVersions:")
    for v in versions:
        print(f"  v{v['version']}: {v['label']} ({v['timestamp'][:10]}) "
              f"- {v['sections_drafted']}/{v['total_sections']} sections - {v['filename']}")


def cmd_diff(args: str):
    """Compare two versions."""
    parts = args.strip().split()
    if len(parts) != 2:
        print("Usage: /diff <v1> <v2>  (e.g. /diff 1 2)")
        return
    try:
        v1, v2 = int(parts[0]), int(parts[1])
        diff_text = compare_versions(v1, v2)
        if not diff_text:
            print("No differences found.")
        else:
            print(diff_text)
    except ValueError as e:
        print(f"Error: {e}")


def cmd_ref_add(args: str):
    """Add a reference by DOI."""
    doi = args.strip()
    if not doi:
        print("Usage: /ref-add <DOI>")
        return
    try:
        entry = add_reference(doi=doi)
        print(f"Added [{entry['key']}]:")
        print(f"  {format_apa(entry)}")
    except Exception as e:
        print(f"Error: {e}")


def cmd_refs():
    """List all bibliography entries."""
    refs = list_references()
    if not refs:
        print("No references yet. Add one with /ref-add <DOI>")
        return
    print(f"\nBibliography ({len(refs)} entries):")
    print("-" * 60)
    for r in refs:
        print(f"  [{r['key']}] {format_apa(r)}")
    print("-" * 60)


def cmd_ref_remove(args: str):
    """Remove a reference."""
    key = args.strip()
    if not key:
        print("Usage: /ref-remove <citation-key or DOI>")
        return
    if remove_reference(key):
        print(f"Removed '{key}'.")
    else:
        print(f"Reference '{key}' not found.")


def cmd_scan_rules():
    """List active scanner rules."""
    rules = load_rules()
    print(f"\nScanner Rules ({len(rules)} active):")
    print("-" * 70)
    print(f"  {'ID':<20} {'Category':<22} {'Severity':<8} Description")
    print("-" * 70)
    for r in rules:
        print(f"  {r['id']:<20} {r['category']:<22} {r['severity']:<8} {r['description']}")


def cmd_allowlist_add(args: str):
    """Add a file to the scanner allowlist by path."""
    path = Path(args.strip())
    if not path.exists():
        print(f"File not found: {path}")
        return
    file_hash = compute_file_hash(path)
    add_to_allowlist(file_hash, path.name, "Manually allowlisted via CLI")
    print(f"Added {path.name} to allowlist (hash: {file_hash[:16]}...)")


def cmd_allowlist_remove(args: str):
    """Remove a hash from the scanner allowlist."""
    h = args.strip()
    if remove_from_allowlist(h):
        print(f"Removed {h[:16]}... from allowlist.")
    else:
        print(f"Hash not found in allowlist.")


def cmd_quarantine_release_cmd(args: str):
    """Release a file from quarantine."""
    h = args.strip()
    if not h:
        history = load_scan_history()
        blocked = {k: v for k, v in history.items() if v["result"] == "blocked"}
        if not blocked:
            print("No quarantined files.")
            return
        print("Quarantined files:")
        for k, v in blocked.items():
            print(f"  {k[:16]}... — {v['filename']} ({v.get('last_seen', 'unknown')})")
        print("\nUsage: /quarantine-release <hash>")
        return
    try:
        dest = quarantine_release(h)
        print(f"Released to {dest}. Added to allowlist. Ready for re-ingestion.")
    except (ValueError, FileNotFoundError) as e:
        print(f"Error: {e}")


def handle_chat(message: str, history: list[dict]):
    """Handle free-text chat with RAG."""
    print("Searching references...")
    chunks = search(message, top_k=8)
    print(f"  Found {len(chunks)} relevant chunks.")
    print("Generating response...\n")
    response, stats, input_warnings = chat(message, chunks, history)
    if input_warnings:
        print(f"  [SCANNER WARNING] Potential injection in input:")
        for w in input_warnings:
            print(f"    {w}")
    print(response)
    print(f"\n  [{stats.provider}:{stats.model} | {stats.tokens_per_sec} tok/s | {stats.completion_tokens} tokens | {stats.elapsed_sec}s]")

    # Update history and persist
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": response})
    _save_chat_history(history)
    return history


def run():
    """Main REPL loop."""
    print("=" * 60)
    print("  PaperMill — AI-Powered Academic Writing Assistant")
    print("=" * 60)
    print("Type /help for commands, or ask a question.\n")

    history: list[dict] = _load_chat_history()
    if history:
        print(f"  Loaded {len(history) // 2} previous exchange(s) from history.\n")

    while True:
        try:
            user_input = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.startswith("/"):
            parts = user_input.split(None, 1)
            cmd = parts[0].lower()
            args = parts[1] if len(parts) > 1 else ""

            if cmd == "/quit" or cmd == "/exit":
                print("Goodbye!")
                break
            elif cmd == "/help":
                print_help()
            elif cmd == "/ingest":
                cmd_ingest(args)
            elif cmd == "/ingest-images":
                cmd_ingest_images(args)
            elif cmd == "/sources":
                cmd_sources()
            elif cmd == "/delete-source":
                cmd_delete_source(args)
            elif cmd == "/outline":
                cmd_outline()
            elif cmd == "/draft":
                cmd_draft(args)
            elif cmd == "/rewrite":
                cmd_rewrite(args)
            elif cmd == "/show":
                cmd_show(args)
            elif cmd == "/export":
                cmd_export(args)
            elif cmd == "/version":
                cmd_version(args)
            elif cmd == "/versions":
                cmd_versions()
            elif cmd == "/diff":
                cmd_diff(args)
            elif cmd == "/ref-add":
                cmd_ref_add(args)
            elif cmd == "/refs":
                cmd_refs()
            elif cmd == "/ref-remove":
                cmd_ref_remove(args)
            elif cmd == "/scan-rules":
                cmd_scan_rules()
            elif cmd == "/allowlist-add":
                cmd_allowlist_add(args)
            elif cmd == "/allowlist-remove":
                cmd_allowlist_remove(args)
            elif cmd == "/quarantine-release":
                cmd_quarantine_release_cmd(args)
            elif cmd == "/clear-history":
                history = []
                _save_chat_history(history)
                print("Chat history cleared.")
            else:
                print(f"Unknown command: {cmd}. Type /help for available commands.")
        else:
            history = handle_chat(user_input, history)
