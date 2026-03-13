"""PaperMill RAG MCP Server — semantic code search over PaperMill codebase.

Provides RAG (vector search) and grep tools for Claude Code via MCP.
Separate LanceDB at ~/.claude/lancedb/papermill-rag/.
Reuses Jina v3 embeddings from core/embedder.py.
"""

import ast
import hashlib
import json
import re
import sys
import logging
from pathlib import Path
from datetime import datetime, timezone

# Add project root to path so we can import core modules
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from mcp.server.fastmcp import FastMCP
import lancedb
from lancedb.pydantic import LanceModel, Vector

import config
from core.embedder import embed_passages, embed_query, get_embedding_dim

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────

DEVRAG_DB_DIR = Path.home() / ".claude" / "lancedb" / "papermill-rag"
DEVRAG_TABLE = "code_chunks"
SKIP_DIRS = {"__pycache__", ".git", "venv", ".venv", ".pytest_cache", "node_modules", ".superpowers"}
SKIP_FILES = {"devrag_mcp.py"}
FILE_EXTENSIONS = {".py", ".json"}

# ── LanceDB setup ────────────────────────────────────────────────────────

_db = None


def _get_db():
    global _db
    if _db is None:
        DEVRAG_DB_DIR.mkdir(parents=True, exist_ok=True)
        _db = lancedb.connect(str(DEVRAG_DB_DIR))
    return _db


def _make_record_class(dim: int):
    class CodeChunk(LanceModel):
        id: str
        vector: Vector(dim)
        text: str
        source_file: str
        chunk_type: str  # "function", "class", "module", "json"
        chunk_name: str  # function/class name or "module"
        line_start: int
        line_end: int
        ingested_at: str
    return CodeChunk


def _get_table():
    db = _get_db()
    if DEVRAG_TABLE in db.table_names():
        return db.open_table(DEVRAG_TABLE)
    return None


def _create_table(records: list[dict]):
    db = _get_db()
    if DEVRAG_TABLE in db.table_names():
        db.drop_table(DEVRAG_TABLE)
    dim = get_embedding_dim()
    RecordClass = _make_record_class(dim)
    return db.create_table(DEVRAG_TABLE, schema=RecordClass, data=records)


# ── AST-based Python chunking ────────────────────────────────────────────

def _chunk_id(source: str, name: str, line: int) -> str:
    raw = f"{source}:{name}:{line}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _get_source_lines(source: str, start: int, end: int) -> str:
    lines = source.split("\n")
    return "\n".join(lines[start - 1:end])


def _chunk_python(filepath: Path, rel_path: str) -> list[dict]:
    """Parse a Python file with AST, extract functions and classes as chunks."""
    try:
        source = filepath.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return []

    try:
        tree = ast.parse(source, filename=str(filepath))
    except SyntaxError:
        # Fall back to file-level chunk if unparseable
        return [{
            "text": f"[{rel_path}]\n{source[:4000]}",
            "source_file": rel_path,
            "chunk_type": "module",
            "chunk_name": "module",
            "line_start": 1,
            "line_end": source.count("\n") + 1,
        }]

    chunks = []
    used_lines = set()

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            chunk_type = "function"
            name = node.name
        elif isinstance(node, ast.ClassDef):
            chunk_type = "class"
            name = node.name
        else:
            continue

        start = node.lineno
        end = node.end_lineno or start
        text = _get_source_lines(source, start, end)

        # For classes, include the full class (methods are part of it)
        # but also extract methods separately for granular search
        if chunk_type == "class":
            chunks.append({
                "text": f"[{rel_path}:{name}]\n{text}",
                "source_file": rel_path,
                "chunk_type": "class",
                "chunk_name": name,
                "line_start": start,
                "line_end": end,
            })
            used_lines.update(range(start, end + 1))
            # Extract methods within the class
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    m_start = item.lineno
                    m_end = item.end_lineno or m_start
                    m_text = _get_source_lines(source, m_start, m_end)
                    chunks.append({
                        "text": f"[{rel_path}:{name}.{item.name}]\n{m_text}",
                        "source_file": rel_path,
                        "chunk_type": "function",
                        "chunk_name": f"{name}.{item.name}",
                        "line_start": m_start,
                        "line_end": m_end,
                    })
        else:
            # Skip if this function is inside a class (already handled above)
            # Check parent — if it's a top-level function
            is_top_level = False
            for top_node in ast.iter_child_nodes(tree):
                if top_node is node:
                    is_top_level = True
                    break
            if is_top_level:
                chunks.append({
                    "text": f"[{rel_path}:{name}]\n{text}",
                    "source_file": rel_path,
                    "chunk_type": "function",
                    "chunk_name": name,
                    "line_start": start,
                    "line_end": end,
                })
                used_lines.update(range(start, end + 1))

    # Module-level code (imports, constants, etc.)
    module_lines = []
    all_lines = source.split("\n")
    for i, line in enumerate(all_lines, 1):
        if i not in used_lines and line.strip():
            module_lines.append(line)

    if module_lines:
        module_text = "\n".join(module_lines)
        if len(module_text.strip()) > 20:  # Skip trivially small module chunks
            chunks.append({
                "text": f"[{rel_path}:module]\n{module_text}",
                "source_file": rel_path,
                "chunk_type": "module",
                "chunk_name": "module",
                "line_start": 1,
                "line_end": len(all_lines),
            })

    return chunks


def _chunk_json(filepath: Path, rel_path: str) -> list[dict]:
    """Chunk a JSON file as a single unit."""
    try:
        text = filepath.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return []

    # Truncate very large JSON files
    if len(text) > 8000:
        text = text[:8000] + "\n... (truncated)"

    return [{
        "text": f"[{rel_path}]\n{text}",
        "source_file": rel_path,
        "chunk_type": "json",
        "chunk_name": filepath.stem,
        "line_start": 1,
        "line_end": text.count("\n") + 1,
    }]


def _collect_files() -> list[Path]:
    """Collect all .py and .json files from the project."""
    files = []
    for path in PROJECT_ROOT.rglob("*"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.name in SKIP_FILES:
            continue
        if path.suffix in FILE_EXTENSIONS and path.is_file():
            files.append(path)
    return sorted(files)


# ── MCP Server ────────────────────────────────────────────────────────────

mcp = FastMCP("papermill-rag", instructions="PaperMill codebase RAG — use rag_search for semantic questions, rag_grep for exact matches")


@mcp.tool()
def rag_ingest() -> str:
    """Re-index the PaperMill codebase. Run this after code changes to update the index."""
    files = _collect_files()
    all_chunks = []

    for f in files:
        rel = f.relative_to(PROJECT_ROOT).as_posix()
        if f.suffix == ".py":
            all_chunks.extend(_chunk_python(f, rel))
        elif f.suffix == ".json":
            all_chunks.extend(_chunk_json(f, rel))

    if not all_chunks:
        return "No files found to index."

    # Truncate long chunks to ~6000 chars (~1500 tokens) to avoid GPU OOM
    MAX_CHUNK_CHARS = 6000
    for c in all_chunks:
        if len(c["text"]) > MAX_CHUNK_CHARS:
            c["text"] = c["text"][:MAX_CHUNK_CHARS] + "\n... (truncated)"

    # Embed in small batches to avoid GPU OOM
    texts = [c["text"] for c in all_chunks]
    vectors = []
    BATCH = 16
    for i in range(0, len(texts), BATCH):
        vectors.extend(embed_passages(texts[i:i + BATCH]))

    now = datetime.now(timezone.utc).isoformat()
    records = []
    for i, (chunk, vec) in enumerate(zip(all_chunks, vectors)):
        records.append({
            "id": _chunk_id(chunk["source_file"], chunk["chunk_name"], chunk["line_start"]),
            "vector": vec,
            "text": chunk["text"],
            "source_file": chunk["source_file"],
            "chunk_type": chunk["chunk_type"],
            "chunk_name": chunk["chunk_name"],
            "line_start": chunk["line_start"],
            "line_end": chunk["line_end"],
            "ingested_at": now,
        })

    _create_table(records)
    file_count = len(set(c["source_file"] for c in all_chunks))
    return f"Indexed {len(records)} chunks from {file_count} files."


@mcp.tool()
def rag_search(query: str, top_k: int = 8) -> str:
    """Semantic search over the PaperMill codebase. Use for 'how does X work?' questions.

    Args:
        query: Natural language question about the code.
        top_k: Number of results to return (default 8).
    """
    table = _get_table()
    if table is None:
        return "Index not built yet. Run rag_ingest() first."

    query_vec = embed_query(query)
    results = table.search(query_vec).limit(top_k).to_list()

    if not results:
        return "No results found."

    output = []
    for i, r in enumerate(results, 1):
        dist = r.get("_distance", 0)
        output.append(
            f"**[{i}]** `{r['source_file']}:{r['line_start']}-{r['line_end']}` "
            f"({r['chunk_type']}: {r['chunk_name']}) [dist: {dist:.3f}]\n"
            f"```python\n{r['text']}\n```"
        )

    return "\n\n".join(output)


@mcp.tool()
def rag_grep(pattern: str, glob: str = "*.py") -> str:
    """Regex search across indexed codebase files. Use for exact symbol/string lookups.

    Args:
        pattern: Regex pattern to search for (case-insensitive).
        glob: File glob pattern to filter (default '*.py'). Use '*.json' for config files.
    """
    files = _collect_files()
    # Filter by glob
    from fnmatch import fnmatch
    files = [f for f in files if fnmatch(f.name, glob)]

    try:
        regex = re.compile(pattern, re.IGNORECASE)
    except re.error as e:
        return f"Invalid regex: {e}"

    results = []
    for f in files:
        try:
            lines = f.read_text(encoding="utf-8").split("\n")
        except (UnicodeDecodeError, OSError):
            continue

        rel = f.relative_to(PROJECT_ROOT).as_posix()
        for i, line in enumerate(lines, 1):
            if regex.search(line):
                # Show context: 1 line before and after
                start = max(0, i - 2)
                end = min(len(lines), i + 1)
                context = "\n".join(f"  {start + j + 1:4d} | {lines[start + j]}" for j in range(end - start))
                results.append(f"`{rel}:{i}`\n```\n{context}\n```")

                if len(results) >= 30:
                    results.append("... (truncated at 30 matches)")
                    return "\n\n".join(results)

    if not results:
        return f"No matches for `{pattern}` in `{glob}` files."

    return f"**{len(results)} matches:**\n\n" + "\n\n".join(results)


@mcp.tool()
def rag_sources() -> str:
    """List all files currently indexed in the code RAG."""
    table = _get_table()
    if table is None:
        return "Index not built yet. Run rag_ingest() first."

    df = table.to_pandas()
    if df.empty:
        return "Index is empty."

    sources = df.groupby("source_file").agg(
        chunks=("chunk_name", "count"),
        types=("chunk_type", lambda x: ", ".join(sorted(set(x)))),
    ).reset_index()

    lines = [f"**{len(sources)} files indexed, {len(df)} total chunks:**\n"]
    for _, row in sources.iterrows():
        lines.append(f"- `{row['source_file']}` — {row['chunks']} chunks ({row['types']})")

    return "\n".join(lines)


if __name__ == "__main__":
    mcp.run()
