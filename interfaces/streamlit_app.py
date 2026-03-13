"""Streamlit web interface for the RAG Paper Writing Assistant."""

import re
import sys
import json
from pathlib import Path, PurePosixPath

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import config
from config import PDF_DIR, DATA_DIR, VERSIONS_DIR, REPORTS_DIR, CHAT_HISTORY_PATH, save_settings
from core.scanner import (
    load_rules, ContentBlockedError, load_scan_history,
    add_to_allowlist, remove_from_allowlist, quarantine_release, compute_file_hash,
)
from core.ingestion import ingest_pdf
from core.image_ingestion import ingest_image, ingest_images_dir, IMAGE_EXTENSIONS
from core.retrieval import search
from core.generation import chat, draft_section, rewrite_section
from core.prompts import get_prompt, get_default_prompt, PROMPT_LABELS
from core.paper_structure import (
    list_sections_status, load_section, save_section, get_section_title,
    get_section_target_words,
    add_section, remove_section, move_section, reset_outline, _load_outline,
    undo_section, has_undo,
)
from core.db import list_sources, delete_source
from core.docexport import export_full_paper, export_section
from core.versioning import save_version, list_versions
from core.bibliography import add_reference, remove_reference, list_references, format_apa


def _sanitize_filename(name: str) -> str:
    """Sanitize an uploaded filename: strip path components, remove unsafe chars."""
    # Take only the basename (strip any directory components)
    name = Path(name).name
    name = PurePosixPath(name).name  # Also handle Unix-style paths
    # Remove any remaining path separators and null bytes
    name = re.sub(r'[\x00/\\]', '', name)
    # Remove leading dots (hidden files / traversal)
    name = name.lstrip('.')
    # Replace other unsafe chars
    name = re.sub(r'[<>:"|?*]', '_', name)
    if not name:
        name = "unnamed_file"
    return name


st.set_page_config(
    page_title="PaperMill",
    page_icon="\U0001f4dd",
    layout="wide",
)

# ── Chat history helpers ───────────────────────────────────────────────────

def _save_chat_history(history: list[dict]) -> None:
    """Write chat history to CHAT_HISTORY_PATH as JSON."""
    try:
        CHAT_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        CHAT_HISTORY_PATH.write_text(
            json.dumps(history, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        pass


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


# ── Session State ──────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    # Reconstruct display messages from persisted history
    loaded = _load_chat_history()
    st.session_state.messages = loaded
    st.session_state.chat_history = loaded
elif "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""<style>
        section[data-testid="stSidebar"] * {font-size: 14px !important;}
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3 {font-size: 16px !important;}
        section[data-testid="stSidebar"] > div:first-child {padding-top: 0;}
        section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {padding-top: 0;}
    </style>""", unsafe_allow_html=True)
    st.markdown('<p style="font-size:24px; font-weight:bold; margin-bottom:0;">PaperMill</p>', unsafe_allow_html=True)
    st.caption("AI-Powered Academic Writing Assistant")

    # ━━ Settings ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    with st.expander("LLM Settings", expanded=False):
        # ── LLM Provider ─────────────────────────────────────────────
        providers = ["ollama", "lmstudio", "claude", "openai", "openrouter"]
        _provider_labels = {
            "ollama": "Local Ollama",
            "lmstudio": "LM Studio",
            "claude": "Claude (Sonnet/Opus)",
            "openai": f"OpenAI ({config.OPENAI_DRAFT_MODEL})",
            "openrouter": f"OpenRouter ({config.OPENROUTER_DRAFT_MODEL})",
        }
        current_idx = providers.index(config.LLM_PROVIDER) if config.LLM_PROVIDER in providers else 0
        selected_provider = st.selectbox(
            "LLM Provider",
            providers,
            index=current_idx,
            key="llm_provider",
            format_func=lambda p: _provider_labels.get(p, p),
        )
        if selected_provider != config.LLM_PROVIDER:
            config.LLM_PROVIDER = selected_provider
            _model_map = {
                "ollama": (config.OLLAMA_DRAFT_MODEL, config.OLLAMA_POLISH_MODEL),
                "lmstudio": (config.LMSTUDIO_DRAFT_MODEL, config.LMSTUDIO_POLISH_MODEL),
                "claude": (config.CLAUDE_DRAFT_MODEL, config.CLAUDE_POLISH_MODEL),
                "openai": (config.OPENAI_DRAFT_MODEL, config.OPENAI_POLISH_MODEL),
                "openrouter": (config.OPENROUTER_DRAFT_MODEL, config.OPENROUTER_POLISH_MODEL),
            }
            config.DRAFT_MODEL, config.POLISH_MODEL = _model_map.get(
                selected_provider, (config.OLLAMA_DRAFT_MODEL, config.OLLAMA_POLISH_MODEL)
            )

        # ── Model Selection (Ollama auto-detect) ─────────────────────
        if selected_provider == "ollama":
            @st.cache_data(ttl=30)
            def _fetch_ollama_models():
                import urllib.request
                try:
                    with urllib.request.urlopen(f"{config.OLLAMA_URL}/api/tags", timeout=5) as resp:
                        data = json.loads(resp.read().decode("utf-8"))
                    return sorted([m["name"] for m in data.get("models", [])])
                except Exception:
                    return []

            available_models = _fetch_ollama_models()
            if available_models:
                st.markdown("**Draft Model**")
                draft_idx = available_models.index(config.DRAFT_MODEL) if config.DRAFT_MODEL in available_models else 0
                new_draft = st.selectbox("Draft", available_models, index=draft_idx, key="ollama_draft_model", label_visibility="collapsed")
                st.markdown("**Polish Model**")
                polish_idx = available_models.index(config.POLISH_MODEL) if config.POLISH_MODEL in available_models else 0
                new_polish = st.selectbox("Polish", available_models, index=polish_idx, key="ollama_polish_model", label_visibility="collapsed")

                if new_draft != config.DRAFT_MODEL or new_polish != config.POLISH_MODEL:
                    config.DRAFT_MODEL = new_draft
                    config.OLLAMA_DRAFT_MODEL = new_draft
                    config.POLISH_MODEL = new_polish
                    config.OLLAMA_POLISH_MODEL = new_polish
            else:
                st.warning("Ollama not reachable")

        # ── Model Selection (LM Studio auto-detect) ──────────────────
        if selected_provider == "lmstudio":
            @st.cache_data(ttl=30)
            def _fetch_lmstudio_models():
                import urllib.request
                try:
                    with urllib.request.urlopen(f"{config.LMSTUDIO_URL}/models", timeout=5) as resp:
                        data = json.loads(resp.read().decode("utf-8"))
                    return sorted([m["id"] for m in data.get("data", [])])
                except Exception:
                    return []

            lms_models = _fetch_lmstudio_models()
            if lms_models:
                st.markdown("**Draft Model**")
                lms_draft_idx = lms_models.index(config.DRAFT_MODEL) if config.DRAFT_MODEL in lms_models else 0
                new_lms_draft = st.selectbox("Draft", lms_models, index=lms_draft_idx, key="lms_draft_model", label_visibility="collapsed")
                st.markdown("**Polish Model**")
                lms_polish_idx = lms_models.index(config.POLISH_MODEL) if config.POLISH_MODEL in lms_models else 0
                new_lms_polish = st.selectbox("Polish", lms_models, index=lms_polish_idx, key="lms_polish_model", label_visibility="collapsed")

                if new_lms_draft != config.DRAFT_MODEL or new_lms_polish != config.POLISH_MODEL:
                    config.DRAFT_MODEL = new_lms_draft
                    config.LMSTUDIO_DRAFT_MODEL = new_lms_draft
                    config.POLISH_MODEL = new_lms_polish
                    config.LMSTUDIO_POLISH_MODEL = new_lms_polish
            else:
                st.warning("LM Studio not reachable — is it running on localhost:1234?")

        # ── Embedding Settings ─────────────────────────────────────────
        st.markdown("**Embedding**")
        _embed_providers = ["local", "ollama", "openai"]
        _embed_labels = {
            "local": "Local (sentence-transformers)",
            "ollama": "Ollama",
            "openai": "OpenAI",
        }
        _embed_idx = _embed_providers.index(config.EMBEDDING_PROVIDER) if config.EMBEDDING_PROVIDER in _embed_providers else 0
        _sel_embed_provider = st.selectbox(
            "Embedding Provider",
            _embed_providers,
            index=_embed_idx,
            key="embed_provider",
            format_func=lambda p: _embed_labels.get(p, p),
        )

        # ── Model selection per provider ──────────────────────────────
        _embed_key = ""
        _popular_local = [
            "jinaai/jina-embeddings-v3",
            "BAAI/bge-large-en-v1.5",
            "sentence-transformers/all-MiniLM-L6-v2",
            "nomic-ai/nomic-embed-text-v1.5",
        ]

        if _sel_embed_provider == "local":
            _local_idx = _popular_local.index(config.EMBEDDING_MODEL) if config.EMBEDDING_MODEL in _popular_local else 0
            _sel_embed_model = st.selectbox("Embedding Model", _popular_local, index=_local_idx, key="embed_model_local")
            _custom_embed = st.text_input("Or custom HuggingFace model ID", key="embed_custom_local", placeholder="org/model-name")
            if _custom_embed.strip():
                _sel_embed_model = _custom_embed.strip()

        elif _sel_embed_provider == "ollama":
            @st.cache_data(ttl=30)
            def _fetch_ollama_embed_models():
                import urllib.request as _ur
                try:
                    with _ur.urlopen(f"{config.OLLAMA_URL}/api/tags", timeout=5) as resp:
                        data = json.loads(resp.read().decode("utf-8"))
                    return sorted([m["name"] for m in data.get("models", [])])
                except Exception:
                    return []

            _ollama_embed_models = _fetch_ollama_embed_models()
            st.caption("Select an embedding model (e.g. nomic-embed-text, mxbai-embed-large)")
            if _ollama_embed_models:
                _oll_idx = _ollama_embed_models.index(config.EMBEDDING_MODEL) if config.EMBEDDING_MODEL in _ollama_embed_models else 0
                _sel_embed_model = st.selectbox("Embedding Model", _ollama_embed_models, index=_oll_idx, key="embed_model_ollama")
            else:
                _sel_embed_model = config.EMBEDDING_MODEL
                st.warning("Ollama not reachable")
            _custom_ollama = st.text_input("Or custom model name", key="embed_custom_ollama", placeholder="nomic-embed-text")
            if _custom_ollama.strip():
                _sel_embed_model = _custom_ollama.strip()

        elif _sel_embed_provider == "openai":
            _openai_models = ["text-embedding-3-small", "text-embedding-3-large"]
            _oai_idx = _openai_models.index(config.EMBEDDING_MODEL) if config.EMBEDDING_MODEL in _openai_models else 0
            _sel_embed_model = st.selectbox("Embedding Model", _openai_models, index=_oai_idx, key="embed_model_openai")
            _embed_key = st.text_input(
                "Embedding API Key (optional, uses OpenAI key if empty)",
                value=config.EMBEDDING_API_KEY,
                key="embed_api_key",
                type="password",
            )
        else:
            _sel_embed_model = config.EMBEDDING_MODEL

        if st.button("Save Embedding Settings", key="save_embed_btn", use_container_width=True):
            _embed_updates = {
                "embedding_provider": _sel_embed_provider,
                "embedding_model": _sel_embed_model,
            }
            if _sel_embed_provider == "openai":
                _embed_updates["embedding_api_key"] = _embed_key
            save_settings(_embed_updates)
            config.EMBEDDING_PROVIDER = _sel_embed_provider
            config.EMBEDDING_MODEL = _sel_embed_model
            if _sel_embed_provider == "openai":
                config.EMBEDDING_API_KEY = _embed_key
            st.success("Embedding settings saved!")
            st.rerun()

        # ── Stale vector warning ──────────────────────────────────────
        _current_embed_key = f"{config.EMBEDDING_PROVIDER}:{config.EMBEDDING_MODEL}"
        if config.LAST_EMBEDDING_MODEL and config.LAST_EMBEDDING_MODEL != _current_embed_key:
            from core.db import list_sources as _list_embed_sources
            if _list_embed_sources():
                st.warning("Embedding model changed. Re-ingest sources for accurate results.")

        if st.button("Re-ingest All Sources", key="reingest_all_btn", use_container_width=True):
            st.session_state["confirm_reingest"] = True
            st.rerun()

        if st.session_state.get("confirm_reingest"):
            _pdf_count = len(list(config.PDF_DIR.glob("*.pdf")))
            st.warning(f"This will re-process {_pdf_count} PDFs. Scanner checks will run again. Continue?")
            _rc1, _rc2 = st.columns(2)
            if _rc1.button("Confirm Re-ingest", key="confirm_reingest_btn", type="primary"):
                st.session_state.pop("confirm_reingest", None)
                from core.ingestion import reingest_all
                _progress_bar = st.progress(0)
                _progress_text = st.empty()

                def _reingest_progress(current, total, filename):
                    if total > 0:
                        _progress_bar.progress(current / total)
                    _progress_text.caption(f"Processing {filename}... ({current + 1}/{total})")

                result = reingest_all(progress_callback=_reingest_progress)
                _progress_bar.progress(1.0)
                _progress_text.empty()
                st.success(f"Re-ingested {result['ingested']} PDFs, {result['total_chunks']} chunks.")
                if result["failed"]:
                    for f in result["failed"]:
                        st.error(f)
                st.rerun()
            if _rc2.button("Cancel", key="cancel_reingest_btn"):
                st.session_state.pop("confirm_reingest", None)
                st.rerun()

        # ── Prompt Settings (opens dialog) ────────────────────────────
        if st.button("System Prompt Settings", key="open_prompt_editor_btn", use_container_width=True):
            st.session_state["show_prompt_editor"] = True
            st.rerun()

    # ━━ Paper Sections Manager ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    with st.expander("Paper Sections", expanded=False):
        outline = _load_outline()

        for i, (sid, title, words) in enumerate(outline):
            cols = st.columns([3, 1, 1, 1])
            cols[0].caption(f"**{sid}** — {title} ({words}w)")
            if cols[1].button("\u2191", key=f"mv_up_{sid}", help="Move up"):
                move_section(sid, "up")
                st.rerun()
            if cols[2].button("\u2193", key=f"mv_dn_{sid}", help="Move down"):
                move_section(sid, "down")
                st.rerun()
            if cols[3].button("\u2716", key=f"rm_{sid}", help="Remove"):
                remove_section(sid)
                st.rerun()

        st.markdown("**Add Section**")
        ac1, ac2, ac3 = st.columns([2, 3, 1])
        new_sid = ac1.text_input("ID", placeholder="ch3.6", key="new_sec_id")
        new_stitle = ac2.text_input("Title", placeholder="3.6 New Subsection", key="new_sec_title")
        new_swords = ac3.number_input("Words", value=800, min_value=0, step=100, key="new_sec_words")

        after_options = ["(end)"] + [f"{sid}" for sid, _, _ in outline]
        insert_after = st.selectbox("Insert after", after_options, key="insert_after")

        if st.button("Add Section", key="add_sec_btn", use_container_width=True):
            if new_sid.strip() and new_stitle.strip():
                try:
                    after_val = None if insert_after == "(end)" else insert_after
                    add_section(new_sid.strip(), new_stitle.strip(), int(new_swords), after=after_val)
                    st.success(f"Added {new_sid}")
                    st.rerun()
                except ValueError as e:
                    st.error(str(e))
            else:
                st.warning("ID and Title are required.")

        if st.button("Reset to Default", key="reset_outline_btn", use_container_width=True):
            reset_outline()
            st.success("Outline reset to default.")
            st.rerun()

    # ━━ Write ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    st.subheader("Write")
    statuses = list_sections_status()
    status_icons = {"empty": "\u2b1c", "draft": "\U0001f4dd", "review": "\U0001f50d", "final": "\u2705"}

    section_options = [f"{status_icons.get(s['status'], '\u2b1c')} {s['id']}: {s['title']}" for s in statuses]
    selected_idx = st.selectbox(
        "Select section",
        range(len(section_options)),
        format_func=lambda i: section_options[i],
        key="section_select",
    )

    if selected_idx is not None:
        selected = statuses[selected_idx]
        existing_section = load_section(selected["id"])

        # Show generation stats from previous draft/rewrite
        # Clear if user switched to a different section
        if st.session_state.get("gen_stats_section") != selected["id"]:
            st.session_state.pop("gen_stats", None)
        if st.session_state.get("gen_stats"):
            st.success(st.session_state["gen_stats"])

        col1, col2, col3 = st.columns(3)

        # ── Draft button with confirmation ───────────────────────────
        if existing_section and selected["status"] != "empty":
            draft_clicked = col1.button("Draft", key="draft_btn", use_container_width=True)
            if draft_clicked:
                st.session_state["confirm_draft"] = selected["id"]

            if st.session_state.get("confirm_draft") == selected["id"]:
                st.warning(f"**{selected['id']}** already has a {selected['status']}. Drafting will overwrite it.")
                cc1, cc2 = st.columns(2)
                if cc1.button("Confirm Draft", key="confirm_draft_btn", type="primary"):
                    st.session_state.pop("confirm_draft", None)
                    with st.spinner(f"Drafting {selected['title']}..."):
                        chunks = search(selected["title"], top_k=12)
                        text, stats = draft_section(selected["id"], chunks)
                    st.session_state["gen_stats"] = f"Draft saved! | {stats.provider}:{stats.model} | {stats.tokens_per_sec} tok/s | {stats.completion_tokens} tokens | {stats.elapsed_sec}s | ctx: {stats.prompt_tokens} tok"
                    st.session_state["gen_stats_section"] = selected["id"]
                    st.rerun()
                if cc2.button("Cancel", key="cancel_draft_btn"):
                    st.session_state.pop("confirm_draft", None)
                    st.rerun()
        else:
            if col1.button("Draft", key="draft_btn", use_container_width=True):
                with st.spinner(f"Drafting {selected['title']}..."):
                    chunks = search(selected["title"], top_k=12)
                    text, stats = draft_section(selected["id"], chunks)
                st.session_state["gen_stats"] = f"Draft saved! | {stats.provider}:{stats.model} | {stats.tokens_per_sec} tok/s | {stats.completion_tokens} tokens | {stats.elapsed_sec}s | ctx: {stats.prompt_tokens} tok"
                st.session_state["gen_stats_section"] = selected["id"]
                st.rerun()

        # ── Rewrite button with confirmation ─────────────────────────
        if col2.button("Rewrite", key="rewrite_btn", use_container_width=True):
            if existing_section:
                st.session_state["confirm_rewrite"] = selected["id"]
            else:
                st.warning("Draft this section first.")

        if st.session_state.get("confirm_rewrite") == selected["id"]:
            st.warning(f"Rewriting **{selected['id']}** will replace the current text.")
            cc1, cc2 = st.columns(2)
            if cc1.button("Confirm Rewrite", key="confirm_rewrite_btn", type="primary"):
                st.session_state.pop("confirm_rewrite", None)
                with st.spinner(f"Rewriting with {config.POLISH_MODEL}..."):
                    chunks = search(selected["title"], top_k=8)
                    text, stats = rewrite_section(selected["id"], chunks=chunks)
                st.session_state["gen_stats"] = f"Rewrite saved! | {stats.provider}:{stats.model} | {stats.tokens_per_sec} tok/s | {stats.completion_tokens} tokens | {stats.elapsed_sec}s | ctx: {stats.prompt_tokens} tok"
                st.session_state["gen_stats_section"] = selected["id"]
                st.rerun()
            if cc2.button("Cancel", key="cancel_rewrite_btn"):
                st.session_state.pop("confirm_rewrite", None)
                st.rerun()

        # ── Undo button ──────────────────────────────────────────────
        if has_undo(selected["id"]):
            if col3.button("Undo", key="undo_btn", use_container_width=True):
                undo_section(selected["id"])
                st.success(f"Restored previous version of {selected['id']}.")
                st.rerun()
        else:
            col3.button("Undo", key="undo_btn", use_container_width=True, disabled=True)


    # ━━ Paper & Export ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    with st.expander("Paper & Export", expanded=False):
        new_title = st.text_input("Paper Title", value=config.PAPER_TITLE, key="cfg_title")
        _font_options = [
            "Times New Roman", "Arial", "Calibri", "Cambria",
            "Garamond", "Georgia", "Palatino Linotype", "Book Antiqua",
            "Century Schoolbook", "Courier New", "Verdana", "Tahoma",
        ]
        _font_idx = _font_options.index(config.DOCX_FONT) if config.DOCX_FONT in _font_options else 0
        new_font = st.selectbox("Font", _font_options, index=_font_idx, key="cfg_font")
        new_font_size = st.number_input("Font Size (pt)", value=config.DOCX_FONT_SIZE_PT, min_value=8, max_value=24, step=1, key="cfg_font_size")
        new_line_spacing = st.number_input("Line Spacing", value=config.DOCX_LINE_SPACING, min_value=1.0, max_value=3.0, step=0.25, key="cfg_spacing")
        new_margin = st.number_input("Margins (inches)", value=config.DOCX_MARGIN_INCHES, min_value=0.5, max_value=2.0, step=0.25, key="cfg_margin")

        if st.button("Save Settings", key="save_settings_btn", use_container_width=True):
            updates = {
                "paper_title": new_title,
                "docx_font": new_font,
                "docx_font_size": int(new_font_size),
                "docx_line_spacing": float(new_line_spacing),
                "docx_margin_inches": float(new_margin),
            }
            save_settings(updates)
            config.PAPER_TITLE = new_title
            config.DOCX_FONT = new_font
            config.DOCX_FONT_SIZE_PT = int(new_font_size)
            config.DOCX_LINE_SPACING = float(new_line_spacing)
            config.DOCX_MARGIN_INCHES = float(new_margin)
            st.success("Settings saved!")
            st.rerun()

    # ━━ Export & Versioning ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    st.subheader("Export & Versioning")

    col1, col2 = st.columns(2)
    if col1.button("Export .docx", key="export_btn", use_container_width=True):
        path = VERSIONS_DIR / "paper_current.docx"
        export_full_paper(path)
        with open(path, "rb") as f:
            st.download_button(
                "Download .docx",
                f.read(),
                file_name="paper_current.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                key="download_current",
            )

    version_label = st.text_input("Version label", key="version_label")
    if col2.button("Save Version", key="version_btn", use_container_width=True):
        entry = save_version(version_label)
        st.success(f"Saved v{entry['version']}: {entry['filename']}")

    # Version list
    versions = list_versions()
    if versions:
        with st.expander(f"{len(versions)} saved versions", expanded=False):
            for v in reversed(versions[-5:]):
                vpath = VERSIONS_DIR / v["filename"]
                if vpath.exists():
                    with open(vpath, "rb") as f:
                        st.download_button(
                            f"v{v['version']}: {v['label']} ({v['timestamp'][:10]})",
                            f.read(),
                            file_name=v["filename"],
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                            key=f"dl_v{v['version']}",
                        )


    # ━━ Bibliography ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    st.subheader("Bibliography")
    doi_input = st.text_input("Add reference by DOI", key="doi_input", placeholder="10.1000/example")
    if st.button("Fetch & Add", key="add_ref_btn"):
        if doi_input.strip():
            try:
                entry = add_reference(doi=doi_input)
                st.success(f"Added [{entry['key']}]: {entry['title']}")
                st.rerun()
            except Exception as e:
                st.error(str(e))

    bib_refs = list_references()
    if bib_refs:
        with st.expander(f"{len(bib_refs)} references", expanded=False):
            for r in bib_refs:
                col1, col2 = st.columns([4, 1])
                col1.caption(f"**[{r['key']}]** {format_apa(r)}")
                if col2.button("X", key=f"rdel_{r['key']}", help=f"Remove {r['key']}"):
                    remove_reference(r["key"])
                    st.rerun()


    # ━━ Add Sources ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    st.subheader("Add Sources")

    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"], key="pdf_uploader")
    if uploaded_file is not None:
        _safe_pdf_name = _sanitize_filename(uploaded_file.name)
        save_path = PDF_DIR / _safe_pdf_name
        uploaded_bytes = uploaded_file.getvalue()
        _is_duplicate = False

        # Check for duplicate by filename + size
        if save_path.exists():
            if save_path.stat().st_size == len(uploaded_bytes):
                st.info(f"**{_safe_pdf_name}** already exists (same size). Skipped saving.")
                _is_duplicate = True
            else:
                st.warning(f"**{_safe_pdf_name}** exists but differs in size ({save_path.stat().st_size} vs {len(uploaded_bytes)} bytes). Overwriting.")
                save_path.write_bytes(uploaded_bytes)
        else:
            # Check if same content exists under a different filename
            _size_match = [p for p in PDF_DIR.glob("*.pdf") if p.stat().st_size == len(uploaded_bytes) and p.name != _safe_pdf_name]
            if _size_match:
                st.warning(f"A file with the same size already exists: **{_size_match[0].name}**. This may be a duplicate.")
            save_path.write_bytes(uploaded_bytes)

        # Also check if already ingested in the vector DB
        if _safe_pdf_name in list_sources():
            st.info(f"**{_safe_pdf_name}** is already ingested in the vector DB.")

        if st.button("Ingest uploaded PDF", key="ingest_btn"):
            with st.spinner(f"Scanning & ingesting {_safe_pdf_name}..."):
                try:
                    count = ingest_pdf(save_path)
                    if count > 0:
                        st.success(f"Ingested {count} chunks from {_safe_pdf_name}")
                    else:
                        st.info("Already ingested.")
                except ContentBlockedError as e:
                    st.error(f"BLOCKED: {e}")
                    st.warning(f"Report: {e.report_path}")

    IMAGES_DIR = DATA_DIR / "images"
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    uploaded_img = st.file_uploader(
        "Upload an image",
        type=["png", "jpg", "jpeg", "gif", "webp", "bmp"],
        key="img_uploader",
    )
    if uploaded_img is not None:
        _safe_img_name = _sanitize_filename(uploaded_img.name)
        img_path = IMAGES_DIR / _safe_img_name
        img_bytes = uploaded_img.getvalue()

        if img_path.exists():
            if img_path.stat().st_size == len(img_bytes):
                st.info(f"**{_safe_img_name}** already exists (same size). Skipped saving.")
            else:
                st.warning(f"**{_safe_img_name}** exists but differs in size. Overwriting.")
                img_path.write_bytes(img_bytes)
        else:
            img_path.write_bytes(img_bytes)

        st.image(img_path, caption=_safe_img_name, width=200)
        if st.button("Ingest image", key="ingest_img_btn"):
            with st.spinner(f"Scanning & indexing {_safe_img_name}..."):
                try:
                    count = ingest_image(img_path)
                    if count > 0:
                        st.success(f"Indexed {_safe_img_name}")
                    else:
                        st.info("Already ingested.")
                except ContentBlockedError as e:
                    st.error(f"BLOCKED: {e}")
                    st.warning(f"Report: {e.report_path}")

    if st.button("Ingest all images", key="ingest_all_imgs_btn"):
        with st.spinner("Scanning & indexing all images..."):
            results = ingest_images_dir(IMAGES_DIR)
        st.success(f"Ingested: {results['ingested']}, Skipped: {results['skipped']}")
        for err in results["errors"]:
            st.error(err)


    # ━━ Ingested Sources ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    sources = list_sources()
    st.subheader(f"Ingested Sources ({len(sources)})")
    if sources:
        for src in sources:
            col1, col2 = st.columns([3, 1])
            col1.markdown(f"- {src}")
            if col2.button("X", key=f"del_{src}", help=f"Delete {src}"):
                st.session_state["confirm_delete_source"] = src

        if "confirm_delete_source" in st.session_state:
            _del_src = st.session_state["confirm_delete_source"]
            st.warning(f"Delete **{_del_src}** from the vector DB? The file on disk will not be removed.")
            cc1, cc2 = st.columns(2)
            if cc1.button("Confirm Delete", key="confirm_del_src_btn", type="primary"):
                delete_source(_del_src)
                st.session_state.pop("confirm_delete_source", None)
                st.success(f"Deleted {_del_src}")
                st.rerun()
            if cc2.button("Cancel", key="cancel_del_src_btn"):
                st.session_state.pop("confirm_delete_source", None)
                st.rerun()
    else:
        st.caption("No sources ingested yet.")

    # ━━ Content Scanner ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    st.subheader("Content Scanner")

    with st.expander("Active Rules", expanded=False):
        rules = load_rules()
        for r in rules:
            severity_icon = "\U0001f534" if r["severity"] == "high" else "\U0001f7e1"
            st.caption(f"{severity_icon} **{r['id']}** [{r['category']}] {r['description']}")

    with st.expander("Scan Reports", expanded=False):
        report_files = sorted(REPORTS_DIR.glob("*.md"), reverse=True)[:10]
        if report_files:
            for rf in report_files:
                col1, col2 = st.columns([3, 1])
                col1.caption(f"\U0001f4c4 {rf.name}")
                if col2.button("View", key=f"view_{rf.name}"):
                    st.session_state["view_report"] = str(rf)
                    st.rerun()
        else:
            st.caption("No reports yet.")

    # Show report in a dialog popup
    if "view_report" in st.session_state:
        _rp = Path(st.session_state["view_report"])

        @st.dialog(f"Report: {_rp.name}", width="large")
        def _show_report():
            if _rp.exists():
                st.markdown(_rp.read_text(encoding="utf-8"))
            else:
                st.warning("Report file not found.")
            if st.button("Close", key="close_report_dialog"):
                st.session_state.pop("view_report", None)
                st.rerun()

        _show_report()

    with st.expander("Quarantine", expanded=False):
        history = load_scan_history()
        blocked = {k: v for k, v in history.items() if v["result"] == "blocked"}
        if blocked:
            for h, entry in blocked.items():
                col1, col2 = st.columns([3, 1])
                col1.caption(f"\U0001f6ab {entry['filename']} ({h[:12]}...)")
                _report_path = entry.get("report", "")
                if _report_path and Path(_report_path).exists():
                    if col2.button("View Report", key=f"qreport_{h[:12]}"):
                        st.session_state["view_report"] = _report_path
                        st.rerun()
                else:
                    col2.caption("No report")
        else:
            st.caption("No quarantined files.")

    st.caption("**PaperMill** v1.3 | [GitHub](https://github.com/cr231521/PaperMill) | CC BY-NC 4.0")


# ── Main Area ─────────────────────────────────────────────────────────────
tab_chat, tab_section = st.tabs(["Chat", "Section Viewer"])

with tab_section:
    if selected_idx is not None:
        selected = statuses[selected_idx]
        section = load_section(selected["id"])
        if section:
            st.markdown(f"### {section['title']}")
            from datetime import datetime, timezone
            _updated = section['updated_at']
            try:
                _dt = datetime.fromisoformat(_updated)
                _local = _dt.astimezone()
                _date_str = _local.strftime("%Y-%m-%d %H:%M")
                _now = datetime.now(timezone.utc)
                _delta = _now - _dt
                _mins = int(_delta.total_seconds() // 60)
                if _mins < 1:
                    _ago = "just now"
                elif _mins < 60:
                    _ago = f"{_mins}m ago"
                elif _mins < 1440:
                    _ago = f"{_mins // 60}h {_mins % 60}m ago"
                else:
                    _ago = f"{_mins // 1440}d ago"
            except Exception:
                _date_str = _updated
                _ago = ""
            st.caption(f"Status: {section['status']} | Updated: {_date_str} ({_ago})")

            # ── Word count progress bar ────────────────────────────
            target = get_section_target_words(selected["id"])
            current_words = len(section["text"].split())
            pct = min(current_words / target, 1.0) if target > 0 else 0.0
            line_count = section["text"].count("\n") + 1
            est_pages = current_words / 250  # ~250 words per page (standard academic)
            st.progress(pct, text=f"{current_words} / {target} words ({pct:.0%}) · {line_count} lines · ~{est_pages:.1f} pages")

            # Toggle between Read and Write mode
            write_mode = st.toggle("Write mode", value=False, key=f"write_mode_{selected['id']}")

            # Show save confirmation if flagged
            if st.session_state.get("section_saved"):
                st.success("Section saved.")
                del st.session_state["section_saved"]

            if write_mode:
                edited_text = st.text_area(
                    "Edit section content (Markdown)",
                    value=section["text"],
                    height=500,
                    key=f"editor_{selected['id']}",
                )
                if st.button("Save", key=f"save_{selected['id']}", type="primary"):
                    save_section(selected["id"], section["title"], edited_text, section["status"])
                    st.session_state["section_saved"] = True
                    st.rerun()
            else:
                st.markdown(section["text"])

            # ── Export single section ──────────────────────────────
            if st.button("Export this section as .docx", key=f"export_section_{selected['id']}"):
                export_path = VERSIONS_DIR / f"{selected['id']}.docx"
                export_section(selected["id"], export_path)
                with open(export_path, "rb") as f:
                    st.download_button(
                        f"Download {selected['id']}.docx",
                        f,
                        file_name=f"{selected['id']}.docx",
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        key=f"dl_section_{selected['id']}",
                    )

            # ── Citation audit ─────────────────────────────────────
            with st.expander("Citation Audit", expanded=False):
                bib_keys = {r["key"] for r in list_references()}
                # Find all (Author, Year) style citations in text
                cited = set(re.findall(r'\(([A-Z][a-zA-Z]+(?:\s*(?:et al\.?|&\s*[A-Z][a-zA-Z]+))*,\s*\d{4})\)', section["text"]))
                # Find [Source: ...] placeholders
                placeholders = re.findall(r'\[Source:\s*([^\]]+)\]', section["text"])

                if placeholders:
                    st.warning(f"**{len(placeholders)} unresolved placeholder(s):**")
                    for p in placeholders:
                        st.caption(f"- [Source: {p}]")

                if cited:
                    matched = []
                    unmatched = []
                    for c in cited:
                        author_part = c.split(",")[0].strip().replace(" ", "")
                        if any(author_part.lower() in k.lower() for k in bib_keys):
                            matched.append(c)
                        else:
                            unmatched.append(c)
                    if unmatched:
                        st.warning(f"**{len(unmatched)} citation(s) not found in bibliography:**")
                        for u in unmatched:
                            st.caption(f"- ({u})")
                    if matched:
                        st.success(f"{len(matched)} citation(s) matched in bibliography.")
                else:
                    st.info("No APA-style citations detected in this section.")
        else:
            st.info(f"Section '{selected['id']}' has not been drafted yet.")

with tab_chat:
    _ch_col1, _ch_col2 = st.columns([6, 1])
    _ch_col1.markdown("### Research Chat")
    _ch_col1.caption("Ask questions about your reference materials.")
    if _ch_col2.button("Clear", key="clear_history_btn"):
        st.session_state.messages = []
        st.session_state.chat_history = []
        _save_chat_history([])
        st.rerun()

    # Display chat history (newest first)
    for msg in reversed(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("stats"):
                st.caption(msg["stats"])

    # Chat input
    if prompt := st.chat_input("Ask a question about your references..."):
        # Display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Searching references and generating response..."):
                chunks = search(prompt, top_k=8)
                response_text, stats, input_warnings = chat(prompt, chunks, st.session_state.chat_history)

            if input_warnings:
                st.warning("Scanner detected potential injection in your message:\n" + "\n".join(input_warnings))

            st.markdown(response_text)

            # Performance stats
            st.caption(
                f"{stats.provider}:{stats.model} | "
                f"{stats.tokens_per_sec} tok/s | "
                f"{stats.completion_tokens} tokens | "
                f"{stats.elapsed_sec}s | "
                f"ctx: {stats.prompt_tokens} tok"
            )

            # Show sources
            if chunks:
                with st.expander("Sources", expanded=False):
                    for i, c in enumerate(chunks, 1):
                        st.markdown(
                            f"**[{i}]** {c['source_pdf']}, pp. {c['page_start']}-{c['page_end']} "
                            f"({c.get('section_hint', 'N/A')})"
                        )

        _stats_str = (
            f"{stats.provider}:{stats.model} | "
            f"{stats.tokens_per_sec} tok/s | "
            f"{stats.completion_tokens} tokens | "
            f"{stats.elapsed_sec}s | "
            f"ctx: {stats.prompt_tokens} tok"
        )
        st.session_state.messages.append({"role": "assistant", "content": response_text, "stats": _stats_str})
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        st.session_state.chat_history.append({"role": "assistant", "content": response_text})
        _save_chat_history(st.session_state.chat_history)


# ── Prompt Settings Dialog ────────────────────────────────────────────────
@st.dialog("Prompt Settings", width="large")
def _prompt_editor_dialog():
    st.caption("View and edit the system prompts used for drafting, rewriting, and chat.")

    _prompt_names = list(PROMPT_LABELS.keys())
    _prompt_display = [PROMPT_LABELS[n] for n in _prompt_names]
    _sel_prompt_idx = st.selectbox(
        "Select prompt to edit",
        range(len(_prompt_names)),
        format_func=lambda i: _prompt_display[i],
        key="prompt_selector",
    )
    _sel_prompt_name = _prompt_names[_sel_prompt_idx]
    _current_val = get_prompt(_sel_prompt_name)
    _default_val = get_default_prompt(_sel_prompt_name)
    _is_custom = _current_val != _default_val

    if _is_custom:
        st.info("This prompt has been customised.")

    # Placeholder hints for template prompts
    if _sel_prompt_name in ("section_draft_template", "rewrite_template", "chat_context_template"):
        if _sel_prompt_name == "section_draft_template":
            st.caption("Placeholders: `{section_name}` `{section_id}` `{instructions}` `{context}` `{bibliography}` `{target_length}`")
        elif _sel_prompt_name == "rewrite_template":
            st.caption("Placeholders: `{section_name}` `{section_id}` `{instructions}` `{current_text}` `{context}` `{bibliography}`")
        elif _sel_prompt_name == "chat_context_template":
            st.caption("Placeholders: `{context}` `{question}`")

    _edited_prompt = st.text_area(
        "Prompt text",
        value=_current_val,
        height=400,
        key=f"prompt_edit_{_sel_prompt_name}",
        label_visibility="collapsed",
    )

    _pc1, _pc2 = st.columns(2)
    if _pc1.button("Save Prompt", key="save_prompt_btn", type="primary", use_container_width=True):
        _settings = config._load_settings()
        _prompts = _settings.get("prompts", {})
        if _edited_prompt.strip() == _default_val.strip():
            _prompts.pop(_sel_prompt_name, None)
        else:
            _prompts[_sel_prompt_name] = _edited_prompt
        save_settings({"prompts": _prompts})
        st.session_state.pop("show_prompt_editor", None)
        st.rerun()

    if _pc2.button("Reset to Default", key="reset_prompt_btn", use_container_width=True, disabled=not _is_custom):
        _settings = config._load_settings()
        _prompts = _settings.get("prompts", {})
        _prompts.pop(_sel_prompt_name, None)
        save_settings({"prompts": _prompts})
        st.session_state.pop("show_prompt_editor", None)
        st.rerun()


if st.session_state.get("show_prompt_editor"):
    _prompt_editor_dialog()
