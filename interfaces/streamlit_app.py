"""Streamlit web interface for the RAG Paper Writing Assistant."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import config
from config import PDF_DIR, DATA_DIR, VERSIONS_DIR
from core.ingestion import ingest_pdf
from core.image_ingestion import ingest_image, ingest_images_dir, IMAGE_EXTENSIONS
from core.retrieval import search
from core.generation import chat, draft_section, rewrite_section
from core.paper_structure import list_sections_status, load_section, get_section_title
from core.db import list_sources, delete_source
from core.docexport import export_full_paper, export_section
from core.versioning import save_version, list_versions
from core.bibliography import add_reference, remove_reference, list_references, format_apa


st.set_page_config(
    page_title="Paper Writing Assistant",
    page_icon="\U0001f4dd",
    layout="wide",
)

# ── Session State ──────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Paper Assistant")
    st.caption("CRA Compliance Toolkit for SMEs")

    # ── LLM Provider ─────────────────────────────────────────────────────
    providers = ["ollama", "claude"]
    current_idx = providers.index(config.LLM_PROVIDER) if config.LLM_PROVIDER in providers else 0
    selected_provider = st.selectbox(
        "LLM Provider",
        providers,
        index=current_idx,
        key="llm_provider",
        format_func=lambda p: f"Ollama ({config.OLLAMA_DRAFT_MODEL})" if p == "ollama" else "Claude (Sonnet/Opus)",
    )
    if selected_provider != config.LLM_PROVIDER:
        config.LLM_PROVIDER = selected_provider
        if selected_provider == "ollama":
            config.DRAFT_MODEL = config.OLLAMA_DRAFT_MODEL
            config.POLISH_MODEL = config.OLLAMA_POLISH_MODEL
        else:
            config.DRAFT_MODEL = config.CLAUDE_DRAFT_MODEL
            config.POLISH_MODEL = config.CLAUDE_POLISH_MODEL

    st.divider()

    # ── Bibliography ──────────────────────────────────────────────────────
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
        st.markdown(f"**{len(bib_refs)} references:**")
        for r in bib_refs:
            col1, col2 = st.columns([4, 1])
            col1.caption(f"**[{r['key']}]** {format_apa(r)}")
            if col2.button("X", key=f"rdel_{r['key']}", help=f"Remove {r['key']}"):
                remove_reference(r["key"])
                st.rerun()

    st.divider()

    # ── Paper Sections ────────────────────────────────────────────────────
    st.subheader("Paper Sections")
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

        col1, col2 = st.columns(2)
        if col1.button("Draft", key="draft_btn", use_container_width=True):
            with st.spinner(f"Drafting {selected['title']}..."):
                chunks = search(selected["title"], top_k=12)
                text, stats = draft_section(selected["id"], chunks)
            st.success(f"Draft saved! ({stats.tokens_per_sec} tok/s, {stats.elapsed_sec}s)")
            st.rerun()

        if col2.button("Rewrite", key="rewrite_btn", use_container_width=True):
            section = load_section(selected["id"])
            if section:
                with st.spinner(f"Rewriting with {config.POLISH_MODEL}..."):
                    chunks = search(selected["title"], top_k=8)
                    text, stats = rewrite_section(selected["id"], chunks=chunks)
                st.success(f"Rewrite saved! ({stats.tokens_per_sec} tok/s, {stats.elapsed_sec}s)")
                st.rerun()
            else:
                st.warning("Draft this section first.")

    st.divider()

    # ── Export & Versioning ───────────────────────────────────────────────
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
        st.markdown("**Saved Versions:**")
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

    st.divider()

    # ── Reference PDFs ────────────────────────────────────────────────────
    st.subheader("Reference PDFs")
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"], key="pdf_uploader")
    if uploaded_file is not None:
        save_path = PDF_DIR / uploaded_file.name
        if not save_path.exists():
            save_path.write_bytes(uploaded_file.getvalue())
        if st.button("Ingest uploaded PDF", key="ingest_btn"):
            with st.spinner(f"Ingesting {uploaded_file.name}..."):
                count = ingest_pdf(save_path)
            if count > 0:
                st.success(f"Ingested {count} chunks from {uploaded_file.name}")
            else:
                st.info("Already ingested.")

    # ── Images & Screenshots ──────────────────────────────────────────────
    st.subheader("Images & Screenshots")
    IMAGES_DIR = DATA_DIR / "images"
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    uploaded_img = st.file_uploader(
        "Upload an image",
        type=["png", "jpg", "jpeg", "gif", "webp", "bmp"],
        key="img_uploader",
    )
    if uploaded_img is not None:
        img_path = IMAGES_DIR / uploaded_img.name
        if not img_path.exists():
            img_path.write_bytes(uploaded_img.getvalue())
        st.image(img_path, caption=uploaded_img.name, width=200)
        if st.button("Ingest image", key="ingest_img_btn"):
            with st.spinner(f"Describing & indexing {uploaded_img.name}..."):
                count = ingest_image(img_path)
            if count > 0:
                st.success(f"Indexed {uploaded_img.name}")
            else:
                st.info("Already ingested.")

    if st.button("Ingest all images", key="ingest_all_imgs_btn"):
        with st.spinner("Describing & indexing all images..."):
            results = ingest_images_dir(IMAGES_DIR)
        st.success(f"Ingested: {results['ingested']}, Skipped: {results['skipped']}")
        for err in results["errors"]:
            st.error(err)

    st.divider()

    # ── Ingested Sources ──────────────────────────────────────────────────
    sources = list_sources()
    if sources:
        st.markdown("**Ingested Sources:**")
        for src in sources:
            col1, col2 = st.columns([3, 1])
            col1.markdown(f"- {src}")
            if col2.button("X", key=f"del_{src}", help=f"Delete {src}"):
                delete_source(src)
                st.rerun()

# ── Main Area: Section Viewer + Chat ──────────────────────────────────────
tab_chat, tab_section = st.tabs(["Chat", "Section Viewer"])

with tab_section:
    if selected_idx is not None:
        selected = statuses[selected_idx]
        section = load_section(selected["id"])
        if section:
            st.markdown(f"### {section['title']}")
            st.caption(f"Status: {section['status']} | Updated: {section['updated_at']}")
            st.markdown(section["text"])
        else:
            st.info(f"Section '{selected['id']}' has not been drafted yet.")

with tab_chat:
    st.markdown("### Research Chat")
    st.caption("Ask questions about your reference materials. Responses include source citations.")

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

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

        st.session_state.messages.append({"role": "assistant", "content": response_text})
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        st.session_state.chat_history.append({"role": "assistant", "content": response_text})
