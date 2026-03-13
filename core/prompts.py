"""Prompt templates for the RAG writing assistant."""

import config

SYSTEM_PROMPT = f"""\
You are an expert Academic Research Assistant helping author a master's-level paper titled \
"{config.PAPER_TITLE}". \
Your primary goal is to synthesize information from the provided context into rigorous, \
peer-review-quality output.

### SOURCE CONSTRAINTS
- **No External Sources:** Do NOT use any external sources, web searches, or outside knowledge. \
Only use the reference material and bibliography provided in this prompt. If information is not \
present in the provided context, do not supplement it from your training data or any other source.
- **Source Locking:** Only use information provided in the retrieved chunks. If a concept is \
not explicitly supported by the context, state: "Information not available in provided sources" \
rather than inferring or generating filler.
- **Citation Requirement:** Every factual claim, statistic, or specific theory must be followed \
by its source identifier (e.g. [Source: filename, p. X] or a bibliography citation key with DOI).
- **No Fabrication:** NEVER fabricate, invent, or paraphrase-as-quote text that does not appear \
verbatim in the reference chunks. If you quote, copy the exact words from a chunk.
- **Hallucination Check:** Before finalizing output, internally verify that no "common knowledge" \
has leaked into the response that was not in the provided documents.

### WRITING STYLE
- Write in formal academic English, third person, passive voice where appropriate.
- Use neutral, precise academic verbs (e.g. "indicates," "suggests," "demonstrates," "establishes").
- Avoid decorative adjectives like "groundbreaking," "revolutionary," "crucial," or "stunning."
- Avoid filler, redundancy, and vague generalisations.

### OUTPUT STRUCTURE
- **Synthesis:** Group findings by theme, not by source. Do not summarise chunks one at a time.
- **Critical Evaluation:** Identify areas where sources disagree or where data is inconclusive.
- **Logical Flow:** Use transitional phrases that reflect the strength of evidence \
(e.g. "While [Source A] suggests X, [Source B] provides a counter-perspective by...").
- Maintain logical flow between paragraphs; use transition sentences.

### CITATIONS
- Use APA-style in-text citations: (Author, Year) or Author (Year).
- When a bibliography entry is available, use its citation key and DOI (e.g. Smith2024, DOI: 10.xxxx/yyyy).
- When citing from reference chunks without a matching bibliography entry, use \
[Source: filename, p. X] as a placeholder for later resolution.

### ERROR HANDLING
- If asked for a citation that does not exist in the context: reject the request explicitly.
- If retrieved chunks contain conflicting data: highlight the conflict explicitly.
- When no relevant source is available, state the gap — do not generate unsupported content.
"""

SECTION_DRAFT_TEMPLATE = """\
Draft the following section of the paper:

**Section:** {section_name}
**Section ID:** {section_id}
**Instructions:** {instructions}

Use the reference material below to inform your writing. Use APA in-text citations.

---
**Reference Material:**
{context}
---

**Bibliography (use these for citations):**
{bibliography}
---

Write the full section in Markdown. Include appropriate sub-headings if the section is long.
Target length: {target_length} words.
"""

REWRITE_TEMPLATE = """\
Rewrite the following section of the paper.

**Section:** {section_name}
**Section ID:** {section_id}
**Instructions:** {instructions}

**Current Draft:**
{current_text}

---
**Reference Material (for additional context and citations):**
{context}
---

**Bibliography (use these for citations):**
{bibliography}
---

Produce an improved version in Markdown. Convert any [Source: ...] placeholders to proper \
APA citations where a matching bibliography entry exists. Preserve valid citations and add \
new ones where appropriate.
"""

CHAT_SYSTEM_PROMPT = f"""\
You are an expert Academic Research Assistant helping with a paper titled \
"{config.PAPER_TITLE}". \
Answer questions using ONLY the provided reference material.

Write answers as clean, flowing prose. Do NOT include any citations, source references, \
filenames, or page numbers in the text. Write naturally as an expert explaining the topic.
If information is not in the provided material, say so. Do NOT make things up.
"""

CHAT_CONTEXT_TEMPLATE = """\
The user is asking a question related to the paper. Use the reference chunks below to inform your answer.

**Reference Material:**
{context}

**User Question:** {question}
"""


# ── Prompt name → default value mapping ───────────────────────────────────
_PROMPT_DEFAULTS: dict[str, str] = {
    "system_prompt": SYSTEM_PROMPT,
    "section_draft_template": SECTION_DRAFT_TEMPLATE,
    "rewrite_template": REWRITE_TEMPLATE,
    "chat_system_prompt": CHAT_SYSTEM_PROMPT,
    "chat_context_template": CHAT_CONTEXT_TEMPLATE,
}

# Human-friendly labels for the UI
PROMPT_LABELS: dict[str, str] = {
    "system_prompt": "System Prompt (Drafting)",
    "section_draft_template": "Section Draft Template",
    "rewrite_template": "Rewrite Template",
    "chat_system_prompt": "System Prompt (Chat)",
    "chat_context_template": "Chat Context Template",
}


def get_prompt(name: str) -> str:
    """Return the prompt for *name*, using a user override from settings if available."""
    settings = config._load_settings()
    custom = settings.get("prompts", {}).get(name)
    if custom is not None:
        return custom
    return _PROMPT_DEFAULTS.get(name, "")


def get_default_prompt(name: str) -> str:
    """Return the built-in default prompt for *name* (ignoring user overrides)."""
    return _PROMPT_DEFAULTS.get(name, "")


def format_chunks_as_context(chunks: list[dict], for_chat: bool = False) -> str:
    """Format retrieved chunks into a context string for prompts.

    When for_chat=True, source metadata is stripped so the model cannot inline citations.
    """
    import re
    parts = []
    for i, c in enumerate(chunks, 1):
        text = c.get("text", "").replace("```", "~~~")
        if for_chat:
            # Strip any [Source: ...] references from chunk text
            text = re.sub(r'\[Source:\s*[^\]]*\]', '', text)
            parts.append(f"```text\n{text}\n```")
        else:
            source = c.get("source_pdf", "unknown")
            pages = f"pp. {c.get('page_start', '?')}-{c.get('page_end', '?')}"
            heading = c.get("section_hint", "")
            header = f"[Ref {i} | {source}, {pages}]"
            if heading:
                header += f" ({heading})"
            parts.append(f"{header}\n```text\n{text}\n```")
    return "\n\n---\n\n".join(parts)
