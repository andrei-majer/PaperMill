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


# ── Tree Index Prompts ────────────────────────────────────────────────────

TREE_TOC_DETECTION_PROMPT = """\
You are analyzing the first pages of an academic PDF document. Determine whether these pages \
contain a Table of Contents (TOC).

**Pages:**
{pages_text}

**Instructions:**
1. If a TOC is present, extract it as a JSON array of sections. Each section has:
   - "title": the section/chapter title as written
   - "start_page": the page number listed in the TOC (integer)
   - "summary": leave as empty string (will be filled later)
   - "children": nested subsections in the same format, or empty array
2. If NO TOC is found, return exactly: {{"has_toc": false}}
3. If a TOC IS found, return: {{"has_toc": true, "sections": [...]}}

Return ONLY valid JSON, no markdown fences, no explanation.
"""

TREE_STRUCTURE_GENERATION_PROMPT = """\
You are analyzing an academic PDF document to build a hierarchical structure index. \
The document has {total_pages} pages.

**Page summaries:**
{pages_text}

**Instructions:**
Build a hierarchical JSON structure representing the document's organization. Each node has:
- "title": descriptive section title
- "summary": 1-2 sentence summary of what this section covers
- "start_page": first page of the section (integer, 1-indexed)
- "end_page": last page of the section (integer, 1-indexed)
- "children": array of subsections (same format), or empty array for leaf nodes

Guidelines:
- Top-level sections should be major divisions (Abstract, Introduction, Literature Review, Methods, etc.)
- Subdivide sections that span more than 15 pages into meaningful children
- Page ranges must not overlap between siblings and must cover the full document
- Every page in the document must belong to exactly one leaf section

Return ONLY valid JSON as a single object with key "sections" containing the array. \
No markdown fences, no explanation.
"""

TREE_SECTION_SUBDIVISION_PROMPT = """\
You are analyzing a large section of an academic document that needs to be subdivided \
into meaningful subsections.

**Section:** {section_title}
**Pages {start_page}-{end_page} ({page_count} pages)**

**Page content:**
{pages_text}

**Instructions:**
Subdivide this section into 2-6 meaningful subsections. Each subsection has:
- "title": descriptive subsection title
- "summary": 1-2 sentence summary
- "start_page": first page (integer, 1-indexed)
- "end_page": last page (integer, 1-indexed)
- "children": empty array

Guidelines:
- Subsections must cover the full page range [{start_page}-{end_page}] with no gaps or overlaps
- Split on natural topic boundaries, not arbitrary page counts
- Each subsection should be a coherent unit

Return ONLY valid JSON as a single object with key "subsections" containing the array. \
No markdown fences, no explanation.
"""

TREE_RETRIEVAL_PROMPT = """\
You are a research assistant helping find relevant sections in academic documents.

**Query:** {query}

**Document structures:**
{structures}

**Instructions:**
For each document, identify the sections most relevant to the query. Return a JSON object:
{{
  "selections": [
    {{
      "document": "filename.pdf",
      "sections": [
        {{"title": "section title", "start_page": N, "end_page": M, "relevance": "brief reason"}}
      ]
    }}
  ]
}}

Guidelines:
- Select 1-3 most relevant sections per document
- Prefer specific subsections over broad parent sections
- Only select sections that are genuinely relevant to the query
- If no section in a document is relevant, omit that document entirely

Return ONLY valid JSON, no markdown fences, no explanation.
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
