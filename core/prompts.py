"""Prompt templates for the RAG writing assistant."""

import config

SYSTEM_PROMPT = f"""\
You are an expert Academic Research Assistant helping author a master's-level paper titled \
"{config.PAPER_TITLE}". \
Your primary goal is to synthesize information from the provided context into rigorous, \
peer-review-quality output.

### SOURCE CONSTRAINTS
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

CHAT_CONTEXT_TEMPLATE = """\
The user is asking a question related to the paper. Use the reference chunks below to inform your answer.

**Reference Chunks:**
{context}

**Bibliography:**
{bibliography}

**User Question:** {question}

Answer thoroughly, citing sources with [Source: filename, p. X]. When relevant, include DOI references from the bibliography.
"""


def format_chunks_as_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a context string for prompts."""
    parts = []
    for i, c in enumerate(chunks, 1):
        source = c.get("source_pdf", "unknown")
        pages = f"pp. {c.get('page_start', '?')}-{c.get('page_end', '?')}"
        heading = c.get("section_hint", "")
        header = f"[Chunk {i} | {source}, {pages}]"
        if heading:
            header += f" ({heading})"
        text = c.get("text", "").replace("```", "~~~")
        parts.append(f"{header}\n```text\n{text}\n```")
    return "\n\n---\n\n".join(parts)
