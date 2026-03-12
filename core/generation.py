"""LLM generation with RAG context — supports Claude and Ollama."""

import json
import logging
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

import config
from config import (
    ANTHROPIC_API_KEY, OLLAMA_URL,
    SCANNER_SCAN_CHAT_INPUT,
)
from core.prompts import (
    SYSTEM_PROMPT,
    SECTION_DRAFT_TEMPLATE,
    REWRITE_TEMPLATE,
    CHAT_CONTEXT_TEMPLATE,
    format_chunks_as_context,
)
from core.paper_structure import (
    get_section_title,
    get_section_target_words,
    save_section,
    load_section,
)
from core.bibliography import format_refs_for_prompt

logger = logging.getLogger(__name__)


def _get_model(role: str) -> str:
    """Return the model name for *role* ('draft' or 'polish') based on the
    current config.LLM_PROVIDER at call time, so runtime provider switches
    are respected.
    """
    if config.LLM_PROVIDER == "ollama":
        return config.OLLAMA_POLISH_MODEL if role == "polish" else config.OLLAMA_DRAFT_MODEL
    return config.CLAUDE_POLISH_MODEL if role == "polish" else config.CLAUDE_DRAFT_MODEL


@dataclass
class GenerationStats:
    """Performance stats from an LLM generation call."""
    provider: str = ""
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    elapsed_sec: float = 0.0
    tokens_per_sec: float = 0.0


# ── Provider dispatch ─────────────────────────────────────────────────────

def _generate(model: str, system: str, messages: list[dict], max_tokens: int = 4096) -> tuple[str, GenerationStats]:
    """Generate text using the configured LLM provider. Returns (text, stats)."""
    # ── Fix 3: rough token guard ───────────────────────────────────────────
    # Estimate ~4 chars per token as a conservative heuristic.
    estimated_prompt_tokens = (
        len(system) + sum(len(m.get("content", "")) for m in messages)
    ) // 4
    estimated_total = estimated_prompt_tokens + max_tokens
    if estimated_total > config.MAX_CONTEXT_TOKENS:
        logger.warning(
            "Estimated context (%d tokens) exceeds MAX_CONTEXT_TOKENS (%d). "
            "Truncating message content to fit.",
            estimated_total,
            config.MAX_CONTEXT_TOKENS,
        )
        # How many chars are available for message content after system + max_tokens headroom.
        available_tokens = config.MAX_CONTEXT_TOKENS - max_tokens - len(system) // 4
        available_chars = max(available_tokens * 4, 0)
        truncated: list[dict] = []
        remaining = available_chars
        # Walk messages in reverse so the most-recent context is kept.
        for msg in reversed(messages):
            content = msg.get("content", "")
            if remaining <= 0:
                truncated.insert(0, {**msg, "content": ""})
            elif len(content) <= remaining:
                truncated.insert(0, msg)
                remaining -= len(content)
            else:
                truncated.insert(0, {**msg, "content": content[:remaining]})
                remaining = 0
        messages = truncated

    if config.LLM_PROVIDER == "ollama":
        return _generate_ollama(model, system, messages, max_tokens)
    return _generate_claude(model, system, messages, max_tokens)


def _generate_claude(model: str, system: str, messages: list[dict], max_tokens: int) -> tuple[str, GenerationStats]:
    import anthropic
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    t0 = time.perf_counter()
    with client.messages.stream(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=messages,
    ) as stream:
        response = stream.get_final_message()
    elapsed = time.perf_counter() - t0

    text = next((b.text for b in response.content if b.type == "text"), "")
    usage = response.usage
    prompt_tokens = getattr(usage, "input_tokens", 0)
    completion_tokens = getattr(usage, "output_tokens", 0)
    total = prompt_tokens + completion_tokens
    tps = completion_tokens / elapsed if elapsed > 0 else 0

    stats = GenerationStats(
        provider="claude",
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total,
        elapsed_sec=round(elapsed, 2),
        tokens_per_sec=round(tps, 1),
    )
    return text, stats


def _generate_ollama(model: str, system: str, messages: list[dict], max_tokens: int) -> tuple[str, GenerationStats]:
    ollama_messages = [{"role": "system", "content": system}]
    ollama_messages.extend(messages)

    payload = json.dumps({
        "model": model,
        "messages": ollama_messages,
        "stream": False,
        "options": {
            "num_predict": max_tokens,
        },
    }).encode("utf-8")

    url = f"{OLLAMA_URL}/api/chat"
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Ollama returned HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Ollama unreachable at {url} — is it running? ({exc.reason})"
        ) from exc
    elapsed = time.perf_counter() - t0

    text = data.get("message", {}).get("content", "")

    # Ollama returns timing/token stats in response
    prompt_tokens = data.get("prompt_eval_count", 0)
    completion_tokens = data.get("eval_count", 0)
    total = prompt_tokens + completion_tokens

    # Ollama provides eval_duration in nanoseconds
    eval_ns = data.get("eval_duration", 0)
    if eval_ns > 0 and completion_tokens > 0:
        tps = completion_tokens / (eval_ns / 1e9)
    elif elapsed > 0 and completion_tokens > 0:
        tps = completion_tokens / elapsed
    else:
        tps = 0

    total_ns = data.get("total_duration", 0)
    total_elapsed = total_ns / 1e9 if total_ns > 0 else elapsed

    stats = GenerationStats(
        provider="ollama",
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total,
        elapsed_sec=round(total_elapsed, 2),
        tokens_per_sec=round(tps, 1),
    )
    return text, stats


# ── Public API ────────────────────────────────────────────────────────────

def scan_chat_input(message: str) -> list:
    """Scan a chat input message for prompt injection attempts.

    Returns a list of warning strings (empty if safe).
    """
    if not SCANNER_SCAN_CHAT_INPUT:
        return []
    try:
        from core.scanner import scan_text
        result = scan_text(message, source="chat_input", location="chat:input", regex_only=True, scope="chat")
        return [f"[{t.pattern_name}] {t.matched_text}" for t in result.threats]
    except Exception:
        return []


def chat(
    message: str,
    chunks: list[dict],
    history: list[dict] | None = None,
) -> tuple[str, GenerationStats, list]:
    """RAG-powered chat: answer a question using retrieved chunks.

    Returns a 3-tuple: (response_text, stats, input_warnings).
    input_warnings is a list of warning strings from the chat input scanner (empty if safe).
    """
    input_warnings = scan_chat_input(message)

    context = format_chunks_as_context(chunks) if chunks else "No reference material available."
    bib = format_refs_for_prompt() or "No bibliography entries yet."

    user_content = CHAT_CONTEXT_TEMPLATE.format(
        context=context,
        bibliography=bib,
        question=message,
    )

    messages = []
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_content})

    text, stats = _generate(_get_model("draft"), SYSTEM_PROMPT, messages, max_tokens=4096)
    return text, stats, input_warnings


def draft_section(
    section_id: str,
    chunks: list[dict],
    instructions: str = "",
) -> tuple[str, GenerationStats]:
    """Draft a paper section using RAG context."""
    title = get_section_title(section_id)
    target_words = get_section_target_words(section_id)
    context = format_chunks_as_context(chunks) if chunks else "No reference material available."

    bib = format_refs_for_prompt() or "No bibliography entries yet."
    prompt = SECTION_DRAFT_TEMPLATE.format(
        section_name=title,
        section_id=section_id,
        instructions=instructions or "Write a comprehensive, well-structured section.",
        context=context,
        bibliography=bib,
        target_length=target_words,
    )

    text, stats = _generate(_get_model("draft"), SYSTEM_PROMPT, [{"role": "user", "content": prompt}], max_tokens=8192)

    save_section(section_id, title, text, status="draft")
    return text, stats


def rewrite_section(
    section_id: str,
    instructions: str = "",
    chunks: list[dict] | None = None,
) -> tuple[str, GenerationStats]:
    """Rewrite an existing section using the polish model."""
    section = load_section(section_id)
    if not section:
        raise ValueError(f"Section '{section_id}' has no draft to rewrite.")

    title = get_section_title(section_id)
    context = format_chunks_as_context(chunks) if chunks else "No additional reference material."

    bib = format_refs_for_prompt() or "No bibliography entries yet."
    prompt = REWRITE_TEMPLATE.format(
        section_name=title,
        section_id=section_id,
        instructions=instructions or "Improve clarity, flow, and academic rigour.",
        current_text=section["text"],
        context=context,
        bibliography=bib,
    )

    text, stats = _generate(_get_model("polish"), SYSTEM_PROMPT, [{"role": "user", "content": prompt}], max_tokens=8192)

    save_section(section_id, title, text, status="review")
    return text, stats
