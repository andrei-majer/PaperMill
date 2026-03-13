"""Multi-provider embedding module — local (sentence-transformers), Ollama, OpenAI."""

import json
import logging
import urllib.request
import urllib.error

import config

logger = logging.getLogger(__name__)

# ── Lazy imports (at module level so they can be mocked in tests) ─────────
try:
    logging.getLogger("transformers_modules.jinaai").setLevel(logging.ERROR)
    import transformers
    transformers.logging.set_verbosity_error()
    import torch
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None
    torch = None

# ── Local provider (sentence-transformers) ────────────────────────────────

_local_model = None
_local_model_name: str = ""


def _load_local_model():
    global _local_model, _local_model_name
    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers not installed. Install it or switch to Ollama/OpenAI embeddings.")
    if _local_model is None or _local_model_name != config.EMBEDDING_MODEL:
        device = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"
        kwargs = {"device": device}
        if "jina" in config.EMBEDDING_MODEL.lower():
            kwargs["trust_remote_code"] = True
        _local_model = SentenceTransformer(config.EMBEDDING_MODEL, **kwargs)
        _local_model_name = config.EMBEDDING_MODEL
    return _local_model


def _is_jina() -> bool:
    return "jina" in config.EMBEDDING_MODEL.lower()


def _embed_local_passages(texts: list[str]) -> list[list[float]]:
    model = _load_local_model()
    kwargs = {
        "batch_size": 32,
        "show_progress_bar": True,
        "normalize_embeddings": True,
    }
    if _is_jina():
        kwargs["task"] = "retrieval.passage"
    return model.encode(texts, **kwargs).tolist()


def _embed_local_query(query: str) -> list[float]:
    model = _load_local_model()
    kwargs = {"normalize_embeddings": True}
    if _is_jina():
        kwargs["task"] = "retrieval.query"
    return model.encode([query], **kwargs)[0].tolist()


# ── Ollama provider ──────────────────────────────────────────────────────

def _embed_ollama(texts: list[str]) -> list[list[float]]:
    url = f"{config.OLLAMA_URL}/api/embed"
    all_embeddings = []

    for i in range(0, len(texts), 32):
        batch = texts[i:i + 32]
        payload = json.dumps({"model": config.EMBEDDING_MODEL, "input": batch}).encode()
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode())
            all_embeddings.extend(data["embeddings"])
        except urllib.error.URLError as e:
            raise RuntimeError(f"Ollama unreachable at {url}: {e}") from e
        except KeyError:
            raise RuntimeError(f"Ollama returned unexpected response format from {url}")

    return all_embeddings


# ── OpenAI provider ──────────────────────────────────────────────────────

_OPENAI_DIMS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


def _embed_openai(texts: list[str]) -> list[list[float]]:
    from openai import OpenAI
    api_key = config.EMBEDDING_API_KEY or config.OPENAI_API_KEY
    if not api_key:
        raise RuntimeError("No API key for OpenAI embeddings. Set OPENAI_API_KEY or EMBEDDING_API_KEY.")
    client = OpenAI(api_key=api_key)
    response = client.embeddings.create(model=config.EMBEDDING_MODEL, input=texts)
    return [e.embedding for e in response.data]


# ── Dimension detection ──────────────────────────────────────────────────

def get_embedding_dim() -> int:
    """Return the embedding dimension for the current provider/model."""
    provider = config.EMBEDDING_PROVIDER
    if provider == "openai":
        dim = _OPENAI_DIMS.get(config.EMBEDDING_MODEL)
        if dim:
            return dim
    if config.EMBEDDING_DIM and config.EMBEDDING_DIM > 0:
        return config.EMBEDDING_DIM
    return 1024


def detect_and_save_dim() -> int:
    """Run a test embed to detect dimension, save to settings, return dim."""
    provider = config.EMBEDDING_PROVIDER
    if provider == "openai":
        dim = _OPENAI_DIMS.get(config.EMBEDDING_MODEL, 1536)
    elif provider == "ollama":
        vecs = _embed_ollama(["dimension test"])
        dim = len(vecs[0])
    else:
        model = _load_local_model()
        dim = model.get_sentence_embedding_dimension()
    config.EMBEDDING_DIM = dim
    config.save_settings({"embedding_dim": dim})
    return dim


# ── Public dispatch API ──────────────────────────────────────────────────

def embed_passages(texts: list[str]) -> list[list[float]]:
    """Embed passage texts using the configured provider."""
    provider = config.EMBEDDING_PROVIDER
    if provider == "ollama":
        return _embed_ollama(texts)
    if provider == "openai":
        return _embed_openai(texts)
    return _embed_local_passages(texts)


def embed_query(query: str) -> list[float]:
    """Embed a single query. Returns a single vector."""
    provider = config.EMBEDDING_PROVIDER
    if provider == "ollama":
        return _embed_ollama([query])[0]
    if provider == "openai":
        return _embed_openai([query])[0]
    return _embed_local_query(query)
