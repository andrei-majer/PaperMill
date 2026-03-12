"""Embedding module — jinaai/jina-embeddings-v3 via sentence-transformers."""

import logging

logging.getLogger("transformers_modules.jinaai").setLevel(logging.ERROR)

import transformers
transformers.logging.set_verbosity_error()

import torch
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL, EMBEDDING_DIM

_model: SentenceTransformer | None = None


def load_model() -> SentenceTransformer:
    """Load the Jina v3 model onto GPU (or CPU fallback). Cached after first call."""
    global _model
    if _model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model = SentenceTransformer(EMBEDDING_MODEL, trust_remote_code=True, device=device)
    return _model


def embed_passages(texts: list[str]) -> list[list[float]]:
    """Embed a list of passage texts using the retrieval.passage task hint.
    Returns list of 1024-dim vectors."""
    model = load_model()
    embeddings = model.encode(
        texts,
        task="retrieval.passage",
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    return embeddings.tolist()


def embed_query(query: str) -> list[float]:
    """Embed a single query using the retrieval.query task hint.
    Returns a 1024-dim vector."""
    model = load_model()
    embedding = model.encode(
        [query],
        task="retrieval.query",
        normalize_embeddings=True,
    )
    return embedding[0].tolist()
