"""Tests for multi-provider embedding dispatch."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from unittest.mock import patch, MagicMock


def test_embed_local_passages(monkeypatch):
    import config
    monkeypatch.setattr(config, "EMBEDDING_PROVIDER", "local")
    monkeypatch.setattr(config, "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    import core.embedder as emb
    emb._local_model = None

    mock_model = MagicMock()
    mock_model.encode.return_value = MagicMock(tolist=lambda: [[0.1]*384, [0.2]*384])
    mock_model.get_sentence_embedding_dimension.return_value = 384

    with patch("core.embedder.SentenceTransformer", return_value=mock_model) as mock_cls:
        result = emb.embed_passages(["hello", "world"])
        # Should NOT pass trust_remote_code for non-Jina model
        call_kwargs = mock_cls.call_args
        assert call_kwargs[1].get("trust_remote_code") is not True

    assert len(result) == 2
    assert len(result[0]) == 384


def test_embed_local_jina_uses_trust_remote_code(monkeypatch):
    import config
    monkeypatch.setattr(config, "EMBEDDING_PROVIDER", "local")
    monkeypatch.setattr(config, "EMBEDDING_MODEL", "jinaai/jina-embeddings-v3")

    import core.embedder as emb
    emb._local_model = None

    mock_model = MagicMock()
    mock_model.encode.return_value = MagicMock(tolist=lambda: [[0.1]*1024])
    mock_model.get_sentence_embedding_dimension.return_value = 1024

    with patch("core.embedder.SentenceTransformer", return_value=mock_model) as mock_cls:
        emb.embed_passages(["test"])
        assert mock_cls.call_args[1].get("trust_remote_code") is True
        # Verify Jina task hint passed to encode
        encode_kwargs = mock_model.encode.call_args[1]
        assert encode_kwargs.get("task") == "retrieval.passage"


def test_embed_query_returns_single_vector(monkeypatch):
    import config
    monkeypatch.setattr(config, "EMBEDDING_PROVIDER", "local")
    monkeypatch.setattr(config, "EMBEDDING_MODEL", "jinaai/jina-embeddings-v3")

    import core.embedder as emb
    emb._local_model = None

    import numpy as np
    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[0.5]*1024])
    mock_model.get_sentence_embedding_dimension.return_value = 1024

    with patch("core.embedder.SentenceTransformer", return_value=mock_model):
        result = emb.embed_query("test query")

    assert isinstance(result, list)
    assert len(result) == 1024


def test_embed_ollama(monkeypatch):
    import config
    monkeypatch.setattr(config, "EMBEDDING_PROVIDER", "ollama")
    monkeypatch.setattr(config, "EMBEDDING_MODEL", "nomic-embed-text")
    monkeypatch.setattr(config, "OLLAMA_URL", "http://localhost:11434")

    import core.embedder as emb
    import json

    fake_response = json.dumps({"embeddings": [[0.1]*768, [0.2]*768]}).encode()
    mock_resp = MagicMock()
    mock_resp.read.return_value = fake_response
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)

    with patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
        result = emb.embed_passages(["hello", "world"])
        # Verify URL and payload
        call_args = mock_urlopen.call_args[0][0]  # first positional arg is the Request
        assert "/api/embed" in call_args.full_url
        import json as _json
        sent_payload = _json.loads(call_args.data.decode())
        assert sent_payload["model"] == "nomic-embed-text"
        assert sent_payload["input"] == ["hello", "world"]

    assert len(result) == 2
    assert len(result[0]) == 768


def test_embed_openai(monkeypatch):
    import config
    monkeypatch.setattr(config, "EMBEDDING_PROVIDER", "openai")
    monkeypatch.setattr(config, "EMBEDDING_MODEL", "text-embedding-3-small")
    monkeypatch.setattr(config, "OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(config, "EMBEDDING_API_KEY", "")

    import core.embedder as emb

    mock_embedding1 = MagicMock()
    mock_embedding1.embedding = [0.1]*1536
    mock_embedding2 = MagicMock()
    mock_embedding2.embedding = [0.2]*1536
    mock_response = MagicMock()
    mock_response.data = [mock_embedding1, mock_embedding2]

    mock_client = MagicMock()
    mock_client.embeddings.create.return_value = mock_response

    with patch("openai.OpenAI", return_value=mock_client):
        result = emb.embed_passages(["hello", "world"])
        mock_client.embeddings.create.assert_called_once_with(model="text-embedding-3-small", input=["hello", "world"])

    assert len(result) == 2
    assert len(result[0]) == 1536


def test_get_embedding_dim_openai(monkeypatch):
    import config
    monkeypatch.setattr(config, "EMBEDDING_PROVIDER", "openai")
    monkeypatch.setattr(config, "EMBEDDING_MODEL", "text-embedding-3-small")

    from core.embedder import get_embedding_dim
    assert get_embedding_dim() == 1536

    monkeypatch.setattr(config, "EMBEDDING_MODEL", "text-embedding-3-large")
    assert get_embedding_dim() == 3072
