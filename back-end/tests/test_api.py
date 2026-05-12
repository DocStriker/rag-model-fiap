"""
Testes automatizados — serviços e rotas.

Justificativa (Requisito 3.6):
Testes unitários com mocks garantem que a lógica de negócio funciona
independentemente de serviços externos (Gemini, Qdrant).
Os testes de rota validam contratos HTTP sem depender de infra.

Como rodar:
    pytest tests/ -v
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture()
def mock_gemini_service():
    svc = MagicMock()
    svc.call_llm.return_value = "Resposta simulada do Gemini."
    svc.get_embedding.return_value = [0.1] * 1536
    return svc


@pytest.fixture()
def mock_qdrant_service():
    svc = MagicMock()
    svc.list_collections.return_value = ["docs"]
    svc.collection_exists.return_value = True
    svc.search.return_value = [
        {
            "id": "abc",
            "score": 0.92,
            "payload": {
                "text": "Conteúdo do documento.",
                "source": "arquivo.pdf",
                "page": 1,
            },
        }
    ]
    return svc


@pytest.fixture()
def client(mock_gemini_service, mock_qdrant_service):
    """TestClient com dependências substituídas por mocks."""
    # Import app first
    from src.main import app
    from src.dependencies import get_gemini_service, get_chat_service, get_qdrant_service
    from src.services.chat_service import ChatService

    def _chat_service():
        return ChatService(
            gemini_service=mock_gemini_service,
            qdrant_service=mock_qdrant_service,
        )

    # Apply overrides after app is imported
    app.dependency_overrides[get_gemini_service] = lambda: mock_gemini_service
    app.dependency_overrides[get_qdrant_service] = lambda: mock_qdrant_service
    app.dependency_overrides[get_chat_service] = _chat_service

    with TestClient(app) as c:
        yield c

    app.dependency_overrides.clear()


# ------------------------------------------------------------------
# Testes de rota — /health
# ------------------------------------------------------------------


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


# ------------------------------------------------------------------
# Testes de rota — /chat
# ------------------------------------------------------------------


def test_chat_direct(client, mock_gemini_service):
    r = client.post("/chat", json={"message": "Olá"})
    assert r.status_code == 200
    assert r.json()["response"] == "Resposta simulada do Gemini."
    mock_gemini_service.call_llm.assert_called_once()


def test_chat_empty_message(client):
    r = client.post("/chat", json={"message": ""})
    # Pydantic valida min_length=1 → 422
    assert r.status_code == 422


def test_chat_with_collection(client, mock_gemini_service, mock_qdrant_service):
    r = client.post(
        "/chat",
        json={"message": "O que diz o documento?", "collection": "docs"},
    )
    assert r.status_code == 200
    mock_qdrant_service.search.assert_called_once_with("docs", "O que diz o documento?", limit=4)
    mock_gemini_service.call_llm.assert_called_once()


def test_chat_with_history(client, mock_gemini_service):
    history = [
        {"role": "user", "content": "Quem é você?"},
        {"role": "assistant", "content": "Sou o FIAP AI."},
    ]
    r = client.post(
        "/chat",
        json={"message": "Continue.", "history": history},
    )
    assert r.status_code == 200


# ------------------------------------------------------------------
# Testes de rota — /collections
# ------------------------------------------------------------------


def test_list_collections(client):
    r = client.get("/collections")
    assert r.status_code == 200
    assert "docs" in r.json()["collections"]


def test_create_collection_conflict(client, mock_qdrant_service):
    mock_qdrant_service.collection_exists.return_value = True
    r = client.post("/collections", json={"name": "docs"})
    assert r.status_code == 409


def test_create_collection_success(client, mock_qdrant_service):
    mock_qdrant_service.collection_exists.return_value = False
    mock_qdrant_service.list_collections.return_value = ["nova"]
    r = client.post("/collections", json={"name": "nova"})
    assert r.status_code == 201
    assert "nova" in r.json()["collections"]


def test_create_collection_invalid_name(client):
    r = client.post("/collections", json={"name": "nome inválido!"})
    assert r.status_code == 422


def test_delete_collection_not_found(client, mock_qdrant_service):
    mock_qdrant_service.collection_exists.return_value = False
    r = client.delete("/collections/inexistente")
    assert r.status_code == 404


def test_delete_collection_success(client, mock_qdrant_service):
    mock_qdrant_service.collection_exists.return_value = True
    r = client.delete("/collections/docs")
    assert r.status_code == 204


def test_upload_unsupported_extension(client):
    r = client.post(
        "/collections/docs/documents",
        files={"file": ("arquivo.csv", b"col1,col2\n1,2", "text/csv")},
    )
    assert r.status_code == 415


def test_upload_txt_success(client, mock_qdrant_service):
    mock_qdrant_service.upload_file.return_value = 3
    r = client.post(
        "/collections/docs/documents",
        files={"file": ("notas.txt", b"Conte\xc3\xbado do documento.", "text/plain")},
    )
    assert r.status_code == 200
    assert r.json()["chunks"] == 3


# ------------------------------------------------------------------
# Testes unitários — ChatService
# ------------------------------------------------------------------


def test_chat_service_empty_message(mock_gemini_service):
    from src.services.chat_service import ChatService

    svc = ChatService(gemini_service=mock_gemini_service)
    result = svc.generate_response("   ")
    assert result == "Por favor, envie uma mensagem válida."
    mock_gemini_service.call_llm.assert_not_called()


def test_chat_service_rag_no_results(mock_gemini_service, mock_qdrant_service):
    from src.services.chat_service import ChatService

    mock_qdrant_service.search.return_value = []
    svc = ChatService(
        gemini_service=mock_gemini_service,
        qdrant_service=mock_qdrant_service,
    )
    result = svc.generate_response("Pergunta", collection="vazia")
    assert "Não encontrei" in result
    mock_gemini_service.call_llm.assert_not_called()


# ------------------------------------------------------------------
# Testes unitários — QdrantService._chunk_text
# ------------------------------------------------------------------


def test_chunk_text_basic():
    from src.services.qdrant_service import QdrantService

    text = "Esta é a primeira sentença. Esta é a segunda. Esta é a terceira."
    chunks = QdrantService._chunk_text(text, chunk_size=50, overlap=10)
    assert len(chunks) >= 1
    for chunk in chunks:
        assert chunk.strip()


def test_chunk_text_invalid_params():
    from src.services.qdrant_service import QdrantService

    with pytest.raises(ValueError):
        QdrantService._chunk_text("texto", chunk_size=10, overlap=20)


def test_chunk_text_long_sentence_splits():
    from src.services.qdrant_service import QdrantService

    text = "A" * 120 + "."
    chunks = QdrantService._chunk_text(text, chunk_size=50, overlap=10)
    assert len(chunks) > 1
    assert all(len(chunk) <= 50 for chunk in chunks)


def test_chunk_text_empty_returns_empty():
    from src.services.qdrant_service import QdrantService

    chunks = QdrantService._chunk_text("", chunk_size=50, overlap=10)
    assert chunks == []
