"""
Injeção de dependências centralizada.

Justificativa (Requisito 3.1):
Concentrar a criação de serviços em um único módulo evita instâncias
duplicadas e facilita a substituição em testes (basta sobrescrever o
Depends). O padrão Singleton via lru_cache garante que cada serviço seja
criado apenas uma vez por processo.
"""

from __future__ import annotations

from functools import lru_cache

from src.services.anthropic_service import AnthropicService
from src.services.chat_service import ChatService
from src.services.qdrant_service import QdrantService


@lru_cache(maxsize=1)
def get_anthropic_service() -> AnthropicService:
    return AnthropicService()


@lru_cache(maxsize=1)
def get_qdrant_service() -> QdrantService:
    return QdrantService(anthropic_service=get_anthropic_service())


@lru_cache(maxsize=1)
def get_chat_service() -> ChatService:
    return ChatService(
        anthropic_service=get_anthropic_service(),
        qdrant_service=get_qdrant_service(),
    )
