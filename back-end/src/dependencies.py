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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.services.chat_service import ChatService
    from src.services.gemini_service import GeminiService
    from src.services.qdrant_service import QdrantService


@lru_cache(maxsize=1)
def get_gemini_service() -> "GeminiService":
    from src.services.gemini_service import GeminiService
    return GeminiService()


@lru_cache(maxsize=1)
def get_qdrant_service() -> "QdrantService":
    from src.services.qdrant_service import QdrantService
    return QdrantService(gemini_service=get_gemini_service())


@lru_cache(maxsize=1)
def get_chat_service() -> "ChatService":
    from src.services.chat_service import ChatService
    return ChatService(
        gemini_service=get_gemini_service(),
        qdrant_service=get_qdrant_service(),
    )
