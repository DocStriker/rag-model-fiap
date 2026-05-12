"""
Serviço de integração com Gemini via Google GenAI SDK.

Essa implementação usa a biblioteca oficial google-genai para acesso
gratuito aos embeddings do Gemini, mantendo compatibilidade com o
fluxo de RAG existente.
"""

from __future__ import annotations

import hashlib
import os
from functools import lru_cache

import structlog
import google.genai as genai
from google.genai import types

logger = structlog.get_logger(__name__)

DEFAULT_CHAT_MODEL = "gemini-3.1-flash-lite-preview"
DEFAULT_EMBEDDING_MODEL = "text-embedding-005"
EMBEDDING_DIMENSION = 1024  # Recomendado para text-embedding-005
_EMBEDDING_CACHE_SIZE = 512


class GeminiService:
    """
    Encapsula chamadas para o Gemini usando a biblioteca oficial google-genai.

    Melhorias implementadas:
    - Cache LRU de embeddings por hash SHA-256 do texto.
    - Logging estruturado em cada operação.
    - Tratamento explícito de erros com mensagens claras.
    - Embeddings gratuitos via Gemini API.
    """

    def __init__(
        self,
        gemini_api_key: str | None = None,
    ) -> None:
        api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY não definida. Configure no arquivo .env."
            )

        self._client = genai.Client(api_key=api_key)
        logger.info("gemini_service.initialized", model=DEFAULT_CHAT_MODEL)

    def call_llm(
        self,
        prompt: str,
        model: str = DEFAULT_CHAT_MODEL,
        system: str | None = None,
        temperature: float = 0.7,
        history: list[dict] | None = None,
        max_tokens: int = 2048,
    ) -> str:
        messages: list[types.Content] = []
        if system:
            messages.append(types.Content(
                role="user",
                parts=[types.Part.from_text(f"Sistema: {system}")]
            ))

        if history:
            for msg in history:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                messages.append(types.Content(
                    role=role,
                    parts=[types.Part.from_text(content)]
                ))

        messages.append(types.Content(
            role="user",
            parts=[types.Part.from_text(prompt)]
        ))

        log = logger.bind(model=model, history_len=len(history or []))
        log.info("llm.request")

        try:
            response = self._client.models.generate_content(
                model=model,
                contents=messages,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                )
            )
            text = response.text or ""
            log.info("llm.response", output_tokens=len(text.split()))
            return text.strip()
        except Exception as exc:
            log.error("llm.error", error=str(exc))
            raise RuntimeError(
                f"Erro na chamada do Gemini: {exc}"
            ) from exc

    def get_embedding(
        self,
        text: str,
        model: str = DEFAULT_EMBEDDING_MODEL,
    ) -> list[float]:
        return self._cached_embedding(text, model)

    def _cached_embedding(self, text: str, model: str) -> list[float]:
        cache_key = _embedding_cache_key(text, model)
        cached = _embedding_lru_cache(cache_key)
        if cached is not None:
            logger.debug("embedding.cache_hit", key=cache_key[:12])
            return cached

        logger.info("embedding.request", model=model, text_len=len(text))
        try:
            response = self._client.models.embed_content(
                model=model,
                contents=text,
                config=types.EmbedContentConfig(
                    output_dimensionality=EMBEDDING_DIMENSION
                )
            )
            vector = response.embeddings[0].values
            _store_embedding(cache_key, vector)
            logger.info("embedding.done", dim=len(vector))
            return vector
        except Exception as exc:
            logger.error("embedding.error", error=str(exc))
            raise RuntimeError(
                f"Erro na geração de embedding: {exc}"
            ) from exc


_embedding_store: dict[str, list[float]] = {}
_embedding_keys: list[str] = []


def _embedding_cache_key(text: str, model: str) -> str:
    return hashlib.sha256(f"{model}:{text}".encode()).hexdigest()


def _embedding_lru_cache(key: str) -> list[float] | None:
    return _embedding_store.get(key)


def _store_embedding(key: str, vector: list[float]) -> None:
    if len(_embedding_keys) >= _EMBEDDING_CACHE_SIZE:
        oldest = _embedding_keys.pop(0)
        _embedding_store.pop(oldest, None)
    _embedding_store[key] = vector
    _embedding_keys.append(key)
