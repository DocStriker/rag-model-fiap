"""
Serviço de integração com a API da Anthropic (Claude) para geração de
respostas conversacionais e geração de embeddings via OpenAI
(text-embedding-3-small), mantendo compatibilidade com o Qdrant.

Justificativa de design:
- Claude (claude-3-5-haiku) foi escolhido por ser rápido, econômico e
  excelente em português.
- Embeddings continuam via OpenAI text-embedding-3-small (dim=1536) pois
  a Anthropic não oferece endpoint de embeddings próprio. A separação em
  métodos distintos facilita trocar o provedor de embeddings no futuro.
- Credenciais exclusivamente por variáveis de ambiente (ANTHROPIC_API_KEY,
  OPENAI_API_KEY), nunca hardcoded.
"""

from __future__ import annotations

import hashlib
import os
from functools import lru_cache

import structlog
from anthropic import Anthropic
from openai import OpenAI

logger = structlog.get_logger(__name__)

DEFAULT_CHAT_MODEL = "claude-3-5-haiku-20241022"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536
_EMBEDDING_CACHE_SIZE = 512  # entradas LRU em memória


class AnthropicService:
    """
    Encapsula chamadas ao Claude (chat) e ao endpoint de embeddings da OpenAI.

    Melhorias implementadas:
    - Cache LRU de embeddings por hash SHA-256 do texto (Requisito 3.4).
    - Logging estruturado em cada operação (Requisito 3.2).
    - Tratamento explícito de erros com mensagens claras (Requisito 3.2).
    """

    def __init__(
        self,
        anthropic_api_key: str | None = None,
        openai_api_key: str | None = None,
    ) -> None:
        ant_key = anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
        oai_key = openai_api_key or os.getenv("OPENAI_API_KEY")

        if not ant_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY não definida. Configure no arquivo .env."
            )
        if not oai_key:
            raise RuntimeError(
                "OPENAI_API_KEY não definida. "
                "Necessária para geração de embeddings."
            )

        self._anthropic = Anthropic(api_key=ant_key)
        self._openai = OpenAI(api_key=oai_key)
        logger.info("anthropic_service.initialized", model=DEFAULT_CHAT_MODEL)

    # ------------------------------------------------------------------
    # Chat / LLM
    # ------------------------------------------------------------------

    def call_llm(
        self,
        prompt: str,
        model: str = DEFAULT_CHAT_MODEL,
        system: str | None = None,
        temperature: float = 0.7,
        history: list[dict] | None = None,
        max_tokens: int = 2048,
    ) -> str:
        """
        Gera uma resposta conversacional usando o Claude.

        Args:
            prompt: mensagem atual do usuário.
            model: modelo Anthropic a ser usado.
            system: prompt de sistema (instrução de comportamento).
            temperature: criatividade da resposta (0.0–1.0).
            history: histórico de mensagens no formato
                     [{"role": "user"|"assistant", "content": "..."}].
            max_tokens: limite de tokens na resposta.

        Returns:
            Texto da resposta gerada.
        """
        messages: list[dict] = []
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": prompt})

        log = logger.bind(model=model, history_len=len(history or []))
        log.info("llm.request")

        try:
            response = self._anthropic.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system or "",
                messages=messages,
                temperature=temperature,
            )
            text = response.content[0].text
            log.info("llm.response", output_tokens=response.usage.output_tokens)
            return text
        except Exception as exc:
            log.error("llm.error", error=str(exc))
            raise

    # ------------------------------------------------------------------
    # Embeddings (com cache LRU)
    # ------------------------------------------------------------------

    def get_embedding(
        self,
        text: str,
        model: str = DEFAULT_EMBEDDING_MODEL,
    ) -> list[float]:
        """
        Gera embedding do texto, utilizando cache LRU em memória.

        O cache é baseado no hash SHA-256 do texto + modelo, evitando
        chamadas repetidas à API para o mesmo conteúdo (Requisito 3.4).
        """
        return self._cached_embedding(text, model)

    def _cached_embedding(self, text: str, model: str) -> list[float]:
        cache_key = _embedding_cache_key(text, model)
        cached = _embedding_lru_cache(cache_key)
        if cached is not None:
            logger.debug("embedding.cache_hit", key=cache_key[:12])
            return cached

        logger.info("embedding.request", model=model, text_len=len(text))
        try:
            result = self._openai.embeddings.create(model=model, input=text)
            vector = result.data[0].embedding
            _store_embedding(cache_key, vector)
            logger.info("embedding.done", dim=len(vector))
            return vector
        except Exception as exc:
            logger.error("embedding.error", error=str(exc))
            raise


# ------------------------------------------------------------------
# Cache de embeddings (fora da classe para persistir entre instâncias)
# ------------------------------------------------------------------

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
