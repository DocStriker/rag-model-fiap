"""
Orquestrador do fluxo de chat com suporte a RAG.

Melhorias implementadas vs. versão original:
- Logging estruturado em cada etapa do fluxo (Requisito 3.2).
- Tratamento de erros com fallback gracioso (Requisito 3.2).
- Recebe dependências por injeção (Requisito 3.1).
"""

from __future__ import annotations

import structlog
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.services.gemini_service import GeminiService
    from src.services.qdrant_service import QdrantService

logger = structlog.get_logger(__name__)

SYSTEM_PROMPT = """\
Você é o **FIAP AI**, assistente virtual da FIAP. Sua missão é apoiar \
alunos e profissionais com respostas claras, precisas e práticas.

Diretrizes:
- Responda sempre em português do Brasil.
- Seja direto e objetivo; evite enrolação e adjetivos vazios.
- Use formatação Markdown quando facilitar a leitura (listas, negrito, \
blocos de código).
- Se não souber a resposta, diga que não sabe — nunca invente informação.
- Se a pergunta for ambígua, peça esclarecimento antes de responder.
- Apresente-se como "FIAP AI" apenas quando perguntado quem é você.\
"""

RAG_SYSTEM_PROMPT = """\
Você é o **FIAP AI**, assistente especializado em responder perguntas \
com base em documentos indexados pelo usuário.

Como responder:
- Use **exclusivamente** as informações presentes nos trechos de contexto \
fornecidos abaixo.
- Se a resposta não estiver no contexto, responda exatamente: \
"Não encontrei essa informação nos documentos disponíveis."
- Não complemente com conhecimento externo nem especule.
- Cite as fontes ao final dos parágrafos relevantes no formato \
`(fonte, página X)` quando a metadata estiver disponível.
- Estruture respostas longas em tópicos ou listas para facilitar a leitura.
- Responda sempre em português do Brasil.\
"""

RAG_TOP_K = 4
HISTORY_LIMIT = 5


class ChatService:
    """
    Orquestra o fluxo de chat com ou sem RAG.

    Separação de responsabilidades (Requisito 3.1):
    - ChatService só decide qual fluxo usar e monta o prompt.
    - GeminiService cuida exclusivamente da chamada ao LLM.
    - QdrantService cuida exclusivamente da busca semântica.
"""

    def __init__(
        self,
        gemini_service: "GeminiService",
        qdrant_service: "QdrantService" | None = None,
    ) -> None:
        self.gemini_service = gemini_service
        self._qdrant_service = qdrant_service

    def generate_response(
        self,
        message: str,
        collection: str | None = None,
        history: list[dict] | None = None,
    ) -> str:
        if not message or not message.strip():
            return "Por favor, envie uma mensagem válida."

        trimmed_history = (history or [])[-HISTORY_LIMIT:]
        log = logger.bind(
            collection=collection,
            history_len=len(trimmed_history),
            msg_len=len(message),
        )

        if collection:
            log.info("chat.rag_flow")
            return self._answer_with_rag(message, collection, trimmed_history)

        log.info("chat.direct_flow")
        try:
            return self.gemini_service.call_llm(
                message, system=SYSTEM_PROMPT, history=trimmed_history
            )
        except Exception as exc:
            log.error("chat.direct_flow.error", error=str(exc))
            return "Ocorreu um erro ao processar sua mensagem. Tente novamente."

    def _answer_with_rag(
        self, message: str, collection: str, history: list[dict]
    ) -> str:
        log = logger.bind(collection=collection)
        qdrant = self._get_qdrant()

        try:
            results = qdrant.search(collection, message, limit=RAG_TOP_K)
        except Exception as exc:
            log.error("chat.rag.search_error", error=str(exc))
            return (
                "Ocorreu um erro ao buscar informações na coleção. "
                "Verifique se o banco vetorial está disponível."
            )

        if not results:
            log.info("chat.rag.no_results")
            return (
                f"Não encontrei trechos relevantes na coleção '{collection}' "
                "para responder."
            )

        log.info("chat.rag.results_found", count=len(results))

        blocks: list[str] = []
        for i, r in enumerate(results, start=1):
            payload = r.get("payload") or {}
            text = payload.get("text", "")
            source = payload.get("source", "?")
            page = payload.get("page")
            ref = source + (f", página {page}" if page else "")
            blocks.append(f"[Trecho {i} — {ref}]\n{text}")

        context = "\n\n".join(blocks)
        prompt = (
            f"Pergunta: {message}\n\n"
            f"Contexto recuperado dos documentos:\n{context}\n\n"
            "Responda usando apenas o contexto acima."
        )

        try:
            return self.gemini_service.call_llm(
                prompt, system=RAG_SYSTEM_PROMPT, history=history
            )
        except Exception as exc:
            log.error("chat.rag.llm_error", error=str(exc))
            return "Ocorreu um erro ao gerar a resposta. Tente novamente."

    def _get_qdrant(self) -> QdrantService:
        if self._qdrant_service is None:
            self._qdrant_service = QdrantService(
                gemini_service=self.gemini_service
            )
        return self._qdrant_service
