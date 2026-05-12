"""
Rota de chat.

Melhoria (Requisito 3.1): ChatService injetado via Depends(),
removendo o singleton global da rota original.
Melhoria (Requisito 3.2): HTTPException com mensagens claras em caso
de falha inesperada.
"""

from __future__ import annotations

import structlog
from fastapi import APIRouter, Depends, HTTPException, status
from src.models.chat import ChatRequest, ChatResponse
from src.dependencies import get_chat_service
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.services.chat_service import ChatService

router = APIRouter(prefix="/chat", tags=["chat"])
logger = structlog.get_logger(__name__)


@router.post("", response_model=ChatResponse, status_code=status.HTTP_200_OK)
def chat(
    request: ChatRequest,
    chat_service: "ChatService" = Depends(get_chat_service),
) -> ChatResponse:
    """Envia uma mensagem e recebe uma resposta do assistente."""
    from src.models.chat import ChatResponse
    logger.info(
        "route.chat",
        collection=request.collection,
        history_len=len(request.history),
    )
    history = [m.model_dump() for m in request.history]
    try:
        response_text = chat_service.generate_response(
            request.message,
            collection=request.collection,
            history=history,
        )
    except Exception as exc:
        logger.error("route.chat.unhandled_error", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erro interno ao processar a mensagem.",
        )
    return ChatResponse(response=response_text)
