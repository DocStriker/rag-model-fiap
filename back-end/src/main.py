"""
Entry point da API FastAPI com logging estruturado configurado.

Melhoria (Requisito 3.2): structlog configurado para emitir JSON em
produção (LOG_FORMAT=json) ou texto colorido em desenvolvimento
(LOG_FORMAT=console, padrão).
"""

import logging
import os

import structlog
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

# ------------------------------------------------------------------
# Logging estruturado
# ------------------------------------------------------------------
log_format = os.getenv("LOG_FORMAT", "console")

structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        (
            structlog.processors.JSONRenderer()
            if log_format == "json"
            else structlog.dev.ConsoleRenderer()
        ),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
)

# ------------------------------------------------------------------
# Aplicação
# ------------------------------------------------------------------
app = FastAPI(
    title="FIAP AI Chatbot API",
    version="2.0.0",
    description="API de chat com RAG usando Gemini + Qdrant.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import routers after app creation to avoid circular imports
from src.routes.chat import router as chat_router  # noqa: E402
from src.routes.collections import router as collections_router  # noqa: E402

app.include_router(chat_router)
app.include_router(collections_router)

app.include_router(chat_router)
app.include_router(collections_router)


@app.get("/health", tags=["infra"])
def health() -> dict:
    """Verificação de saúde da API."""
    return {"status": "ok", "version": app.version}
