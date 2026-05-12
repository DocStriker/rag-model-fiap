"""
Rotas de gerenciamento de coleções.

Melhorias (Requisito 3.1): QdrantService injetado via Depends().
Melhorias (Requisito 3.2): tratamento de erros com status codes
semânticos e logging estruturado.
Melhorias (Requisito 3.5): validação de nome da coleção no modelo Pydantic
(pattern alfanumérico + _ -) e limite de tamanho de arquivo (10 MB).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import structlog
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from fastapi.responses import Response
from src.dependencies import get_qdrant_service
from src.models.collection import (
    CollectionCreateRequest,
    CollectionListResponse,
    UploadResponse,
)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.services.qdrant_service import QdrantService

router = APIRouter(prefix="/collections", tags=["collections"])
logger = structlog.get_logger(__name__)

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB
ALLOWED_EXTENSIONS = {".txt", ".md", ".pdf"}


@router.get("", response_model=CollectionListResponse)
def list_collections(
    qdrant: "QdrantService" = Depends(get_qdrant_service),
) -> CollectionListResponse:
    """Lista todas as coleções disponíveis."""
    return CollectionListResponse(collections=qdrant.list_collections())


@router.post("", response_model=CollectionListResponse, status_code=status.HTTP_201_CREATED)
def create_collection(
    req: CollectionCreateRequest,
    qdrant: "QdrantService" = Depends(get_qdrant_service),
) -> CollectionListResponse:
    """Cria uma nova coleção no banco vetorial."""
    if qdrant.collection_exists(req.name):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Coleção '{req.name}' já existe.",
        )
    qdrant.create_collection(req.name, vector_size=req.vector_size)
    logger.info("route.collections.created", name=req.name)
    return CollectionListResponse(collections=qdrant.list_collections())


@router.delete("/{name}", status_code=status.HTTP_204_NO_CONTENT, response_class=Response)
def delete_collection(
    name: str,
    qdrant: "QdrantService" = Depends(get_qdrant_service),
) -> Response:
    """Remove uma coleção e todos os seus documentos."""
    if not qdrant.collection_exists(name):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Coleção '{name}' não encontrada.",
        )
    qdrant.delete_collection(name)
    logger.info("route.collections.deleted", name=name)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/{name}/documents", response_model=UploadResponse)
async def upload_document(
    name: str,
    file: UploadFile = File(...),
    qdrant: "QdrantService" = Depends(get_qdrant_service),
) -> UploadResponse:
    """
    Faz upload e indexação de um documento (.txt, .md, .pdf) na coleção.
    Limite: 10 MB por arquivo.
    """
    if not qdrant.collection_exists(name):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Coleção '{name}' não encontrada.",
        )

    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Extensão '{suffix}' não suportada. Use: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    contents = await file.read()
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Arquivo excede o limite de 10 MB.",
        )

    log = logger.bind(collection=name, file=file.filename, size=len(contents))
    log.info("route.documents.upload_start")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        chunks = qdrant.upload_file(name, tmp_path)
    except UnicodeDecodeError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Arquivo não é texto UTF-8 válido.",
        )
    except RuntimeError as exc:
        log.error("route.documents.upload_error", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )
    except Exception as exc:
        log.error("route.documents.upload_error", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    log.info("route.documents.upload_done", chunks=chunks)
    return UploadResponse(collection=name, chunks=chunks)
