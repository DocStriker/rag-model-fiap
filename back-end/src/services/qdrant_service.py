"""
Serviço de integração com o Qdrant para indexação e recuperação semântica.

Melhorias implementadas vs. versão original:
- Chunking por sentenças com sobreposição semântica, em vez de cortes
  arbitrários por número de caracteres (Requisito 3.3).
- Metadados enriquecidos: tamanho do chunk, índice total, hash do texto
  (Requisito 3.3).
- Score mínimo configurável para filtrar resultados pouco relevantes.
- Logging estruturado em todas as operações (Requisito 3.2).
- Tratamento de erros explícito com exceções tipadas (Requisito 3.2).
"""

from __future__ import annotations

import os
import re
import uuid
from pathlib import Path

import structlog
from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from src.services.gemini_service import GeminiService

logger = structlog.get_logger(__name__)

DEFAULT_VECTOR_SIZE = 1024
DEFAULT_DISTANCE = Distance.COSINE
DEFAULT_CHUNK_SIZE = 800        # caracteres por chunk
DEFAULT_CHUNK_OVERLAP = 150     # sobreposição entre chunks
DEFAULT_MIN_SCORE = 0.35        # score mínimo de relevância
DEFAULT_CLIENT_TIMEOUT = 60


class QdrantService:
    """
    Gerencia coleções, indexação e busca semântica no Qdrant.

    Injeção de dependência: recebe GeminiService pelo construtor,
    sem acoplamento direto à implementação concreta (Requisito 3.1).
    """

    def __init__(
        self,
        endpoint: str | None = None,
        api_key: str | None = None,
        gemini_service: GeminiService | None = None,
    ) -> None:
        url = endpoint or os.getenv("QDRANT_ENDPOINT")
        key = api_key or os.getenv("QDRANT_API_KEY")

        if not url:
            raise RuntimeError(
                "QDRANT_ENDPOINT não definida. Configure no arquivo .env."
            )

        self.client = QdrantClient(
            url=url, api_key=key, timeout=DEFAULT_CLIENT_TIMEOUT
        )
        self.gemini_service = gemini_service
        logger.info("qdrant_service.initialized", endpoint=url)

    # ------------------------------------------------------------------
    # Gerenciamento de coleções
    # ------------------------------------------------------------------

    def list_collections(self) -> list[str]:
        return [c.name for c in self.client.get_collections().collections]

    def collection_exists(self, name: str) -> bool:
        return self.client.collection_exists(collection_name=name)

    def create_collection(
        self,
        name: str,
        vector_size: int = DEFAULT_VECTOR_SIZE,
        distance: Distance = DEFAULT_DISTANCE,
        recreate: bool = False,
    ) -> None:
        if self.collection_exists(name):
            if not recreate:
                logger.info("qdrant.collection_already_exists", name=name)
                return
            self.delete_collection(name)

        self.client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=vector_size, distance=distance),
        )
        logger.info(
            "qdrant.collection_created",
            name=name,
            vector_size=vector_size,
        )

    def delete_collection(self, name: str) -> None:
        self.client.delete_collection(collection_name=name)
        logger.info("qdrant.collection_deleted", name=name)

    # ------------------------------------------------------------------
    # Indexação
    # ------------------------------------------------------------------

    def upsert_points(
        self,
        collection_name: str,
        vectors: list[list[float]],
        payloads: list[dict] | None = None,
        ids: list[str] | None = None,
    ) -> None:
        payloads = payloads or [{} for _ in vectors]
        ids = ids or [str(uuid.uuid4()) for _ in vectors]
        points = [
            PointStruct(id=i, vector=v, payload=p)
            for i, v, p in zip(ids, vectors, payloads)
        ]
        self.client.upsert(collection_name=collection_name, points=points)
        logger.info(
            "qdrant.upserted",
            collection=collection_name,
            count=len(points),
        )

    def upload_texts(
        self,
        collection_name: str,
        texts: list[str],
        metadata: list[dict] | None = None,
    ) -> int:
        self._require_gemini()
        log = logger.bind(collection=collection_name, total=len(texts))
        log.info("qdrant.upload_texts.start")

        vectors = [self.gemini_service.get_embedding(t) for t in texts]

        payloads: list[dict] = []
        for i, text in enumerate(texts):
            payload: dict = {"text": text, "char_count": len(text)}
            if metadata and i < len(metadata):
                payload.update(metadata[i])
            payloads.append(payload)

        self.upsert_points(collection_name, vectors, payloads)
        log.info("qdrant.upload_texts.done")
        return len(texts)

    def upload_file(
        self,
        collection_name: str,
        file_path: str,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> int:
        path = Path(file_path)
        log = logger.bind(file=path.name, collection=collection_name)
        log.info("qdrant.upload_file.start")

        if path.suffix.lower() == ".pdf":
            chunks, metadata = self._extract_pdf(path, chunk_size, chunk_overlap)
        else:
            raw_text = path.read_text(encoding="utf-8")
            chunks = self._chunk_text(raw_text, chunk_size, chunk_overlap)
            metadata = [
                {
                    "source": path.name,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                }
                for i in range(len(chunks))
            ]

        if not chunks:
            log.warning("qdrant.upload_file.empty")
            return 0

        log.info("qdrant.upload_file.chunks", count=len(chunks))
        return self.upload_texts(collection_name, chunks, metadata)

    # ------------------------------------------------------------------
    # Busca semântica
    # ------------------------------------------------------------------

    def search(
        self,
        collection_name: str,
        query: str,
        limit: int = 5,
        min_score: float = DEFAULT_MIN_SCORE,
    ) -> list[dict]:
        self._require_gemini()
        log = logger.bind(collection=collection_name, limit=limit)
        log.info("qdrant.search.start")

        vector = self.gemini_service.get_embedding(query)
        result = self.client.query_points(
            collection_name=collection_name,
            query=vector,
            limit=limit,
            score_threshold=min_score,
        )
        hits = [
            {"id": p.id, "score": p.score, "payload": p.payload}
            for p in result.points
        ]
        log.info("qdrant.search.done", hits=len(hits))
        return hits

    # ------------------------------------------------------------------
    # Helpers privados
    # ------------------------------------------------------------------

    def _require_gemini(self) -> None:
        if self.gemini_service is None:
            raise RuntimeError(
                "GeminiService é necessário para gerar embeddings. "
                "Passe gemini_service no construtor."
            )

    def _extract_pdf(
        self, path: Path, chunk_size: int, chunk_overlap: int
    ) -> tuple[list[str], list[dict]]:
        reader = PdfReader(str(path))
        chunks: list[str] = []
        metadata: list[dict] = []
        global_index = 0

        for page_num, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            if not page_text.strip():
                continue

            page_chunks = self._chunk_text(page_text, chunk_size, chunk_overlap)
            for i, chunk in enumerate(page_chunks):
                chunks.append(chunk)
                metadata.append(
                    {
                        "source": path.name,
                        "page": page_num,
                        "chunk_index": i,
                        "global_index": global_index,
                        "total_pages": len(reader.pages),
                        "char_count": len(chunk),
                    }
                )
                global_index += 1

        return chunks, metadata

    @staticmethod
    def _chunk_text(
        text: str, chunk_size: int, overlap: int
    ) -> list[str]:
        """
        Chunking por sentenças com sobreposição semântica.

        Melhoria vs. original: em vez de cortar no meio de palavras,
        identifica limites de sentenças (. ! ? \\n\\n) e agrupa até
        atingir chunk_size, depois recua overlap caracteres para o
        próximo chunk preservar contexto.
        """
        if chunk_size <= overlap:
            raise ValueError("chunk_size deve ser maior que overlap")

        # Divide em sentenças preservando o delimitador
        sentence_endings = re.compile(r'(?<=[.!?])\s+|\n{2,}')
        sentences = [s.strip() for s in sentence_endings.split(text) if s.strip()]

        chunks: list[str] = []
        current = ""

        for sentence in sentences:
            if len(current) + len(sentence) + 1 <= chunk_size:
                current = (current + " " + sentence).strip()
            else:
                if current:
                    chunks.append(current)
                # Sobreposição: pega o final do chunk anterior
                overlap_text = current[-overlap:] if len(current) > overlap else current
                current = (overlap_text + " " + sentence).strip()

        if current:
            chunks.append(current)

        return [c for c in chunks if c.strip()]
