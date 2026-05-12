# Decisões de Design — FIAP AI Back-end

## 1. Banco de dados vetorial: Qdrant

O Qdrant foi mantido como banco vetorial por oferecer suporte nativo a busca por similaridade cosseno, API simples via Python SDK, cluster gerenciado gratuito disponível em cloud.qdrant.io e imagem Docker oficial para desenvolvimento local. A instância é provisionada via Docker Compose para uso local, ou via cluster cloud configurado por variável de ambiente `QDRANT_ENDPOINT`, sem alterar nenhuma linha de código.

## 2. Modelo de linguagem: Claude 3.5 Haiku (Anthropic)

O modelo `claude-3-5-haiku-20241022` foi escolhido por três razões principais: excelente desempenho em português brasileiro, latência baixa para uso conversacional e custo-benefício adequado para aplicações educacionais. Os embeddings continuam sendo gerados via `text-embedding-3-small` da OpenAI (dimensão 1536), pois a Anthropic não oferece endpoint de embeddings próprio. Essa separação é explícita no código — `anthropic_service.py` expõe dois métodos distintos (`call_llm` e `get_embedding`), facilitando a troca de qualquer um dos provedores de forma independente.

## 3. Melhorias de design implementadas

**3.1 — Separação de responsabilidades e injeção de dependências**
O módulo `src/dependencies.py` centraliza a criação de todos os serviços via `lru_cache`, garantindo instância única por processo. As rotas recebem dependências via `Depends()` do FastAPI, eliminando singletons globais nas rotas e permitindo substituição completa por mocks nos testes com `app.dependency_overrides`.

**3.2 — Tratamento de erros e logging estruturado**
`structlog` foi adotado em substituição ao `print`/`logging` padrão. Todos os serviços e rotas emitem eventos com campos estruturados (collection, file, error, count, etc.), facilitando filtragem em ferramentas como Datadog ou Loki. Em produção (`LOG_FORMAT=json`) os logs são emitidos em JSON; em desenvolvimento, em texto colorido. As rotas retornam `HTTPException` com status codes semânticos (409, 404, 415, 413, 500) em vez de 500 genérico.

**3.3 — Chunking por sentenças e metadados enriquecidos**
O método `_chunk_text` foi reescrito para identificar limites de sentença (`. ! ? \n\n`) antes de cortar, evitando truncar palavras no meio. A sobreposição preserva contexto entre chunks adjacentes. Os metadados foram enriquecidos com `global_index`, `total_pages`, `char_count` e `total_chunks`, melhorando a rastreabilidade das fontes nas respostas RAG.

**3.4 — Cache de embeddings em memória**
`anthropic_service.py` implementa um cache LRU de até 512 entradas, indexado pelo hash SHA-256 do texto concatenado com o nome do modelo. Isso elimina chamadas repetidas à API OpenAI para textos idênticos (queries frequentes, re-uploads do mesmo documento), reduzindo latência e custo.

**3.5 — Segurança e validação de entrada**
O modelo `CollectionCreateRequest` valida o nome da coleção com regex `^[a-zA-Z0-9_\-]+$`, prevenindo injeção de caracteres especiais. O endpoint de upload rejeita extensões não suportadas (415) e arquivos acima de 10 MB (413). O campo `message` em `ChatRequest` tem `max_length=4000`, prevenindo payloads abusivos.

**3.6 — Cobertura de testes automatizados**
`tests/test_api.py` cobre rotas HTTP (chat, collections, upload, health), lógica do `ChatService` (mensagem vazia, RAG sem resultados) e utilitários (`_chunk_text`). Os testes usam mocks completos para Anthropic e Qdrant, rodando sem dependências externas via `pytest tests/ -v`.

**3.7 — Empacotamento com Docker Compose**
`docker-compose.yml` orquestra Qdrant + API com um único `docker compose up --build`. O Qdrant persiste dados em volume nomeado e expõe healthcheck; a API aguarda o Qdrant estar saudável antes de iniciar (`depends_on: condition: service_healthy`). O `Dockerfile` usa imagem slim do Python 3.11 para minimizar o tamanho da imagem.
