# FIAP AI — Back-end (v2.0)

API FastAPI com fluxo RAG completo: **Claude (Anthropic)** como LLM, **embeddings OpenAI** e **Qdrant** como banco vetorial.

---

## Pré-requisitos

- Python 3.11+
- Docker e Docker Compose (opcional, recomendado)
- Chaves de API: Anthropic e OpenAI

---

## Configuração

```bash
cd back-end
cp .env.example .env
# Edite o .env e preencha ANTHROPIC_API_KEY, OPENAI_API_KEY e QDRANT_ENDPOINT
```

### Variáveis de ambiente obrigatórias

| Variável           | Descrição                                        |
|--------------------|--------------------------------------------------|
| `ANTHROPIC_API_KEY`| Chave da API Anthropic (claude-3-5-haiku)        |
| `OPENAI_API_KEY`   | Chave da API OpenAI (embeddings text-embedding-3-small) |
| `QDRANT_ENDPOINT`  | URL do cluster Qdrant (ex: `http://localhost:6333`) |

### Variáveis opcionais

| Variável       | Padrão    | Descrição                              |
|----------------|-----------|----------------------------------------|
| `QDRANT_API_KEY` | vazio   | Chave de API do Qdrant (clusters cloud)|
| `LOG_FORMAT`   | `console` | `console` (dev) ou `json` (produção)   |
| `CORS_ORIGINS` | `*`       | Origens CORS permitidas (separadas por vírgula) |

---

## Execução com Docker Compose (recomendado)

Sobe o Qdrant local + a API em um único comando:

```bash
docker compose up --build
```

A API estará disponível em `http://localhost:8000`.
O Qdrant estará disponível em `http://localhost:6333`.

Para parar:

```bash
docker compose down
```

---

## Execução local (sem Docker)

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn src.main:app --reload --port 8000
```

> Configure `QDRANT_ENDPOINT` apontando para uma instância Qdrant disponível
> (local em Docker ou cluster cloud em https://cloud.qdrant.io).

---

## Endpoints

### Infra

```
GET /health
→ {"status": "ok", "version": "2.0.0"}
```

### Chat

```
POST /chat
Content-Type: application/json

{
  "message": "O que é RAG?",
  "collection": "minha-colecao",   // opcional — ativa modo RAG
  "history": [                      // opcional
    {"role": "user",      "content": "Olá"},
    {"role": "assistant", "content": "Olá! Como posso ajudar?"}
  ]
}

→ {"response": "RAG (Retrieval Augmented Generation) é..."}
```

### Coleções

```
GET /collections
→ {"collections": ["docs", "aulas"]}

POST /collections
{"name": "minha-colecao", "vector_size": 1536}
→ 201 {"collections": ["minha-colecao"]}

DELETE /collections/{name}
→ 204

POST /collections/{name}/documents
form-data: file=<arquivo .txt | .md | .pdf> (máx 10 MB)
→ {"collection": "minha-colecao", "chunks": 42}
```

---

## Testes

```bash
pytest tests/ -v
```

Os testes usam mocks para Anthropic e Qdrant — não precisam de chaves reais.

---

## Estrutura do projeto

```
back-end/
├── src/
│   ├── main.py                        # Entry point FastAPI + logging
│   ├── dependencies.py                # Injeção de dependências (DI)
│   ├── models/
│   │   ├── chat.py                    # Schemas de chat
│   │   └── collection.py             # Schemas de coleção
│   ├── routes/
│   │   ├── chat.py                    # POST /chat
│   │   └── collections.py            # CRUD /collections
│   └── services/
│       ├── anthropic_service.py      # LLM (Claude) + Embeddings
│       ├── qdrant_service.py         # Banco vetorial + chunking
│       └── chat_service.py           # Orquestrador RAG
├── tests/
│   └── test_api.py                   # Testes automatizados
├── .env.example                      # Template de variáveis
├── docker-compose.yml                # Qdrant + API
├── Dockerfile                        # Imagem da API
├── requirements.txt
└── DECISOES.md                       # Decisões de design
```

---

## Documentação interativa

Com a API rodando, acesse:

- Swagger UI: http://localhost:8000/docs
- Redoc: http://localhost:8000/redoc
