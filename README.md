# ProjetoAI — Chatbot

Chatbot dividido em dois projetos:

- [back-end/](back-end/) — API FastAPI com endpoint `/chat` (resposta mocada).
- [front-end/](front-end/) — Interface Streamlit para conversar com o back-end.

## Como rodar (dois terminais)

### 1. Back-end

```bash
cd back-end
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn src.main:app --reload --port 8000
```

### 2. Front-end

```bash
cd front-end
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run src/app.py
```

Abra a URL exibida pelo Streamlit (geralmente `http://localhost:8501`) e converse com o bot.

## Estrutura

```
ProjetoAI/
├── back-end/
│   ├── src/
│   │   ├── main.py
│   │   ├── models/chat.py
│   │   ├── routes/chat.py
│   │   └── services/chat_service.py
│   └── requirements.txt
└── front-end/
    ├── src/
    │   ├── app.py
    │   ├── components/chat.py
    │   └── services/api_client.py
    └── requirements.txt
```

## Próximos passos

Substituir o mock em [back-end/src/services/chat_service.py](back-end/src/services/chat_service.py) pela integração real com um modelo de IA.
