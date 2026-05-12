# Front-end — ProjetoAI Chatbot

Interface Streamlit que conversa com a API do back-end.

## Setup

```bash
cd front-end
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Rodar

Com o back-end já em execução em `http://localhost:8000`:

```bash
streamlit run src/app.py
```

Para apontar para outra URL do back-end:

```bash
BACKEND_URL=http://meu-host:8000 streamlit run src/app.py
```

## Estrutura

```
front-end/
├── src/
│   ├── app.py                   # entrypoint Streamlit
│   ├── components/chat.py       # UI do chat
│   └── services/api_client.py   # cliente HTTP do back-end
└── requirements.txt
```
