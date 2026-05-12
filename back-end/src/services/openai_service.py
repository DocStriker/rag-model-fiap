import os

from openai import OpenAI

DEFAULT_CHAT_MODEL = "gpt-4o-mini"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


class OpenAIService:
    def __init__(self, api_key: str | None = None):
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError(
                "OPENAI_API_KEY não definida. Configure no .env do back-end."
            )
        self.client = OpenAI(api_key=key)

    def call_llm(
        self,
        prompt: str,
        model: str = DEFAULT_CHAT_MODEL,
        system: str | None = None,
        temperature: float = 0.7,
        history: list[dict] | None = None,
    ) -> str:
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": prompt})

        completion = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
        )
        return completion.choices[0].message.content or ""

    def get_embedding(
        self,
        text: str,
        model: str = DEFAULT_EMBEDDING_MODEL,
    ) -> list[float]:
        result = self.client.embeddings.create(model=model, input=text)
        return result.data[0].embedding
