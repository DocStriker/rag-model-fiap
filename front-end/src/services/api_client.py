import os

import requests

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
DEFAULT_TIMEOUT = 30
COLLECTIONS_TIMEOUT = 90
UPLOAD_TIMEOUT = 300


class ApiClient:
    def __init__(self, base_url: str = BACKEND_URL):
        self.base_url = base_url.rstrip("/")

    def send_message(
        self,
        message: str,
        collection: str | None = None,
        history: list[dict] | None = None,
    ) -> str:
        url = f"{self.base_url}/chat"
        payload: dict = {"message": message}
        if collection:
            payload["collection"] = collection
        if history:
            payload["history"] = history
        try:
            res = requests.post(
                url,
                json=payload,
                timeout=DEFAULT_TIMEOUT,
            )
            res.raise_for_status()
            return res.json().get("response", "")
        except requests.HTTPError as exc:
            response = exc.response
            detail = None
            if response is not None:
                try:
                    detail = response.json().get("detail")
                except ValueError:
                    detail = None
            if detail:
                return f"[Erro do backend: {detail}]"
            return f"[Erro ao contatar o back-end: {exc}]"
        except requests.RequestException as e:
            return f"[Erro ao contatar o back-end: {e}]"

    def list_collections(self) -> list[str]:
        res = requests.get(
            f"{self.base_url}/collections", timeout=COLLECTIONS_TIMEOUT
        )
        res.raise_for_status()
        return res.json().get("collections", [])

    def create_collection(self, name: str) -> list[str]:
        res = requests.post(
            f"{self.base_url}/collections",
            json={"name": name},
            timeout=COLLECTIONS_TIMEOUT,
        )
        if res.status_code == 409:
            raise ValueError("Coleção já existe.")
        res.raise_for_status()
        return res.json().get("collections", [])

    def upload_document(
        self, collection: str, file_name: str, file_bytes: bytes
    ) -> dict:
        files = {"file": (file_name, file_bytes)}
        try:
            res = requests.post(
                f"{self.base_url}/collections/{collection}/documents",
                files=files,
                timeout=UPLOAD_TIMEOUT,
            )
            res.raise_for_status()
            return res.json()
        except requests.HTTPError as exc:
            response = exc.response
            detail = None
            if response is not None:
                try:
                    detail = response.json().get("detail")
                except ValueError:
                    detail = None
            if detail:
                raise RuntimeError(detail) from exc
            raise RuntimeError(f"Erro ao enviar documento: {exc}") from exc
        except requests.RequestException as exc:
            raise RuntimeError(f"Erro ao contatar o back-end: {exc}") from exc
