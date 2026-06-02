"""HTTP client for the FinSight FastAPI backend."""

from __future__ import annotations

from typing import Any, BinaryIO

import requests


class FinSightAPIError(Exception):
    """Raised when the API returns an error response."""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


class FinSightClient:
    """Thin wrapper around the FinSight REST API."""

    def __init__(self, base_url: str = "http://127.0.0.1:8000") -> None:
        self.base_url = base_url.rstrip("/")

    def _request(
        self,
        method: str,
        path: str,
        *,
        timeout: tuple[float, float] | float = (5, 120),
        **kwargs: Any,
    ) -> Any:
        url = f"{self.base_url}{path}"
        response = requests.request(method, url, timeout=timeout, **kwargs)
        if response.ok:
            if response.content:
                return response.json()
            return None
        detail = response.text
        try:
            detail = response.json().get("detail", detail)
        except Exception:  # noqa: BLE001
            pass
        raise FinSightAPIError(response.status_code, str(detail))

    def health(self) -> dict:
        return self._request("GET", "/health", timeout=(3, 15))

    def upload_pdf(self, file_name: str, file_obj: bytes | BinaryIO) -> dict:
        return self._request(
            "POST",
            "/upload",
            files={"file": (file_name, file_obj, "application/pdf")},
            timeout=(5, 300),
        )

    def ingest(self, payload: dict) -> dict:
        return self._request(
            "POST",
            "/ingest",
            json=payload,
            timeout=(5, 3600),
        )

    def query(self, payload: dict) -> dict:
        return self._request(
            "POST",
            "/query",
            json=payload,
            timeout=(5, 600),
        )

    def list_documents(self) -> list[dict]:
        return self._request("GET", "/documents", timeout=10)

    def delete_document(self, source_file: str) -> dict:
        return self._request(
            "DELETE",
            f"/documents/{source_file}",
            timeout=30,
        )
