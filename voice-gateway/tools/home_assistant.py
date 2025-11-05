"""Home Assistant REST tool."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests


class HomeAssistantError(RuntimeError):
    """Raised when Home Assistant requests fail."""


@dataclass
class HomeAssistantClient:
    base_url: Optional[str]
    token: Optional[str]
    timeout: float = 6.0

    def __post_init__(self) -> None:
        self._session = requests.Session()
        if self.is_configured:
            assert self.base_url is not None
            assert self.token is not None
            self._session.headers.update(
                {
                    "Authorization": f"Bearer {self.token}",
                    "Content-Type": "application/json",
                }
            )

    @property
    def is_configured(self) -> bool:
        return bool(self.base_url and self.token)

    def call_service(
        self,
        domain: str,
        service: str,
        entity_id: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not self.is_configured:
            raise HomeAssistantError("Home Assistant integration is not configured.")

        assert self.base_url is not None
        payload: Dict[str, Any] = dict(data or {})
        if entity_id:
            payload.setdefault("entity_id", entity_id)

        url = f"{self.base_url.rstrip('/')}/api/services/{domain}/{service}"
        response = self._session.post(url, json=payload, timeout=self.timeout)
        if response.status_code >= 400:
            raise HomeAssistantError(
                f"Home Assistant call failed ({response.status_code}): {response.text}"
            )
        if not response.content:
            return {}
        return response.json()
