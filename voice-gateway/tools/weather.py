"""Simple weather lookup tool."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
from urllib.parse import quote_plus

import requests


class WeatherError(RuntimeError):
    """Raised when the weather service cannot be queried."""


@dataclass
class WeatherClient:
    base_url: str
    default_location: Optional[str]
    timeout: float = 6.0

    def fetch(self, location: Optional[str] = None) -> Dict[str, Any]:
        loc = (location or self.default_location or "").strip()
        if not loc:
            raise WeatherError("No location provided for weather lookup.")

        url = f"{self.base_url.rstrip('/')}/{quote_plus(loc)}"
        response = requests.get(url, params={"format": "j1"}, timeout=self.timeout)
        if response.status_code >= 400:
            raise WeatherError(f"Weather lookup failed ({response.status_code}).")

        data = response.json()
        current = (data.get("current_condition") or [{}])[0]
        desc_list = current.get("weatherDesc") or [{}]
        description = desc_list[0].get("value", "").strip()

        def _safe_float(value: Any) -> Optional[float]:
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        summary = description or "Unavailable"
        temp_c = _safe_float(current.get("temp_C"))
        temp_f = _safe_float(current.get("temp_F"))
        feels_c = _safe_float(current.get("FeelsLikeC"))
        feels_f = _safe_float(current.get("FeelsLikeF"))
        humidity = _safe_float(current.get("humidity"))
        wind_kph = _safe_float(current.get("windspeedKmph"))

        if temp_c is not None and feels_c is not None:
            summary = f"{description} with {temp_c:.0f}°C, feels like {feels_c:.0f}°C"
        elif temp_f is not None and feels_f is not None:
            summary = f"{description} with {temp_f:.0f}°F, feels like {feels_f:.0f}°F"

        return {
            "location": loc,
            "summary": summary,
            "description": description,
            "temperature_c": temp_c,
            "temperature_f": temp_f,
            "feels_like_c": feels_c,
            "feels_like_f": feels_f,
            "humidity_percent": humidity,
            "wind_kph": wind_kph,
        }
