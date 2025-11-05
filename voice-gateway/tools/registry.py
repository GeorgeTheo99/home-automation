"""Tool schema registration and dispatch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .home_assistant import HomeAssistantClient, HomeAssistantError
from .weather import WeatherClient, WeatherError


TOOL_HOME_ASSISTANT = "home_assistant_call_service"
TOOL_WEATHER = "get_weather_summary"


class ToolDispatchError(RuntimeError):
    """Raised when a tool call cannot be handled."""


@dataclass
class ToolRegistry:
    ha_client: HomeAssistantClient
    weather_client: WeatherClient

    def schemas(self) -> List[Dict[str, Any]]:
        """Return tool definitions for the realtime session."""
        tools: List[Dict[str, Any]] = [
            {
                "type": "function",
                "name": TOOL_HOME_ASSISTANT,
                "description": "Call a Home Assistant service to control smart devices.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "domain": {
                            "type": "string",
                            "description": "Home Assistant domain, e.g. 'light' or 'switch'.",
                        },
                        "service": {
                            "type": "string",
                            "description": "Service name within the domain, e.g. 'turn_on'.",
                        },
                        "entity_id": {
                            "type": "string",
                            "description": "Target entity_id to act on.",
                        },
                        "data": {
                            "type": "object",
                            "description": "Additional service payload.",
                        },
                    },
                    "required": ["domain", "service"],
                },
            },
            {
                "type": "function",
                "name": TOOL_WEATHER,
                "description": "Retrieve a short weather summary for a location.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City or address to fetch weather for. Optional if a default is configured.",
                        }
                    },
                },
            },
        ]
        return tools

    def dispatch(self, name: str, arguments: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Handle a tool call and return a JSON-serialisable result."""
        arguments = arguments or {}
        if name == TOOL_HOME_ASSISTANT:
            return self._call_home_assistant(arguments)
        if name == TOOL_WEATHER:
            return self._fetch_weather(arguments)
        raise ToolDispatchError(f"Unknown tool: {name}")

    def _call_home_assistant(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        if not self.ha_client.is_configured:
            raise ToolDispatchError("Home Assistant credentials are not configured.")

        domain = arguments.get("domain")
        service = arguments.get("service")
        if not isinstance(domain, str) or not domain:
            raise ToolDispatchError("Home Assistant call requires a 'domain' string.")
        if not isinstance(service, str) or not service:
            raise ToolDispatchError("Home Assistant call requires a 'service' string.")

        entity_id = arguments.get("entity_id")
        data = arguments.get("data") if isinstance(arguments.get("data"), dict) else None

        try:
            result = self.ha_client.call_service(domain, service, entity_id=entity_id, data=data)
        except HomeAssistantError as exc:
            raise ToolDispatchError(str(exc)) from exc

        return {
            "status": "ok",
            "domain": domain,
            "service": service,
            "entity_id": entity_id,
            "response": result,
        }

    def _fetch_weather(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        location = arguments.get("location")
        if location is not None and not isinstance(location, str):
            raise ToolDispatchError("Weather lookup 'location' must be a string if provided.")
        try:
            result = self.weather_client.fetch(location=location)
        except WeatherError as exc:
            raise ToolDispatchError(str(exc)) from exc
        return result
