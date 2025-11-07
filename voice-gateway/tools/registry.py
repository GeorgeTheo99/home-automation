"""Tool schema registration and dispatch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from .home_assistant import HomeAssistantClient, HomeAssistantError
from .weather import WeatherClient, WeatherError


TOOL_HOME_ASSISTANT = "home_assistant_call_service"
TOOL_WEATHER = "get_weather_summary"
TOOL_SIGNAL_CONFUSION = "signal_confusion"


class ToolDispatchError(RuntimeError):
    """Raised when a tool call cannot be handled."""


@dataclass
class ToolRegistry:
    ha_client: HomeAssistantClient
    weather_client: WeatherClient
    audio_playback: Optional[Callable[[np.ndarray], None]] = None

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
            {
                "type": "function",
                "name": TOOL_SIGNAL_CONFUSION,
                "description": "Play a negative chime when you don't understand the user's input or cannot help. Use this instead of speaking when confused.",
                "parameters": {
                    "type": "object",
                    "properties": {},
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
        if name == TOOL_SIGNAL_CONFUSION:
            return self._signal_confusion()
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

    def _signal_confusion(self) -> Dict[str, Any]:
        """Generate and play a negative chime to indicate confusion/inability to help."""
        if self.audio_playback is None:
            return {"status": "skipped", "reason": "No audio playback available"}

        # Generate a descending two-tone "negative" chime
        sample_rate = 24000  # Match typical output rate
        duration = 0.3  # 300ms total

        # First tone: 400 Hz for 150ms
        t1 = np.linspace(0, 0.15, int(sample_rate * 0.15), dtype=np.float32)
        tone1 = np.sin(2 * np.pi * 400 * t1)

        # Second tone: 300 Hz for 150ms
        t2 = np.linspace(0, 0.15, int(sample_rate * 0.15), dtype=np.float32)
        tone2 = np.sin(2 * np.pi * 300 * t2)

        # Combine tones
        chime = np.concatenate([tone1, tone2])

        # Apply envelope to avoid clicks (fade in/out)
        envelope = np.ones_like(chime)
        fade_samples = int(sample_rate * 0.01)  # 10ms fade
        envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
        envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)
        chime = chime * envelope

        # Convert to PCM16 with reduced volume (-12dB)
        volume = 0.25
        pcm16 = np.clip(chime * volume * 32767, -32768, 32767).astype(np.int16)

        # Play the chime
        self.audio_playback(pcm16)

        return {"status": "played", "message": "Confusion chime played"}
