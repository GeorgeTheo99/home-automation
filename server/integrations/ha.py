from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, Tuple

import httpx


class HAClient:
    def __init__(self, base_url: str, token: str, timeout: float = 5.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
            },
            timeout=timeout,
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def _get_states(self) -> List[dict]:
        r = await self._client.get("/api/states")
        r.raise_for_status()
        return r.json()

    async def list_lights(self) -> List[dict]:
        entities = await self._get_states()
        lights = [e for e in entities if isinstance(e, dict) and (e.get("entity_id", "").startswith("light."))]
        out: List[dict] = []
        for e in lights:
            ent_id = e.get("entity_id")
            attrs = e.get("attributes", {}) or {}
            caps: List[str] = ["on"]
            modes = set((attrs.get("supported_color_modes") or []))
            if "brightness" in modes or attrs.get("brightness") is not None or attrs.get("supported_features", 0) != 0:
                caps.append("brightness")
            if modes & {"hs", "rgb", "xy", "color_temp"}:
                caps.append("color")
            state = e.get("state")
            brightness_pct = None
            if isinstance(attrs.get("brightness"), int):
                try:
                    brightness_pct = int(round((attrs["brightness"] / 255.0) * 100))
                except Exception:
                    brightness_pct = None
            out.append(
                {
                    "id": ent_id,
                    "name": attrs.get("friendly_name", ent_id),
                    "type": "light",
                    "adapter": "ha",
                    "capabilities": caps,
                    "state": {"on": state == "on", "brightness": brightness_pct},
                }
            )
        return out

    async def list_outlets(self) -> List[dict]:
        entities = await self._get_states()
        switches = [e for e in entities if isinstance(e, dict) and (e.get("entity_id", "").startswith("switch."))]
        out: List[dict] = []
        for e in switches:
            ent_id = e.get("entity_id")
            attrs = e.get("attributes", {}) or {}
            state = e.get("state")
            out.append(
                {
                    "id": ent_id,
                    "name": attrs.get("friendly_name", ent_id),
                    "type": "outlet",
                    "adapter": "ha",
                    "capabilities": ["on"],
                    "state": {"on": state == "on"},
                }
            )
        return out

    async def set_light_state(self, entity_id: str, on: Optional[bool] = None, brightness: Optional[int] = None, color: Optional[str] = None) -> dict:
        domain = "light"
        data: Dict[str, Any] = {"entity_id": entity_id}
        if brightness is not None:
            try:
                bri = int(max(0, min(100, brightness)))
                data["brightness"] = int(round((bri / 100.0) * 255))
            except Exception:
                pass
        if color:
            from ..util.colors import parse_color_rgb

            rgb = parse_color_rgb(color)
            if rgb:
                data["rgb_color"] = list(rgb)
        if on is None and ("brightness" in data or "rgb_color" in data):
            service = "turn_on"
        else:
            service = "turn_on" if on else "turn_off"
        r = await self._client.post(f"/api/services/{domain}/{service}", json=data)
        r.raise_for_status()
        # Return a minimal acknowledgement; HA returns the updated state in a list
        return {"on": on, "brightness": brightness, "color": color}

    async def set_outlet_state(self, entity_id: str, on: bool) -> dict:
        domain = "switch"
        service = "turn_on" if on else "turn_off"
        data = {"entity_id": entity_id}
        r = await self._client.post(f"/api/services/{domain}/{service}", json=data)
        r.raise_for_status()
        return {"on": on}

