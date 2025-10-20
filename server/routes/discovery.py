from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, Request

from ..discovery import scan_all


router = APIRouter()


@router.get("/devices")
async def list_devices(request: Request) -> list[dict]:
    # If HA backend is enabled, combine HA lights + outlets
    if getattr(request.app.state, "backend", "native") == "ha":
        ha = request.app.state.ha
        lights = await ha.list_lights()
        outlets = await ha.list_outlets()
        return lights + outlets
    return request.app.state.registry.list()


@router.post("/discovery/scan")
async def discovery_scan(request: Request) -> Dict[str, Any]:
    if getattr(request.app.state, "backend", "native") == "ha":
        return {"status": "ok", "message": "HA backend active; use Home Assistant's UI for discovery"}
    result = await scan_all(request.app, timeout=5)
    return {"status": "ok", **result}
