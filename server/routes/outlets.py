from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, HTTPException, Request

from ..models import LightState


router = APIRouter()


@router.get("/outlets")
async def list_outlets(request: Request) -> list[dict]:
    if getattr(request.app.state, "backend", "native") == "ha":
        return await request.app.state.ha.list_outlets()
    reg = request.app.state.registry
    outlets = reg.list_by_type("outlet")
    for d in outlets:
        sid = d["id"]
        state = reg.outlet_states.get(sid)
        d["state"] = state or {"on": False}
    return outlets


@router.post("/outlets/{device_id}/state")
async def set_outlet_state(device_id: str, payload: LightState, request: Request) -> Dict[str, Any]:
    if getattr(request.app.state, "backend", "native") == "ha":
        if payload.on is None:
            raise HTTPException(status_code=400, detail="Missing 'on' field")
        await request.app.state.ha.set_outlet_state(device_id, on=bool(payload.on))
        return {"ok": True, "id": device_id}
    reg = request.app.state.registry
    if reg.get(device_id) is None:
        raise HTTPException(status_code=404, detail="Outlet not found")
    current = reg.outlet_states.get(device_id, {})
    if payload.on is None:
        raise HTTPException(status_code=400, detail="Missing 'on' field")
    current["on"] = bool(payload.on)
    reg.outlet_states[device_id] = current
    await request.app.state.events.broadcast({"type": "outlet.state", "id": device_id, "state": current})
    return {"ok": True, "id": device_id, "state": current}
