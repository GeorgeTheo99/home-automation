from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, HTTPException, Request

from ..models import LightState


router = APIRouter()


@router.get("/lights")
async def list_lights(request: Request) -> list[dict]:
    if getattr(request.app.state, "backend", "native") == "ha":
        return await request.app.state.ha.list_lights()
    reg = request.app.state.registry
    lights = reg.list_by_type("light")
    # merge state
    for d in lights:
        sid = d["id"]
        state = reg.light_states.get(sid)
        d["state"] = state or {"on": False}
    return lights


@router.post("/lights/{device_id}/state")
async def set_light_state(device_id: str, payload: LightState, request: Request) -> Dict[str, Any]:
    if getattr(request.app.state, "backend", "native") == "ha":
        # device_id is HA entity_id when using HA backend
        await request.app.state.ha.set_light_state(device_id, on=payload.on, brightness=payload.brightness, color=payload.color)
        return {"ok": True, "id": device_id}

    reg = request.app.state.registry
    if reg.get(device_id) is None:
        raise HTTPException(status_code=404, detail="Light not found")
    # Apply to underlying adapter if available
    device = reg.get(device_id)
    try:
        from ..adapters.lights.control import apply_state as apply_light_state
        applied = await apply_light_state(device, on=payload.on, brightness=payload.brightness, color=payload.color)
        current = reg.light_states.get(device_id, {})
        current.update({k: v for k, v in applied.items() if v is not None})
        reg.light_states[device_id] = current
        await request.app.state.events.broadcast({"type": "light.state", "id": device_id, "state": current})
        return {"ok": True, "id": device_id, "state": current}
    except HTTPException:
        raise
    except Exception as e:
        # Fallback: store desired state only
        current = reg.light_states.get(device_id, {})
        if payload.on is not None:
            current["on"] = bool(payload.on)
        if payload.brightness is not None:
            current["brightness"] = int(payload.brightness)
        if payload.color is not None:
            current["color"] = payload.color
        reg.light_states[device_id] = current
        await request.app.state.events.broadcast({"type": "light.state", "id": device_id, "state": current, "warning": str(e)})
        return {"ok": True, "id": device_id, "state": current, "warning": str(e)}
