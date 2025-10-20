from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, HTTPException, Request

from ..models import SpeakerState


router = APIRouter()


@router.get("/speakers")
async def list_speakers(request: Request) -> list[dict]:
    reg = request.app.state.registry
    speakers = reg.list_by_type("speaker")
    for d in speakers:
        sid = d["id"]
        state = reg.speaker_states.get(sid)
        d["state"] = state or {"playing": False, "volume": 25}
    return speakers


@router.post("/speakers/{device_id}/play")
async def play(device_id: str, request: Request) -> Dict[str, Any]:
    reg = request.app.state.registry
    if reg.get(device_id) is None:
        raise HTTPException(status_code=404, detail="Speaker not found")
    state = reg.speaker_states.get(device_id, {"playing": False, "volume": 25})
    state["playing"] = True
    reg.speaker_states[device_id] = state
    await request.app.state.events.broadcast({"type": "speaker.play", "id": device_id})
    return {"ok": True}


@router.post("/speakers/{device_id}/pause")
async def pause(device_id: str, request: Request) -> Dict[str, Any]:
    reg = request.app.state.registry
    if reg.get(device_id) is None:
        raise HTTPException(status_code=404, detail="Speaker not found")
    state = reg.speaker_states.get(device_id, {"playing": False, "volume": 25})
    state["playing"] = False
    reg.speaker_states[device_id] = state
    await request.app.state.events.broadcast({"type": "speaker.pause", "id": device_id})
    return {"ok": True}


@router.post("/speakers/{device_id}/volume")
async def set_volume(device_id: str, payload: SpeakerState, request: Request) -> Dict[str, Any]:
    if payload.volume is None:
        raise HTTPException(status_code=400, detail="Missing volume")
    reg = request.app.state.registry
    if reg.get(device_id) is None:
        raise HTTPException(status_code=404, detail="Speaker not found")
    state = reg.speaker_states.get(device_id, {"playing": False, "volume": 25})
    state["volume"] = int(payload.volume)
    reg.speaker_states[device_id] = state
    await request.app.state.events.broadcast({"type": "speaker.volume", "id": device_id, "volume": state["volume"]})
    return {"ok": True}

