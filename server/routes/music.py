from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request

from ..models import PlayRequest, SearchResponse


router = APIRouter()


@router.get("/music/search", response_model=SearchResponse)
async def music_search(service: str = Query(..., pattern="^(spotify|tidal)$"), q: str = Query("")):
    # Stubbed search response
    return SearchResponse(service=service, q=q, results=[])


@router.post("/music/play")
async def music_play(payload: PlayRequest, request: Request):
    if not payload.uri and not payload.trackId:
        raise HTTPException(status_code=400, detail="uri or trackId required")
    # Stub: Broadcast an event only
    await request.app.state.events.broadcast({
        "type": "music.play",
        "service": payload.service,
        "uri": payload.uri or payload.trackId,
        "target": payload.target,
    })
    return {"ok": True}

