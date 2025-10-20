from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class Device(BaseModel):
    id: str
    name: str
    type: Literal["light", "speaker", "outlet", "other"]
    capabilities: List[str] = Field(default_factory=list)


class LightState(BaseModel):
    on: Optional[bool] = None
    brightness: Optional[int] = Field(default=None, ge=0, le=100)
    color: Optional[str] = None


class SpeakerState(BaseModel):
    playing: Optional[bool] = None
    volume: Optional[int] = Field(default=None, ge=0, le=100)
    uri: Optional[str] = None


class PlayRequest(BaseModel):
    service: Literal["spotify", "tidal"]
    uri: Optional[str] = None
    trackId: Optional[str] = None
    target: Optional[str] = None  # speaker id or connect endpoint


class SearchResponse(BaseModel):
    service: Literal["spotify", "tidal"]
    q: str
    results: list = Field(default_factory=list)
