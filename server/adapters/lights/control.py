from __future__ import annotations

from typing import Optional

from fastapi import HTTPException

from . import lifx as lifx_adapter


async def apply_state(device: dict, on: Optional[bool] = None, brightness: Optional[int] = None, color: Optional[str] = None) -> dict:
    adapter = (device.get("adapter") or "").lower()
    try:
        if adapter == "lifx":
            return await lifx_adapter.set_state(device, on=on, brightness=brightness, color=color)
        # Future: kasa, tuya
        raise NotImplementedError(f"Adapter '{adapter}' does not support control yet")
    except NotImplementedError as e:
        raise HTTPException(status_code=501, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to apply state: {e}")

