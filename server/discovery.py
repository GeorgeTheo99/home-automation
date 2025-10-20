from __future__ import annotations

import asyncio
from typing import Dict, List

from fastapi import FastAPI


async def scan_all(app: FastAPI, timeout: int = 5) -> Dict[str, int]:
    from .adapters.lights import kasa as kasa_discovery
    from .adapters.lights import lifx as lifx_discovery
    from .adapters.lights import tuya as tuya_discovery

    tasks = [
        kasa_discovery.discover(timeout=timeout),
        lifx_discovery.discover(timeout=timeout),
        tuya_discovery.discover(timeout=timeout),
    ]
    results: List[List[dict]] = await asyncio.gather(*tasks, return_exceptions=False)

    registry = app.state.registry
    events = app.state.events
    added = 0
    for group in results:
        for d in group:
            if registry.get(d["id"]) is None:
                registry.upsert(d)
                added += 1
                await events.broadcast({"type": "device.discovered", "device": d})

    return {"added": added, "total": len(registry.list())}


async def periodic_scan(app: FastAPI, interval_seconds: int = 300) -> None:
    while True:
        try:
            await scan_all(app, timeout=5)
        except Exception:
            pass
        await asyncio.sleep(interval_seconds)

