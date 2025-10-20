from __future__ import annotations

import asyncio
from typing import Set

from fastapi import WebSocket


class EventManager:
    def __init__(self) -> None:
        self._clients: Set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def add(self, ws: WebSocket) -> None:
        async with self._lock:
            self._clients.add(ws)
        await self.broadcast({"type": "client.joined"})

    async def remove(self, ws: WebSocket) -> None:
        async with self._lock:
            self._clients.discard(ws)
        await self.broadcast({"type": "client.left"})

    async def broadcast(self, event: dict) -> None:
        async with self._lock:
            clients = list(self._clients)
        for ws in clients:
            try:
                await ws.send_json(event)
            except Exception:
                # Drop broken clients silently for MVP
                async with self._lock:
                    self._clients.discard(ws)

