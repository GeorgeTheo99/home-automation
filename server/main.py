from __future__ import annotations

import asyncio
import os
from typing import List

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from .config import Config
from .core.events import EventManager
from .core.registry import DeviceRegistry
from .routes.discovery import router as discovery_router
from .routes.lights import router as lights_router
from .routes.outlets import router as outlets_router
from .routes.speakers import router as speakers_router
from .routes.music import router as music_router
from .discovery_generic import start_mdns_thread, ssdp_search
from .integrations.ha import HAClient


config = Config.load()
registry = DeviceRegistry()
events = EventManager()


def create_app() -> FastAPI:
    app = FastAPI(title="Home Automation Server", version="0.1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Attach shared state
    app.state.config = config
    app.state.registry = registry
    app.state.events = events
    app.state.backend = os.getenv("BACKEND", "native").lower()
    if app.state.backend == "ha":
        ha_url = os.getenv("HA_URL")
        ha_token = os.getenv("HA_TOKEN")
        if not ha_url or not ha_token:
            raise RuntimeError("BACKEND=ha requires HA_URL and HA_TOKEN to be set")
        app.state.ha = HAClient(ha_url, ha_token)

    # Routers
    app.include_router(discovery_router, prefix="/api")
    app.include_router(lights_router, prefix="/api")
    app.include_router(speakers_router, prefix="/api")
    app.include_router(outlets_router, prefix="/api")
    app.include_router(music_router, prefix="/api")

    @app.get("/api/health")
    async def health():
        return {
            "status": "ok",
            "port": int(os.getenv("HOME_AUTOMATION_PORT", "8123")),
            "discovery": os.getenv("HOME_AUTOMATION_DISCOVERY", "true").lower() == "true",
            "devices": registry.counts(),
            "backend": app.state.backend,
        }

    @app.websocket("/api/events")
    async def events_ws(ws: WebSocket):
        await ws.accept()
        await events.add(ws)
        try:
            while True:
                # keepalive; we don't process incoming messages yet
                await ws.receive_text()
        except WebSocketDisconnect:
            await events.remove(ws)

    @app.on_event("startup")
    async def startup():
        if app.state.backend == "native":
            # Load devices from config if present
            registry.load_from_config(config)
            # Optionally start passive discovery later (stubs for now)
            if os.getenv("HOME_AUTOMATION_DISCOVERY", "true").lower() == "true":
                asyncio.create_task(_announce_startup())
                # Start periodic discovery
                from .discovery import periodic_scan
                asyncio.create_task(periodic_scan(app, interval_seconds=300))
                # Start mDNS watcher thread and periodic SSDP scans
                start_mdns_thread(app)
                async def _ssdp_loop():
                    while True:
                        try:
                            await ssdp_search(app, duration=3.0)
                        except Exception:
                            pass
                        await asyncio.sleep(300)
                asyncio.create_task(_ssdp_loop())
        else:
            asyncio.create_task(_announce_startup())

    return app

    
@app.on_event("shutdown")
async def shutdown_event():
    if getattr(app.state, "backend", "native") == "ha":
        try:
            await app.state.ha.close()
        except Exception:
            pass


async def _announce_startup():
    await asyncio.sleep(0.1)
    await events.broadcast({"type": "server.started"})


app = create_app()
