from __future__ import annotations

import asyncio
import ipaddress
import socket
import threading
from typing import Dict, Optional

from fastapi import FastAPI


# ---- mDNS (zeroconf) ----

try:
    from zeroconf import ServiceBrowser, ServiceListener, Zeroconf
except Exception:  # pragma: no cover - optional dep
    Zeroconf = None  # type: ignore
    ServiceBrowser = None  # type: ignore
    ServiceListener = object  # type: ignore


MDNS_MAP = {
    "_googlecast._tcp.local.": {"type": "speaker", "adapter": "mdns", "capabilities": ["play", "pause", "volume"]},
    "_hue._tcp.local.": {"type": "light", "adapter": "mdns", "capabilities": ["bridge"]},
    "_hap._tcp.local.": {"type": "other", "adapter": "mdns", "capabilities": ["homekit"]},
}


class _MDNSListener(ServiceListener):
    def __init__(self, app: FastAPI, zc: "Zeroconf") -> None:
        self.app = app
        self.zc = zc

    def add_service(self, zc: "Zeroconf", service_type: str, name: str) -> None:  # noqa: D401
        try:
            info = zc.get_service_info(service_type, name)
            if not info:
                return
            meta = MDNS_MAP.get(service_type)
            if not meta:
                return
            addr: Optional[str] = None
            if info.addresses:
                try:
                    addr = str(ipaddress.ip_address(info.addresses[0]))
                except Exception:
                    pass
            device = {
                "id": f"mdns-{name}",
                "name": name.split(".")[0],
                "type": meta["type"],
                "adapter": meta["adapter"],
                "address": addr,
                "capabilities": meta.get("capabilities", []),
            }

            # Avoid event loop use in zeroconf thread; schedule to loop
            loop = asyncio.get_event_loop()
            loop.call_soon_threadsafe(asyncio.create_task, _register_device(self.app, device))
        except Exception:
            return


async def _register_device(app: FastAPI, device: dict) -> None:
    reg = app.state.registry
    if reg.get(device["id"]) is None:
        reg.upsert(device)
        await app.state.events.broadcast({"type": "device.discovered", "device": device})


def start_mdns(app: FastAPI) -> Optional[Zeroconf]:
    if Zeroconf is None:
        return None
    zc = Zeroconf()

    # Browse known types plus service directory to pick up others later
    listener = _MDNSListener(app, zc)
    ServiceBrowser(zc, "_services._dns-sd._udp.local.", listener)  # type discovery
    for t in MDNS_MAP.keys():
        ServiceBrowser(zc, t, listener)

    # Keep Zeroconf instance around so it doesn't get GC'd
    app.state._mdns = zc  # type: ignore[attr-defined]
    return zc


# ---- SSDP (UPnP) ----

SSDP_ADDR = ("239.255.255.250", 1900)
MSEARCH = (
    "M-SEARCH * HTTP/1.1\r\n"
    "HOST: 239.255.255.250:1900\r\n"
    "MAN: \"ssdp:discover\"\r\n"
    "MX: 2\r\n"
    "ST: ssdp:all\r\n\r\n"
).encode()


def _classify_ssdp(headers: Dict[str, str]) -> Optional[dict]:
    server = headers.get("server", "").lower()
    st = headers.get("st", "").lower()
    usn = headers.get("usn", "").lower()
    location = headers.get("location", "").lower()

    if "sonos" in server or "sonos" in usn:
        return {"type": "speaker", "capabilities": ["play", "pause", "volume"], "name": "Sonos"}
    if "philips hue" in server or "hue" in location or "hue" in usn:
        return {"type": "light", "capabilities": ["bridge"], "name": "Hue Bridge"}
    if "belkin" in server or "wemo" in usn:
        return {"type": "outlet", "capabilities": ["on"], "name": "WeMo"}
    if "mediarenderer" in st:
        return {"type": "speaker", "capabilities": ["play", "pause", "volume"], "name": "DLNA Renderer"}
    return None


async def ssdp_search(app: FastAPI, duration: float = 3.0) -> None:
    loop = asyncio.get_running_loop()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setblocking(False)
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)
    transport, _ = await loop.create_datagram_endpoint(lambda: asyncio.DatagramProtocol(), sock=sock)
    try:
        # Send a few probes
        for _ in range(2):
            transport.sendto(MSEARCH, SSDP_ADDR)
            await asyncio.sleep(0.5)

        responses: Dict[str, Dict[str, str]] = {}
        end = loop.time() + duration
        while loop.time() < end:
            try:
                data, addr = await asyncio.wait_for(loop.sock_recv(sock, 65536), timeout=0.5)
            except asyncio.TimeoutError:
                continue
            text = data.decode(errors="ignore")
            # Parse HTTP-like headers
            lines = [l for l in text.split("\r\n") if ":" in l]
            headers = {k.strip().lower(): v.strip() for k, v in (l.split(":", 1) for l in lines)}
            key = headers.get("usn") or headers.get("location") or str(addr)
            responses[key] = headers

        # Classify and register
        for key, headers in responses.items():
            meta = _classify_ssdp(headers)
            if not meta:
                continue
            id_ = f"ssdp-{(headers.get('usn') or headers.get('location') or key).replace(':', '_')}"
            device = {
                "id": id_,
                "name": meta.get("name", id_),
                "type": meta["type"],
                "adapter": "ssdp",
                "address": headers.get("location"),
                "capabilities": meta.get("capabilities", []),
            }
            await _register_device(app, device)
    finally:
        transport.close()


def start_mdns_thread(app: FastAPI) -> None:
    if Zeroconf is None:
        return
    # Run zeroconf in its own thread
    def _run():
        try:
            start_mdns(app)
        except Exception:
            pass

    t = threading.Thread(target=_run, name="mdns-browser", daemon=True)
    t.start()
    app.state._mdns_thread = t  # type: ignore[attr-defined]

