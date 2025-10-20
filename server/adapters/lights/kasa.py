from __future__ import annotations

import asyncio
from typing import Any, Dict, List


async def discover(timeout: int = 5) -> List[Dict[str, Any]]:
    try:
        from kasa import Discover
    except Exception:
        return []

    try:
        devices = await Discover.discover(timeout=timeout)
    except Exception:
        return []

    results: List[Dict[str, Any]] = []
    for _, dev in devices.items():
        try:
            # Query basic info
            await dev.update()
            alias = getattr(dev, "alias", None) or getattr(dev, "sys_info", {}).get("alias") or "Kasa Device"
            host = getattr(dev, "host", None)
            mac = getattr(dev, "mac", None) or getattr(dev, "device_id", None)
            cls = dev.__class__.__name__.lower()
            if getattr(dev, "is_bulb", False) or ("bulb" in cls or "light" in cls):
                dev_type = "light"
                capabilities = ["on", "brightness"]
                if getattr(dev, "is_color", False):
                    capabilities.append("color")
            elif getattr(dev, "is_plug", False) or ("plug" in cls or "strip" in cls or "socket" in cls):
                dev_type = "outlet"
                capabilities = ["on"]
            else:
                dev_type = "other"
                capabilities = []

            results.append(
                {
                    "id": f"kasa-{mac or host}",
                    "name": alias,
                    "type": dev_type,
                    "adapter": "kasa",
                    "address": host,
                    "mac": mac,
                    "capabilities": capabilities,
                }
            )
        except Exception:
            continue
    return results

