from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, Tuple

from ...util.colors import parse_color, pct_to_16bit


async def discover(timeout: int = 4) -> List[Dict[str, Any]]:
    try:
        # lifxlan is synchronous; run in a thread
        from lifxlan import LifxLAN
    except Exception:
        return []

    def _scan() -> List[Dict[str, Any]]:
        try:
            lan = LifxLAN()
            lights = lan.get_lights()
        except Exception:
            return []
        found: List[Dict[str, Any]] = []
        for light in lights or []:
            try:
                label = light.get_label() or "LIFX Light"
                ip = light.get_ip_addr()
                mac = light.get_mac_addr()
                found.append(
                    {
                        "id": f"lifx-{mac}",
                        "name": label,
                        "type": "light",
                        "adapter": "lifx",
                        "address": ip,
                        "mac": mac,
                        "capabilities": ["on", "brightness", "color"],
                    }
                )
            except Exception:
                continue
        return found

    try:
        return await asyncio.to_thread(_scan)
    except Exception:
        return []


async def set_state(device: dict, on: Optional[bool] = None, brightness: Optional[int] = None, color: Optional[str] = None) -> dict:
    """Apply state to a LIFX light.

    brightness: 0-100
    color: named (red, green, blue, white, warm, cool) or #RRGGBB
    """
    try:
        from lifxlan import Light
    except Exception as e:
        raise RuntimeError(f"lifxlan not available: {e}")

    mac = device.get("mac")
    ip = device.get("address")
    if not mac or not ip:
        raise RuntimeError("Missing MAC or IP for LIFX device")

    light = Light(mac, ip)

    def _apply():
        # Power
        if on is not None:
            try:
                light.set_power(65535 if on else 0, rapid=False)
            except Exception:
                pass

        # Color
        if color:
            hsbk = parse_color(color)
            if hsbk:
                try:
                    light.set_color(hsbk, rapid=False)
                except Exception:
                    pass

        # Brightness
        if brightness is not None:
            try:
                # Fetch current color and adjust brightness only
                h, s, _b, k = light.get_color()
                bri16 = pct_to_16bit(float(brightness))
                light.set_color((h, s, bri16, k), rapid=False)
            except Exception:
                pass

        # Return current snapshot
        snapshot = {"on": on, "brightness": brightness, "color": color}
        try:
            import time
            time.sleep(0.2)
            # Try to read back power and color
            pwr = light.get_power()  # 0 or 65535
            snapshot["on"] = bool(pwr)
            h, s, b, k = light.get_color()
            snapshot["brightness"] = int(round((b / 65535.0) * 100))
        except Exception:
            pass
        return snapshot

    return await asyncio.to_thread(_apply)
