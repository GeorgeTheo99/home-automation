from __future__ import annotations

from typing import Any, Dict, List


def _safe_get(d: dict, key: str, default=None):
    try:
        return d.get(key, default)
    except Exception:
        return default


async def discover(timeout: int = 5) -> List[Dict[str, Any]]:
    try:
        import tinytuya
    except Exception:
        return []

    try:
        res = tinytuya.deviceScan(timeout=timeout)
    except Exception:
        return []

    results: List[Dict[str, Any]] = []
    for dev_id, info in (res or {}).items():
        ip = _safe_get(info, "ip")
        name = _safe_get(info, "name") or _safe_get(info, "gwId") or "Tuya Device"
        ver = _safe_get(info, "version")
        # Classification is best-effort without cloud metadata
        dev_type = "other"
        results.append(
            {
                "id": f"tuya-{dev_id}",
                "name": name,
                "type": dev_type,
                "adapter": "tuya",
                "address": ip,
                "tuya_id": dev_id,
                "version": ver,
                "capabilities": [],
            }
        )
    return results

