from __future__ import annotations

from typing import Dict, List, Optional


class DeviceRegistry:
    def __init__(self) -> None:
        self.devices: Dict[str, dict] = {}
        self.light_states: Dict[str, dict] = {}
        self.speaker_states: Dict[str, dict] = {}
        self.outlet_states: Dict[str, dict] = {}

    def load_from_config(self, config) -> None:
        for d in config.devices or []:
            if isinstance(d, dict) and "id" in d and "type" in d:
                self.devices[d["id"]] = d

    def upsert(self, device: dict) -> None:
        if "id" in device:
            self.devices[device["id"]] = device

    def list(self) -> List[dict]:
        return list(self.devices.values())

    def list_by_type(self, type_: str) -> List[dict]:
        return [d for d in self.devices.values() if d.get("type") == type_]

    def get(self, device_id: str) -> Optional[dict]:
        return self.devices.get(device_id)

    def counts(self) -> dict:
        return {
            "total": len(self.devices),
            "lights": len(self.list_by_type("light")),
            "speakers": len(self.list_by_type("speaker")),
            "outlets": len(self.list_by_type("outlet")),
        }
