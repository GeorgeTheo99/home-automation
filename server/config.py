from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import yaml


CONFIG_DIR = Path(os.getenv("HOME_AUTOMATION_CONFIG_DIR", "/app/config")).resolve()


@dataclass
class Config:
    devices: List[Dict[str, Any]]
    scenes: List[Dict[str, Any]]
    automations: List[Dict[str, Any]]

    @staticmethod
    def load() -> "Config":
        devices = _load_yaml(CONFIG_DIR / "devices.yaml") or []
        scenes = _load_yaml(CONFIG_DIR / "scenes.yaml") or []
        automations = _load_yaml(CONFIG_DIR / "automations.yaml") or []
        return Config(devices=devices, scenes=scenes, automations=automations)


def _load_yaml(path: Path):
    try:
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                return yaml.safe_load(f)
    except Exception:
        # Swallow config errors for MVP; log later
        return None
    return None

