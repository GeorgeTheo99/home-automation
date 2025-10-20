from __future__ import annotations

import colorsys
from typing import Optional, Tuple


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def pct_to_16bit(pct: float) -> int:
    pct = clamp(pct, 0.0, 100.0)
    return int(round((pct / 100.0) * 65535))


def rgb_to_hsbk(r: int, g: int, b: int, kelvin: int = 3500) -> Tuple[int, int, int, int]:
    r_n, g_n, b_n = r / 255.0, g / 255.0, b / 255.0
    h, s, v = colorsys.rgb_to_hsv(r_n, g_n, b_n)
    hue = int(round(h * 65535))
    sat = int(round(s * 65535))
    bri = int(round(v * 65535))
    return hue, sat, bri, kelvin


NAMED = {
    "red": (0x0000, 65535, 65535, 3500),
    "green": (21845, 65535, 65535, 3500),
    "blue": (43690, 65535, 65535, 3500),
    "white": (0, 0, 65535, 3500),
    "warm": (0, 0, 65535, 2700),
    "cool": (0, 0, 65535, 6500),
}


def parse_color(value: str) -> Optional[Tuple[int, int, int, int]]:
    if not value:
        return None
    v = value.strip().lower()
    if v in NAMED:
        return NAMED[v]
    if v.startswith("#") and len(v) == 7:
        try:
            r = int(v[1:3], 16)
            g = int(v[3:5], 16)
            b = int(v[5:7], 16)
            return rgb_to_hsbk(r, g, b)
        except Exception:
            return None
    return None


def parse_color_rgb(value: str) -> Optional[Tuple[int, int, int]]:
    v = (value or "").strip().lower()
    if v in ("white", "warm", "cool"):
        return (255, 255, 255)
    if v == "red":
        return (255, 0, 0)
    if v == "green":
        return (0, 255, 0)
    if v == "blue":
        return (0, 0, 255)
    if v.startswith("#") and len(v) == 7:
        try:
            r = int(v[1:3], 16)
            g = int(v[3:5], 16)
            b = int(v[5:7], 16)
            return (r, g, b)
        except Exception:
            return None
    return None
