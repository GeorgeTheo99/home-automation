#!/usr/bin/env python3
import argparse
import sys
from typing import Optional

import numpy as np

try:
    import sounddevice as sd
except Exception as exc:
    print(f"sounddevice import failed: {exc}", file=sys.stderr)
    sys.exit(1)


def list_devices() -> None:
    default = None
    try:
        default = sd.default.device
    except Exception:
        default = None
    print("Index  Name                                   In  Out  Default")
    print("-----  -------------------------------------  --  ---  -------")
    for idx, dev in enumerate(sd.query_devices()):
        mark = ""
        if isinstance(default, (tuple, list)) and len(default) >= 2:
            in_idx = default[0]
            out_idx = default[1]
            if idx == out_idx:
                mark = "<out>"
            elif idx == in_idx:
                mark = "<in>"
        elif default is not None and idx == default:
            mark = "<def>"
        name = dev.get("name", "") or ""
        print(f"{idx:5d}  {name[:37]:37s}  {dev.get('max_input_channels', 0):2d}  {dev.get('max_output_channels', 0):3d}  {mark}")


def resolve_device(name: Optional[str], index: Optional[int]) -> Optional[int]:
    if index is not None:
        return index
    if not name:
        return None
    low = name.lower()
    for idx, dev in enumerate(sd.query_devices()):
        dname = (dev.get("name", "") or "").lower()
        if low in dname and int(dev.get("max_output_channels", 0) or 0) > 0:
            return idx
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Play a short test tone on the configured output device.")
    parser.add_argument("--device-index", type=int, help="Output device index to use.")
    parser.add_argument("--device-name", type=str, help="Substring match for output device name.")
    parser.add_argument("--frequency", type=float, default=440.0, help="Test tone frequency in Hz (default: 440).")
    parser.add_argument("--duration", type=float, default=2.0, help="Tone duration in seconds (default: 2.0).")
    parser.add_argument("--volume", type=float, default=0.3, help="Linear playback gain 0.0-1.0 (default: 0.3).")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Sample rate in Hz (default: 16000).")
    parser.add_argument("--list", action="store_true", help="List devices and exit.")
    args = parser.parse_args()

    if args.list:
        list_devices()
        return

    volume = max(0.0, min(1.0, float(args.volume)))
    duration = max(0.1, float(args.duration))
    samplerate = max(4000, int(args.sample_rate))
    freq = max(40.0, float(args.frequency))

    device_idx = resolve_device(args.device_name, args.device_index)
    if device_idx is not None:
        try:
            defaults = sd.default.device
        except Exception:
            defaults = (None, None)
        in_idx = None
        if isinstance(defaults, (tuple, list)) and len(defaults) >= 1:
            try:
                in_idx = int(defaults[0]) if defaults[0] is not None else None
            except Exception:
                in_idx = None
        if in_idx is None:
            try:
                in_idx = sd.default.device[0]  # type: ignore[index]
            except Exception:
                in_idx = None
        try:
            sd.default.device = (in_idx, device_idx)
        except Exception as exc:
            print(f"Failed setting default output device to index {device_idx}: {exc}", file=sys.stderr)

    try:
        current_default = sd.default.device
    except Exception:
        current_default = None
    print(f"Using output device: {current_default}")

    samples = int(duration * samplerate)
    t = np.linspace(0.0, duration, samples, endpoint=False)
    phase = 2.0 * np.pi * freq * t
    window = np.hanning(samples)
    tone = volume * np.sin(phase) * window

    try:
        sd.play(tone.astype(np.float32), samplerate=samplerate)
        sd.wait()
        print("Playback complete.")
    except Exception as exc:
        print(f"Playback failed: {exc}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
