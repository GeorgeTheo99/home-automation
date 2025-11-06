"""Audio output abstraction for playback."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional, List

import numpy as np
import sounddevice as sd


logger = logging.getLogger(__name__)


class AudioOutputError(RuntimeError):
    """Raised when speaker playback cannot be initialised."""


def _preferred_name_list(env_var: str, defaults: List[str]) -> List[str]:
    raw = os.getenv(env_var, "")
    parts = [p.strip().lower() for p in raw.split(",") if p.strip()] if raw else []
    if not parts:
        parts = defaults
    return parts


def _resolve_device(device_name: Optional[str], device_index: Optional[int]) -> Optional[int]:
    if device_index is not None:
        return device_index
    if not device_name:
        return None
    name_lower = device_name.strip().lower()
    if name_lower in {"pipewire", "pulse", "pulseaudio", "default", "auto"}:
        # Heuristic: prefer known good output devices if available (e.g., Edifier).
        try:
            prefs = _preferred_name_list("SPEAKER_PREFERRED_NAMES", ["edifier", "r19u", "speaker"])
            # Match devices and score by specificity (prefer exact matches over generic terms)
            candidates = []
            for index, info in enumerate(sd.query_devices()):
                if info.get("max_output_channels", 0) <= 0:
                    continue
                devname = str(info.get("name", "")).lower()
                for pref_idx, pref in enumerate(prefs):
                    if pref in devname:
                        # Score: lower is better (prefer earlier prefs, avoid generic "usb audio")
                        score = pref_idx if pref not in {"usb audio", "speaker"} else pref_idx + 100
                        candidates.append((score, index, info.get("name")))
                        break

            if candidates:
                # Pick the best (lowest score)
                candidates.sort()
                _, best_index, best_name = candidates[0]
                logger.info("AudioOutput: auto-selecting output device %s (index=%s) via preferred names %s.", best_name, best_index, prefs)
                return best_index
        except Exception:
            pass
        return None
    for index, info in enumerate(sd.query_devices()):
        if info.get("max_output_channels", 0) <= 0:
            continue
        if name_lower in str(info.get("name", "")).lower():
            return index
    raise AudioOutputError(f"No output device found matching name '{device_name}'.")


def _device_default_sample_rate(device: Optional[int]) -> Optional[int]:
    try:
        info = sd.query_devices(device, "output")
    except Exception:
        return None
    rate = info.get("default_samplerate")
    if rate is None:
        return None
    try:
        return int(round(float(rate)))
    except (TypeError, ValueError):
        return None


@dataclass
class AudioOutput:
    sample_rate: int
    device_name: Optional[str] = None
    device_index: Optional[int] = None
    block_size: int = 1024

    def __post_init__(self) -> None:
        self.channels = 1
        self.input_sample_rate = int(self.sample_rate)
        self._closed = False
        device = _resolve_device(self.device_name, self.device_index)
        dev_info = None
        try:
            dev_info = sd.query_devices(device, kind="output")
        except Exception:
            dev_info = None
        logger.info(
            "AudioOutput: opening stream (requested_sample_rate=%d, channels=%d, block_size=%d, device=%s, name=%s).",
            self.input_sample_rate,
            self.channels,
            self.block_size,
            device if device is not None else "default",
            (dev_info or {}).get("name"),
        )

        try:
            sd.check_output_settings(device=device, samplerate=self.input_sample_rate, channels=self.channels, dtype="int16")
        except Exception as exc:
            # Defer to opening logic to try fallbacks, but log for diagnostics.
            logger.warning("AudioOutput: device validation failed for %s Hz (%s); will attempt open anyway.", self.input_sample_rate, exc)

        try:
            self._stream = self._open_stream(device, self.input_sample_rate)
            self.sample_rate = int(self.input_sample_rate)
            logger.info("AudioOutput: stream started at %d Hz.", self.sample_rate)
        except Exception as exc:
            if isinstance(exc, sd.PortAudioError) and "Invalid sample rate" in str(exc):
                fallback_rate = _device_default_sample_rate(device)
                if fallback_rate and fallback_rate != self.input_sample_rate:
                    logger.warning(
                        "Output device rejected %s Hz; falling back to default %s Hz",
                        self.input_sample_rate,
                        fallback_rate,
                    )
                    try:
                        self._stream = self._open_stream(device, fallback_rate)
                        self.sample_rate = int(fallback_rate)
                        logger.info("AudioOutput: fallback stream started at %d Hz.", self.sample_rate)
                    except Exception as fallback_exc:
                        raise AudioOutputError(
                            f"Failed to open output stream: requested {self.input_sample_rate} Hz invalid "
                            f"and fallback {fallback_rate} Hz failed: {fallback_exc}"
                        ) from fallback_exc
                else:
                    raise AudioOutputError(
                        f"Failed to open output stream with sample rate {self.input_sample_rate} Hz: {exc}"
                    ) from exc
            else:
                raise AudioOutputError(f"Failed to open output stream: {exc}") from exc

    def _open_stream(self, device: Optional[int], samplerate: int) -> sd.RawOutputStream:
        stream = sd.RawOutputStream(
            samplerate=float(samplerate),
            channels=self.channels,
            dtype="int16",
            # Let backend pick a native blocksize for stability
            blocksize=0,
            device=device,
            latency="high",
        )
        stream.start()
        return stream

    @staticmethod
    def _resample_linear(pcm16: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
        if source_rate == target_rate or pcm16.size == 0:
            return pcm16.astype(np.int16, copy=False)
        duration = pcm16.size / float(source_rate)
        new_size = int(round(duration * target_rate))
        if new_size <= 1:
            return pcm16.astype(np.int16, copy=True)
        x_old = np.arange(pcm16.size, dtype=np.float32)
        x_new = np.linspace(0, pcm16.size - 1, new_size, dtype=np.float32)
        interpolated = np.interp(x_new, x_old, pcm16.astype(np.float32))
        return np.clip(interpolated, -32768, 32767).astype(np.int16)

    def write(self, pcm16: np.ndarray) -> None:
        """Blocking write of PCM16 audio."""
        if pcm16.size == 0:
            return
        if self._closed:
            return
        if pcm16.dtype != np.int16:
            pcm16 = np.clip(pcm16.astype(np.float32), -32768, 32767).astype(np.int16)
        original_size = pcm16.size
        if self.sample_rate != self.input_sample_rate:
            pcm16 = self._resample_linear(pcm16, self.input_sample_rate, self.sample_rate)
            logger.debug(
                "AudioOutput: resampled %d→%d Hz (%d samples → %d samples)",
                self.input_sample_rate,
                self.sample_rate,
                original_size,
                pcm16.size,
            )
        logger.debug("AudioOutput: writing %d samples to device", pcm16.size)
        self._stream.write(pcm16.tobytes())

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        stream = getattr(self, "_stream", None)
        if stream is None:
            return
        try:
            stream.stop()
        except Exception:
            pass
        try:
            stream.close()
        except Exception:
            pass
        self._stream = None

    def __enter__(self) -> "AudioOutput":
        return self

    def __exit__(self, exc_type, exc, tb) -> Optional[bool]:
        self.close()
        return None
