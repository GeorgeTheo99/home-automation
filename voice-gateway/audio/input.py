"""Audio input abstraction for microphone capture."""

from __future__ import annotations

import logging
import os
import queue
import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import sounddevice as sd


logger = logging.getLogger(__name__)


class AudioInputError(RuntimeError):
    """Raised when microphone capture cannot be initialised."""


def _preferred_name_list(env_var: str, defaults: List[str]) -> List[str]:
    raw = os.getenv(env_var, "")
    parts = [p.strip().lower() for p in raw.split(",") if p.strip()] if raw else []
    if not parts:
        parts = defaults
    return parts


def _resolve_device(device_name: Optional[str], device_index: Optional[int]) -> Optional[int]:
    """Resolve a user-provided device hint to a PortAudio device index.

    Special-case host/backend names like 'pipewire', 'pulse', 'pulseaudio', or
    'default' to None so PortAudio uses the system default device. This avoids
    binding to a specific node that can hang on some setups.
    """
    if device_index is not None:
        return device_index
    if not device_name:
        return None
    name_lower = device_name.strip().lower()
    if name_lower in {"pipewire", "pulse", "pulseaudio", "default", "auto"}:
        # Heuristic: prefer known good input devices if available (e.g., ReSpeaker).
        try:
            prefs = _preferred_name_list("MIC_PREFERRED_NAMES", ["respeaker", "mic array", "microphone"])
            for index, info in enumerate(sd.query_devices()):
                if info.get("max_input_channels", 0) <= 0:
                    continue
                devname = str(info.get("name", "")).lower()
                if any(p in devname for p in prefs):
                    logger.info("AudioInput: auto-selecting input device %s (index=%s) via preferred names %s.", info.get("name"), index, prefs)
                    return index
        except Exception:
            pass
        return None
    for index, info in enumerate(sd.query_devices()):
        if info.get("max_input_channels", 0) <= 0:
            continue
        if name_lower in str(info.get("name", "")).lower():
            return index
    raise AudioInputError(f"No input device found matching name '{device_name}'.")


@dataclass
class AudioInput:
    sample_rate: int
    gain: float = 1.0
    device_name: Optional[str] = None
    device_index: Optional[int] = None
    block_size: int = 512

    def __post_init__(self) -> None:
        self.channels = 1
        self._closed = False
        self._pending: bytes = b""
        self._queue: "queue.Queue[bytes]" = queue.Queue(maxsize=16)
        self._last_chunk_time = time.monotonic()
        self._last_no_audio_warning = 0.0

        # Ring buffer for pre-roll (e.g., 400ms at 16kHz mono = 6400 samples = 12.8KB)
        self._preroll_ms = int(os.getenv("PREROLL_MS", "400"))
        self._preroll_max_samples = (self._preroll_ms * self.sample_rate) // 1000
        self._ring_buffer: np.ndarray = np.zeros(self._preroll_max_samples, dtype=np.int16)
        self._ring_write_pos = 0

        device = _resolve_device(self.device_name, self.device_index)
        try:
            defaults = sd.default.device
        except Exception:
            defaults = None
        # If caller asked for system default (device None), prefer the explicit
        # default input index when available to avoid some PipeWire/Pulse quirks.
        if device is None and isinstance(defaults, (list, tuple)) and len(defaults) >= 1:
            try:
                default_in = int(defaults[0])
                if default_in >= 0:
                    device = default_in
            except Exception:
                pass
        dev_info = None
        try:
            dev_info = sd.query_devices(device, kind="input")
        except Exception:
            dev_info = None
        logger.info(
            "AudioInput: opening stream (sample_rate=%d, channels=%d, target_block_size=%d, device=%s, defaults=%s, name=%s).",
            self.sample_rate,
            self.channels,
            self.block_size,
            device if device is not None else "default",
            defaults,
            (dev_info or {}).get("name"),
        )

        def _callback(indata, frames, time_info, status) -> None:
            if self._closed:
                return
            if status:
                logger.debug("AudioInput stream status: %s", status)
            if frames <= 0:
                return
            if indata is None or indata.size == 0:
                return
            try:
                # Copy the frame bytes before handing to the queue; PortAudio reuses the buffer.
                chunk = indata.tobytes()
            except Exception:
                return
            try:
                self._queue.put_nowait(chunk)
                self._last_chunk_time = time.monotonic()

                # Update ring buffer for pre-roll
                pcm = np.frombuffer(chunk, dtype=np.int16)
                samples_to_write = min(len(pcm), self._preroll_max_samples)
                if self._ring_write_pos + samples_to_write <= self._preroll_max_samples:
                    self._ring_buffer[self._ring_write_pos:self._ring_write_pos + samples_to_write] = pcm[:samples_to_write]
                    self._ring_write_pos += samples_to_write
                else:
                    # Wrap around
                    first_chunk = self._preroll_max_samples - self._ring_write_pos
                    self._ring_buffer[self._ring_write_pos:] = pcm[:first_chunk]
                    self._ring_buffer[:samples_to_write - first_chunk] = pcm[first_chunk:samples_to_write]
                    self._ring_write_pos = samples_to_write - first_chunk
            except queue.Full:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._queue.put_nowait(chunk)
                except queue.Full:
                    logger.debug("Dropping audio input chunk due to full buffer.")
        try:
            logger.info(
                "AudioInput: validating input settings (device=%s, rate=%s, channels=%s).",
                device,
                self.sample_rate,
                self.channels,
            )
            sd.check_input_settings(
                device=device,
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype="int16",
            )
            logger.info("AudioInput: device settings validated.")
        except Exception as exc:
            raise AudioInputError(f"Input device rejected settings: {exc}") from exc

        try:
            sd.default.samplerate = self.sample_rate
        except Exception:
            # Not fatal; simply rely on explicit stream configuration below.
            pass

        try:
            logger.info("AudioInput: creating input stream…")
            self._stream = sd.InputStream(
                samplerate=float(self.sample_rate),
                channels=self.channels,
                dtype="int16",
                blocksize=self.block_size,  # Explicit blocksize for lower latency
                device=device,
                latency="low",  # Low latency for faster first frame
                callback=_callback,
            )
            logger.info("AudioInput: input stream object created; starting…")
            self._stream.start()
            actual_blocksize = getattr(self._stream, "blocksize", None)
            logger.info(
                "AudioInput: stream started (blocksize=%s, device=%s).",
                actual_blocksize,
                device if device is not None else "default",
            )
        except Exception as exc:
            raise AudioInputError(f"Failed to open input stream: {exc}") from exc

    def read(self, frames: Optional[int] = None, timeout: float = 0.05) -> np.ndarray:
        """Non-blocking-ish read of PCM16 audio, returning empty array on timeout."""
        if frames is None:
            frames = self.block_size
        required_bytes = frames * self.channels * 2
        buffer = bytearray()

        if self._pending:
            buffer.extend(self._pending[:required_bytes])
            self._pending = self._pending[required_bytes:]

        deadline = time.monotonic() + timeout if timeout is not None else None
        while len(buffer) < required_bytes and not self._closed:
            remaining = required_bytes - len(buffer)
            wait = None
            if deadline is not None:
                wait = max(0.0, deadline - time.monotonic())
                if wait == 0.0:
                    break
            try:
                chunk = self._queue.get(timeout=wait)
            except queue.Empty:
                break
            if not chunk:
                continue
            if len(chunk) > remaining:
                buffer.extend(chunk[:remaining])
                self._pending = chunk[remaining:] + self._pending
            else:
                buffer.extend(chunk)

        if not buffer:
            now = time.monotonic()
            idle = now - self._last_chunk_time
            if idle > 2.0 and (now - self._last_no_audio_warning) > 5.0:
                logger.warning("AudioInput: no microphone audio received for %.1fs; awaiting stream data.", idle)
                self._last_no_audio_warning = now
            return np.zeros(0, dtype=np.int16)

        pcm = np.frombuffer(bytes(buffer), dtype=np.int16).copy()
        if pcm.size < frames:
            pcm = np.pad(pcm, (0, frames - pcm.size), mode="constant")
        if self.gain != 1.0:
            amplified = np.clip(pcm.astype(np.float32) * self.gain, -32768, 32767)
            pcm = amplified.astype(np.int16)
        return pcm

    def flush(self) -> None:
        """Clear any queued audio to reduce latency when restarting capture."""
        self._pending = b""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    def get_preroll(self, ms: Optional[int] = None) -> np.ndarray:
        """Extract last N milliseconds from ring buffer without clearing it."""
        if ms is None:
            ms = self._preroll_ms
        num_samples = min((ms * self.sample_rate) // 1000, self._preroll_max_samples)
        if num_samples == 0:
            return np.zeros(0, dtype=np.int16)

        # Extract last num_samples in chronological order
        result = np.zeros(num_samples, dtype=np.int16)
        start_pos = (self._ring_write_pos - num_samples) % self._preroll_max_samples

        if start_pos + num_samples <= self._preroll_max_samples:
            result[:] = self._ring_buffer[start_pos:start_pos + num_samples]
        else:
            # Wrapped: copy tail then head
            tail_size = self._preroll_max_samples - start_pos
            result[:tail_size] = self._ring_buffer[start_pos:]
            result[tail_size:] = self._ring_buffer[:num_samples - tail_size]

        return result

    def drain(self) -> None:
        """Clear queued audio (but preserve ring buffer for continuity)."""
        self._pending = b""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

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
        try:
            self._queue.put_nowait(b"")
        except queue.Full:
            pass

    def __enter__(self) -> "AudioInput":
        return self

    def __exit__(self, exc_type, exc, tb) -> Optional[bool]:
        self.close()
        return None
