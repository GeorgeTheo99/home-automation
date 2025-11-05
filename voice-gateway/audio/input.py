"""Audio input abstraction for microphone capture."""

from __future__ import annotations

import logging
import queue
import time
from dataclasses import dataclass
import threading
from typing import Optional

import numpy as np
import sounddevice as sd


logger = logging.getLogger(__name__)


class AudioInputError(RuntimeError):
    """Raised when microphone capture cannot be initialised."""


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
        logger.info(
            "AudioInput: opening stream (sample_rate=%d, channels=%d, target_block_size=%d, device=%s, defaults=%s).",
            self.sample_rate,
            self.channels,
            self.block_size,
            device if device is not None else "default",
            defaults,
        )

        def _callback(indata, frames, time_info, status) -> None:
            if self._closed:
                return
            if status:
                logger.debug("AudioInput stream status: %s", status)
            if not indata:
                return
            try:
                # Copy the frame bytes before handing to the queue; PortAudio reuses the buffer.
                chunk = bytes(indata)
            except Exception:
                return
            try:
                self._queue.put_nowait(chunk)
                self._last_chunk_time = time.monotonic()
            except queue.Full:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._queue.put_nowait(chunk)
                except queue.Full:
                    logger.debug("Dropping audio input chunk due to full buffer.")
        def _call_with_timeout(fn, timeout_sec: float, *args, **kwargs):
            holder: dict = {}
            def _runner():
                try:
                    holder["result"] = fn(*args, **kwargs)
                except BaseException as e:  # propagate any error
                    holder["error"] = e
            t = threading.Thread(target=_runner, daemon=True)
            t.start()
            t.join(timeout_sec)
            if t.is_alive():
                raise TimeoutError(f"operation timed out after {timeout_sec:.1f}s")
            if "error" in holder:
                raise holder["error"]
            return holder.get("result")

        try:
            logger.info("AudioInput: validating input settings (device=%s, rate=%s, channels=%s).", device, self.sample_rate, self.channels)
            _call_with_timeout(
                sd.check_input_settings,
                2.5,
                device=device,
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype="int16",
            )
            logger.info("AudioInput: device settings validated.")
        except TimeoutError:
            logger.warning("AudioInput: validation timed out; proceeding without check.")
        except Exception as exc:
            raise AudioInputError(f"Input device rejected settings: {exc}") from exc

        try:
            sd.default.samplerate = self.sample_rate
        except Exception:
            # Not fatal; simply rely on explicit stream configuration below.
            pass

        try:
            logger.info("AudioInput: creating input stream…")
            self._stream = _call_with_timeout(
                sd.InputStream,
                3.0,
                samplerate=float(self.sample_rate),
                channels=self.channels,
                dtype="int16",
                # Let backend choose native blocksize; we'll repack to requested size in read().
                blocksize=0,
                device=device,
                latency="high",
                callback=_callback,
            )
            logger.info("AudioInput: input stream object created; starting…")
            _call_with_timeout(self._stream.start, 2.0)
            actual_blocksize = getattr(self._stream, "blocksize", None)
            logger.info(
                "AudioInput: stream started (blocksize=%s, device=%s).",
                actual_blocksize,
                device if device is not None else "default",
            )
        except TimeoutError as exc:
            raise AudioInputError(
                "Timed out while opening microphone stream. Specify MIC_DEVICE_NAME or MIC_DEVICE_INDEX, "
                "or try a different backend device (e.g., 'pulse')."
            ) from exc
        except Exception as exc:
            raise AudioInputError(f"Failed to open input stream: {exc}") from exc

    def read(self, frames: Optional[int] = None, timeout: float = 0.2) -> np.ndarray:
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
