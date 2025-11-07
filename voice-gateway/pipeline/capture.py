"""Speech capture helpers for the voice gateway pipeline."""

from __future__ import annotations

import logging
import os
import time
from typing import Callable, List, Optional

import numpy as np

from vad.silero import VoiceActivityDetector

logger = logging.getLogger(__name__)

AudioReader = Callable[[int], np.ndarray]


def _rms_level(chunk: np.ndarray) -> float:
    if chunk.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2))) / 32768.0


def _peak_level(chunk: np.ndarray) -> int:
    if chunk.size == 0:
        return 0
    return int(np.max(np.abs(chunk)))


def record_utterance(
    vad: VoiceActivityDetector,
    reader: AudioReader,
    block_size: int,
    listen_timeout: Optional[float],
    max_capture_sec: float,
    min_peak: Optional[int] = None,
    min_rms: Optional[float] = None,
    should_stop: Optional[Callable[[], bool]] = None,
    reset_vad: bool = True,
    initial_audio: Optional[np.ndarray] = None,
    start_on_wake: bool = False,
) -> Optional[np.ndarray]:
    """Capture audio until the VAD decides the utterance is complete."""
    if reset_vad:
        vad.reset()
        logger.debug("VAD reset at capture start.")
    else:
        logger.debug("VAD warm start (no reset).")

    captured: List[np.ndarray] = []
    speech_detected = start_on_wake  # Start immediately if wake-triggered

    # Warm VAD with pre-roll but don't include full pre-roll in capture
    if initial_audio is not None and initial_audio.size > 0:
        preroll_ms = (initial_audio.size * 1000) // 16000
        logger.info("Warming VAD with %d ms of pre-roll.", preroll_ms)

        # Feed pre-roll to VAD in block-sized chunks
        for i in range(0, len(initial_audio), block_size):
            chunk = initial_audio[i:i + block_size]
            if chunk.size == block_size:
                vad.process(chunk)

        # Include only the LAST 75-100ms of pre-roll to catch fast speakers
        # without sending the full wake word
        include_ms = int(os.getenv("PREROLL_INCLUDE_MS", "75"))
        include_samples = (include_ms * 16000) // 1000  # Assume 16kHz
        if include_samples > 0 and initial_audio.size >= include_samples:
            tail = initial_audio[-include_samples:]
            captured.append(tail)
            logger.debug("Included last %d ms of pre-roll in captured audio.", include_ms)

    if start_on_wake:
        logger.info("Capture starting immediately (implicit start on wake).")

    start_time = time.monotonic()
    listen_deadline = start_time + listen_timeout if listen_timeout is not None else None
    end_deadline = start_time + max_capture_sec

    while time.monotonic() < end_deadline:
        if should_stop is not None and should_stop():
            logger.debug("record_utterance: stop requested externally.")
            return None
        chunk = reader(block_size)
        if chunk.size == 0:
            if should_stop is not None and should_stop():
                logger.debug("record_utterance: received empty chunk and stop requested.")
                return None
            continue
        captured.append(chunk)
        result = vad.process(chunk)
        if result.speech_started:
            logger.info(
                "VAD speech started (prob=%.2f, rms=%.4f, peak=%d).",
                result.probability,
                _rms_level(chunk),
                _peak_level(chunk),
            )
        if result.speech_ended or result.forced:
            logger.info(
                "VAD speech ended (duration_ms=%.0f, silence_ms=%.0f, forced=%s).",
                result.speech_duration_ms,
                result.silence_duration_ms,
                result.forced,
            )

        if not speech_detected:
            if listen_deadline is not None and time.monotonic() > listen_deadline:
                logger.info("record_utterance: listen timeout reached before speech start.")
                return None
            if result.speech_started:
                meets_peak = min_peak is None or _peak_level(chunk) >= min_peak
                meets_rms = min_rms is None or _rms_level(chunk) >= min_rms
                if meets_peak and meets_rms:
                    speech_detected = True
                else:
                    captured.clear()
                    if reset_vad:  # Only reset if we're managing VAD state
                        vad.reset()
                    continue
        else:
            if result.speech_ended or result.forced:
                break
    if not speech_detected:
        logger.info("record_utterance: no speech detected.")
        return None
    if not captured:
        logger.debug("record_utterance: speech detected but no audio captured.")
        return None
    return np.concatenate(captured)
