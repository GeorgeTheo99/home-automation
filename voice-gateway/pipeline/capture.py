"""Speech capture helpers for the voice gateway pipeline."""

from __future__ import annotations

import logging
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
) -> Optional[np.ndarray]:
    """Capture audio until the VAD decides the utterance is complete."""
    vad.reset()
    captured: List[np.ndarray] = []
    speech_detected = False
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
