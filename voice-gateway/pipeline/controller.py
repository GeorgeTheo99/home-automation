"""Pipeline controller that orchestrates wake, capture, stream, and follow-up states."""

from __future__ import annotations

import logging
import time
from enum import Enum, auto
from typing import Callable, Optional

import numpy as np

from audio.input import AudioInput
from config import AppConfig
from gpt.realtime_client import RealtimeClient, RealtimeError
from pipeline.capture import record_utterance
from vad.silero import VoiceActivityDetector
from wake.vosk import VoskWakeWordDetector

logger = logging.getLogger(__name__)

AudioConsumer = Callable[[np.ndarray], None]


class GatewayState(Enum):
    WAIT_WAKE = auto()
    CAPTURE = auto()
    STREAM = auto()
    FOLLOWUP_ARMED = auto()


class PipelineController:
    """Finite state machine that drives the voice gateway pipeline."""

    def __init__(
        self,
        config: AppConfig,
        audio_input: AudioInput,
        wake_detector: VoskWakeWordDetector,
        vad: VoiceActivityDetector,
        realtime_client: RealtimeClient,
        playback: AudioConsumer,
    ) -> None:
        self._config = config
        self._audio_input = audio_input
        self._wake_detector = wake_detector
        self._vad = vad
        self._realtime_client = realtime_client
        self._playback = playback

        self._block_size = self._wake_detector.frame_length
        self._max_capture_sec = max(self._config.vad.max_segment_ms / 1000.0, 1.0)

        self._running = True
        self._state = GatewayState.WAIT_WAKE
        self._pending_audio: Optional[np.ndarray] = None
        self._followup_deadline: Optional[float] = None

    def request_shutdown(self) -> None:
        """Stop the controller loop on the next iteration."""
        self._running = False

    def run(self) -> None:
        logger.info("Voice gateway ready. Say the wake word to begin.")
        while self._running:
            if self._state == GatewayState.WAIT_WAKE:
                chunk = self._audio_input.read(self._block_size)
                if self._wake_detector.process(chunk):
                    logger.info("Wake word detected.")
                    self._state = GatewayState.CAPTURE
                    self._vad.reset()
                    self._audio_input.flush()
                    continue

            elif self._state == GatewayState.CAPTURE:
                utterance = record_utterance(
                    vad=self._vad,
                    reader=lambda frames: self._audio_input.read(frames),
                    block_size=self._block_size,
                    listen_timeout=None,
                    max_capture_sec=self._max_capture_sec,
                    should_stop=self._should_stop,
                )
                if not self._running:
                    break
                if utterance is None:
                    logger.info("No speech detected after wake word.")
                    self._audio_input.flush()
                    self._state = GatewayState.WAIT_WAKE
                    continue
                self._pending_audio = utterance
                self._state = GatewayState.STREAM

            elif self._state == GatewayState.STREAM:
                if self._pending_audio is None or self._pending_audio.size == 0:
                    self._state = GatewayState.WAIT_WAKE
                    continue
                logger.info("Streaming utterance to realtime API (%d samples).", self._pending_audio.size)
                try:
                    response = self._realtime_client.send_utterance(
                        pcm16_audio=self._pending_audio,
                        source_sample_rate=self._config.audio_input.sample_rate,
                        audio_consumer=self._playback,
                    )
                except RealtimeError as exc:
                    logger.error("Realtime processing failed: %s", exc)
                    self._audio_input.flush()
                    self._pending_audio = None
                    self._state = GatewayState.WAIT_WAKE
                    continue

                if response.text:
                    logger.info("Assistant: %s", response.text)
                else:
                    logger.info("Assistant response completed.")

                self._pending_audio = None
                if self._should_arm_followup(response.text):
                    followup_window = self._config.follow_up.window_sec
                    self._followup_deadline = time.monotonic() + followup_window
                    guard_delay = max(0.0, self._config.follow_up.guard_ms / 1000.0)
                    if guard_delay:
                        time.sleep(guard_delay)
                    self._vad.reset()
                    logger.info(
                        "Follow-up window armed for %.1fs (guard=%.0f ms).",
                        followup_window,
                        self._config.follow_up.guard_ms,
                    )
                    self._state = GatewayState.FOLLOWUP_ARMED
                else:
                    self._audio_input.flush()
                    self._state = GatewayState.WAIT_WAKE

            elif self._state == GatewayState.FOLLOWUP_ARMED:
                if self._followup_deadline is None or time.monotonic() > self._followup_deadline:
                    logger.info("Follow-up window expired.")
                    self._audio_input.flush()
                    self._state = GatewayState.WAIT_WAKE
                    continue

                remaining = self._followup_deadline - time.monotonic()
                utterance = record_utterance(
                    vad=self._vad,
                    reader=lambda frames: self._audio_input.read(frames),
                    block_size=self._block_size,
                    listen_timeout=remaining,
                    max_capture_sec=self._max_capture_sec,
                    min_peak=self._config.follow_up.min_peak,
                    min_rms=self._config.follow_up.min_rms,
                    should_stop=self._should_stop,
                )
                if not self._running:
                    break
                if utterance is None:
                    self._audio_input.flush()
                    self._state = GatewayState.WAIT_WAKE
                    continue
                self._pending_audio = utterance
                self._state = GatewayState.STREAM

            else:
                self._state = GatewayState.WAIT_WAKE

        # Flush pending audio when stopping to minimise latency on next start.
        self._audio_input.flush()

    def _should_stop(self) -> bool:
        return not self._running

    def _should_arm_followup(self, response_text: str) -> bool:
        follow_up = self._config.follow_up
        if not follow_up.enabled:
            return False
        if follow_up.arm_mode == "always":
            return True
        return response_text.strip().endswith("?")
