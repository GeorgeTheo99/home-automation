"""Pipeline controller that orchestrates wake, capture, stream, and follow-up states."""

from __future__ import annotations

import logging
import os
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

        # Zero-start-delay configuration
        self._implicit_start = os.getenv("IMPLICIT_START_ON_WAKE", "true").lower() in {"true", "1", "yes", "on"}
        self._vad_warm_on_wake = os.getenv("VAD_WARM_ON_WAKE", "true").lower() in {"true", "1", "yes", "on"}
        self._post_wake_timeout = float(os.getenv("POST_WAKE_TIMEOUT", "3.0"))

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
                    logger.info("🎤 Microphone OPEN (reason: wake word detected)")

                    # Extract pre-roll and drain queue
                    preroll = self._audio_input.get_preroll()  # Uses default from config
                    self._audio_input.drain()  # Clear queued audio after extracting pre-roll

                    preroll_ms = (preroll.size * 1000) // self._config.audio_input.sample_rate
                    logger.info("Captured %d ms pre-roll (%d samples).", preroll_ms, preroll.size)

                    self._state = GatewayState.CAPTURE
                    self._preroll_audio = preroll  # Store for use in capture
                    continue

            elif self._state == GatewayState.CAPTURE:
                preroll = getattr(self, "_preroll_audio", None)
                self._preroll_audio = None  # Clear after reading

                utterance = record_utterance(
                    vad=self._vad,
                    reader=lambda frames: self._audio_input.read(frames),
                    block_size=self._block_size,
                    listen_timeout=self._post_wake_timeout,  # Shorter timeout for post-wake silence
                    max_capture_sec=self._max_capture_sec,
                    should_stop=self._should_stop,
                    reset_vad=not self._vad_warm_on_wake,  # Keep VAD warm if configured
                    initial_audio=preroll if self._vad_warm_on_wake else None,  # Pass pre-roll if warming enabled
                    start_on_wake=self._implicit_start,  # Implicit start if configured
                )
                if not self._running:
                    break
                if utterance is None:
                    logger.info("No speech detected after wake word.")
                    logger.info("🔇 Microphone CLOSED (reason: no speech detected)")
                    self._close_mic_and_reset()
                    continue
                self._pending_audio = utterance
                self._state = GatewayState.STREAM

            elif self._state == GatewayState.STREAM:
                if self._pending_audio is None or self._pending_audio.size == 0:
                    self._close_mic_and_reset()
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
                    logger.info("🔇 Microphone CLOSED (reason: realtime error)")
                    self._pending_audio = None
                    self._close_mic_and_reset()
                    continue

                if response.text:
                    logger.info("Assistant: %s", response.text)
                else:
                    logger.info("Assistant response completed.")

                self._pending_audio = None
                # Check if confusion was signaled - if so, don't arm follow-up
                confusion_signaled = any(
                    result.get("name") == "signal_confusion"
                    for result in (response.tool_results or [])
                )
                if confusion_signaled:
                    logger.info("🔇 Microphone CLOSED (reason: confusion signaled, user must re-invoke)")
                    self._close_mic_and_reset()
                    continue

                should_arm, followup_window, reason = self._should_arm_followup(response.text)
                if should_arm:
                    self._followup_deadline = time.monotonic() + followup_window
                    guard_delay = max(0.0, self._config.follow_up.guard_ms / 1000.0)
                    if guard_delay:
                        time.sleep(guard_delay)
                    self._vad.reset()
                    logger.info(
                        "🎤 Microphone OPEN (reason: follow-up %s, window=%.1fs, guard=%.0fms)",
                        reason,
                        followup_window,
                        self._config.follow_up.guard_ms,
                    )
                    self._state = GatewayState.FOLLOWUP_ARMED
                else:
                    logger.info("🔇 Microphone CLOSED (reason: follow-up disabled)")
                    self._close_mic_and_reset()

            elif self._state == GatewayState.FOLLOWUP_ARMED:
                if self._followup_deadline is None or time.monotonic() > self._followup_deadline:
                    logger.info("Follow-up window expired.")
                    logger.info("🔇 Microphone CLOSED (reason: follow-up timeout)")
                    self._close_mic_and_reset()
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
                    # Use traditional VAD start detection for follow-up
                    reset_vad=True,  # Reset for clean follow-up detection
                    initial_audio=None,
                    start_on_wake=False,
                )
                if not self._running:
                    break
                if utterance is None:
                    logger.info("🔇 Microphone CLOSED (reason: follow-up no speech)")
                    self._close_mic_and_reset()
                    continue
                logger.info("Follow-up speech detected, processing utterance.")
                self._pending_audio = utterance
                self._state = GatewayState.STREAM

            else:
                # Unknown state - reset to clean state
                self._close_mic_and_reset()

        # Flush pending audio when stopping to minimise latency on next start.
        self._audio_input.flush()

    def _should_stop(self) -> bool:
        return not self._running

    def _close_mic_and_reset(self) -> None:
        """Close microphone and reset conversation for next wake word interaction.

        This ensures each wake word session starts with a fresh conversation,
        while follow-up chains maintain context.
        """
        self._audio_input.flush()
        self._vad.reset()
        self._realtime_client.reset_conversation()
        self._state = GatewayState.WAIT_WAKE

    def _should_arm_followup(self, response_text: str) -> tuple[bool, float, str]:
        """Determine if follow-up should be armed and return (should_arm, timeout_seconds, reason).

        Returns:
            (True, 6.0, "question") if response contains ?
            (True, 3.0, "normal") for normal responses when enabled
            (False, 0.0, "disabled") if follow-up is disabled
        """
        follow_up = self._config.follow_up
        if not follow_up.enabled:
            return (False, 0.0, "disabled")

        is_question = "?" in response_text

        if follow_up.arm_mode == "question_only":
            if is_question:
                return (True, follow_up.window_sec, "question")
            else:
                return (False, 0.0, "not a question")
        else:  # "always" mode
            if is_question:
                return (True, follow_up.window_sec, "question")
            else:
                return (True, follow_up.window_normal_sec, "normal")
