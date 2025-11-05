import logging
import signal
import sys
import time
from enum import Enum, auto
from typing import Callable, List, Optional

import numpy as np
from dotenv import load_dotenv

from audio.input import AudioInput, AudioInputError
from audio.output import AudioOutput, AudioOutputError
from config import AppConfig, load_config
from gpt.realtime_client import RealtimeClient, RealtimeError
from tools.home_assistant import HomeAssistantClient
from tools.registry import ToolRegistry
from tools.weather import WeatherClient
from vad.silero import VoiceActivityDetector
from wake.vosk import VoskWakeWordDetector


class GatewayState(Enum):
    WAIT_WAKE = auto()
    CAPTURE = auto()
    STREAM = auto()
    FOLLOWUP_ARMED = auto()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def rms_level(chunk: np.ndarray) -> float:
    if chunk.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2))) / 32768.0


def peak_level(chunk: np.ndarray) -> int:
    if chunk.size == 0:
        return 0
    return int(np.max(np.abs(chunk)))


def record_utterance(
    vad: VoiceActivityDetector,
    reader: Callable[[int], np.ndarray],
    block_size: int,
    listen_timeout: Optional[float],
    max_capture_sec: float,
    min_peak: Optional[int] = None,
    min_rms: Optional[float] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> Optional[np.ndarray]:
    vad.reset()
    captured: List[np.ndarray] = []
    speech_detected = False
    start_time = time.monotonic()
    listen_deadline = start_time + listen_timeout if listen_timeout is not None else None
    end_deadline = start_time + max_capture_sec

    while time.monotonic() < end_deadline:
        if should_stop is not None and should_stop():
            logging.debug("record_utterance: stop requested externally.")
            return None
        chunk = reader(block_size)
        if chunk.size == 0:
            if should_stop is not None and should_stop():
                logging.debug("record_utterance: received empty chunk and stop requested.")
                return None
            continue
        captured.append(chunk)
        result = vad.process(chunk)
        if result.speech_started:
            logging.info(
                "VAD speech started (prob=%.2f, rms=%.4f, peak=%d).",
                result.probability,
                rms_level(chunk),
                peak_level(chunk),
            )
        if result.speech_ended or result.forced:
            logging.info(
                "VAD speech ended (duration_ms=%.0f, silence_ms=%.0f, forced=%s).",
                result.speech_duration_ms,
                result.silence_duration_ms,
                result.forced,
            )

        if not speech_detected:
            if listen_deadline is not None and time.monotonic() > listen_deadline:
                logging.info("VAD listen timeout reached before speech start.")
                return None
            if result.speech_started:
                meets_peak = min_peak is None or peak_level(chunk) >= min_peak
                meets_rms = min_rms is None or rms_level(chunk) >= min_rms
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
        logging.info("record_utterance: no speech detected.")
        return None
    if not captured:
        logging.debug("record_utterance: speech detected but no audio captured.")
        return None
    return np.concatenate(captured)


def should_arm_followup(response_text: str, config: AppConfig) -> bool:
    follow_up = config.follow_up
    if not follow_up.enabled:
        return False
    if follow_up.arm_mode == "always":
        return True
    return response_text.strip().endswith("?")


def main() -> None:
    load_dotenv()
    setup_logging()

    logging.info("Loading configuration...")
    try:
        config = load_config()
    except Exception as exc:
        logging.error("Failed to load configuration: %s", exc)
        sys.exit(1)
    logging.info(
        "Configuration loaded. wake_model=%s wake_phrases=%s input_device=%s output_device=%s",
        config.wake.model_path,
        config.wake.phrases,
        config.audio_input.device_name or config.audio_input.device_index,
        config.audio_output.device_name or config.audio_output.device_index,
    )

    logging.info("Initialising wake detector...")
    wake_detector = VoskWakeWordDetector(
        model_path=str(config.wake.model_path),
        sample_rate=config.audio_input.sample_rate,
        phrases=config.wake.phrases,
        confidence_threshold=config.wake.confidence_threshold,
        debounce_ms=config.wake.debounce_ms,
    )
    logging.info("Wake detector ready (frame_length=%d).", wake_detector.frame_length)

    logging.info(
        "Initialising audio input (sample_rate=%d, block_size=%d, device=%s).",
        config.audio_input.sample_rate,
        wake_detector.frame_length,
        config.audio_input.device_name or config.audio_input.device_index,
    )
    try:
        audio_input = AudioInput(
            sample_rate=config.audio_input.sample_rate,
            gain=config.audio_input.gain,
            device_name=config.audio_input.device_name,
            device_index=config.audio_input.device_index,
            block_size=wake_detector.frame_length,
        )
    except AudioInputError as exc:
        logging.error("Audio input failure: %s", exc)
        sys.exit(1)
    logging.info("Audio input ready.")

    logging.info(
        "Initialising audio output (sample_rate=%d, device=%s).",
        config.audio_output.sample_rate,
        config.audio_output.device_name or config.audio_output.device_index,
    )
    try:
        audio_output = AudioOutput(
            sample_rate=config.audio_output.sample_rate,
            device_name=config.audio_output.device_name,
            device_index=config.audio_output.device_index,
        )
    except AudioOutputError as exc:
        logging.error("Audio output failure: %s", exc)
        sys.exit(1)
    logging.info("Audio output ready (actual_sample_rate=%d).", audio_output.sample_rate)

    logging.info("Loading voice activity detector...")
    vad = VoiceActivityDetector(
        model_path=config.vad.model_path,
        threshold=config.vad.speech_prob_threshold,
        min_speech_ms=config.vad.min_speech_ms,
        min_silence_ms=config.vad.min_silence_ms,
        max_segment_ms=config.vad.max_segment_ms,
        pad_ms=config.vad.pad_ms,
        sample_rate=config.audio_input.sample_rate,
    )
    logging.info("Voice activity detector ready.")

    ha_client = HomeAssistantClient(
        base_url=config.home_assistant.url,
        token=config.home_assistant.token,
        timeout=config.home_assistant.timeout,
    )
    weather_client = WeatherClient(
        base_url=config.weather.base_url,
        default_location=config.weather.default_location,
        timeout=config.weather.timeout,
    )
    tool_registry = ToolRegistry(ha_client=ha_client, weather_client=weather_client)

    logging.info("Initialising realtime client (%s).", config.realtime.model)
    realtime_client = RealtimeClient(config=config.realtime, tool_registry=tool_registry)
    logging.info("Realtime client initialised.")

    def playback(chunk: np.ndarray) -> None:
        if chunk.size:
            audio_output.write(chunk)

    running = True

    def _handle_signal(signum, frame) -> None:
        nonlocal running
        if running:
            logging.info("Received signal %s; shutting down gracefully.", signum)
        # Only flip the flag here; do not perform blocking work in a signal handler.
        # Cleanup runs in the 'finally' block below.
        running = False

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    block_size = wake_detector.frame_length
    max_capture_sec = max(config.vad.max_segment_ms / 1000.0, 1.0)

    state = GatewayState.WAIT_WAKE
    pending_audio: Optional[np.ndarray] = None
    followup_deadline: Optional[float] = None

    logging.info("Voice gateway ready. Say the wake word to begin.")

    try:
        while running:
            if state == GatewayState.WAIT_WAKE:
                chunk = audio_input.read(block_size)
                if wake_detector.process(chunk):
                    logging.info("Wake word detected.")
                    state = GatewayState.CAPTURE
                    vad.reset()
                    audio_input.flush()
                    continue
            elif state == GatewayState.CAPTURE:
                utterance = record_utterance(
                    vad=vad,
                    reader=lambda frames: audio_input.read(frames),
                    block_size=block_size,
                    listen_timeout=None,
                    max_capture_sec=max_capture_sec,
                    should_stop=lambda: not running,
                )
                if utterance is None:
                    logging.info("No speech detected after wake word.")
                    audio_input.flush()
                    state = GatewayState.WAIT_WAKE
                    continue
                pending_audio = utterance
                state = GatewayState.STREAM
            elif state == GatewayState.STREAM:
                if pending_audio is None or pending_audio.size == 0:
                    state = GatewayState.WAIT_WAKE
                    continue
                logging.info("Streaming utterance to realtime API (%d samples).", pending_audio.size)
                try:
                    response = realtime_client.send_utterance(
                        pcm16_audio=pending_audio,
                        source_sample_rate=config.audio_input.sample_rate,
                        audio_consumer=playback,
                    )
                except RealtimeError as exc:
                    logging.error("Realtime processing failed: %s", exc)
                    audio_input.flush()
                    state = GatewayState.WAIT_WAKE
                    continue

                if response.text:
                    logging.info("Assistant: %s", response.text)
                else:
                    logging.info("Assistant response completed.")

                pending_audio = None
                followup_required = should_arm_followup(response.text, config)
                if followup_required:
                    followup_window = config.follow_up.window_sec
                    followup_deadline = time.monotonic() + followup_window
                    guard_delay = max(0.0, config.follow_up.guard_ms / 1000.0)
                    if guard_delay:
                        time.sleep(guard_delay)
                    vad.reset()
                    logging.info(
                        "Follow-up window armed for %.1fs (guard=%.0f ms).",
                        followup_window,
                        config.follow_up.guard_ms,
                    )
                    state = GatewayState.FOLLOWUP_ARMED
                else:
                    audio_input.flush()
                    state = GatewayState.WAIT_WAKE
            elif state == GatewayState.FOLLOWUP_ARMED:
                if followup_deadline is None or time.monotonic() > followup_deadline:
                    logging.info("Follow-up window expired.")
                    audio_input.flush()
                    state = GatewayState.WAIT_WAKE
                    continue

                remaining = followup_deadline - time.monotonic()
                utterance = record_utterance(
                    vad=vad,
                    reader=lambda frames: audio_input.read(frames),
                    block_size=block_size,
                    listen_timeout=remaining,
                    max_capture_sec=max_capture_sec,
                    min_peak=config.follow_up.min_peak,
                    min_rms=config.follow_up.min_rms,
                    should_stop=lambda: not running,
                )
                if utterance is None:
                    audio_input.flush()
                    state = GatewayState.WAIT_WAKE
                    continue
                pending_audio = utterance
                state = GatewayState.STREAM
            else:
                state = GatewayState.WAIT_WAKE
    finally:
        try:
            realtime_client.close()
        except Exception:
            pass
        try:
            wake_detector.close()
        except Exception:
            pass
        try:
            audio_output.close()
        except Exception:
            pass
        try:
            audio_input.close()
        except Exception:
            pass
        logging.info("Shutdown complete.")


if __name__ == "__main__":
    main()
