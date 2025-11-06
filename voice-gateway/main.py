import logging
import os
import signal
import sys
from typing import Optional

import numpy as np
from dotenv import load_dotenv

from audio.input import AudioInput, AudioInputError
from audio.output import AudioOutput, AudioOutputError
from config import AppConfig, load_config
from utils.single_instance import obtain_lock, SingleInstanceError
from gpt.realtime_client import RealtimeClient
from tools.home_assistant import HomeAssistantClient
from tools.registry import ToolRegistry
from tools.weather import WeatherClient
from vad.silero import VoiceActivityDetector
from wake.vosk import VoskWakeWordDetector
from pipeline import PipelineController


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


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

    # Ensure single-instance so we don't open multiple audio streams accidentally.
    try:
        lock_file = config.paths.log_dir / "voice_gateway.pid"
        app_lock = obtain_lock(lock_file)
        logging.info("Instance lock acquired at %s (pid=%s).", lock_file, str(app_lock.fd))
    except SingleInstanceError as exc:
        logging.error(str(exc))
        sys.exit(1)

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
            logging.debug("Playback: received audio chunk (%d samples)", chunk.size)
            audio_output.write(chunk)
        else:
            logging.debug("Playback: received empty chunk")

    controller = PipelineController(
        config=config,
        audio_input=audio_input,
        wake_detector=wake_detector,
        vad=vad,
        realtime_client=realtime_client,
        playback=playback,
    )

    shutdown_initiated = False

    def _force_exit(reason: str) -> None:
        """Immediately close resources, release lock, and exit the process.

        This is used for Ctrl+Z (SIGTSTP) where the process may be stopped
        before Python can unwind to the main loop's finally block.
        """
        try:
            logging.info("Force shutdown: %s", reason)
        except Exception:
            pass
        try:
            controller.request_shutdown()
        except Exception:
            pass
        try:
            realtime_client.request_shutdown()
        except Exception:
            pass
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
        try:
            app_lock.release()
        except Exception:
            pass
        # Exit without running further cleanup to avoid re-stops
        os._exit(0)

    def _request_shutdown(signum: int, *, reason: Optional[str] = None) -> None:
        nonlocal shutdown_initiated
        if shutdown_initiated:
            return
        shutdown_initiated = True
        message = reason or f"Received signal {signum}; shutting down gracefully."
        logging.info(message)
        try:
            controller.request_shutdown()
        except Exception:
            pass
        try:
            realtime_client.request_shutdown()
        except Exception:
            pass

    def _handle_signal(signum, frame) -> None:
        _request_shutdown(signum)

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, _handle_signal)

    def _handle_stop_signal(signum, frame) -> None:
        # Resume the process group in case it was auto-stopped, then force exit.
        try:
            if hasattr(os, "killpg") and hasattr(signal, "SIGCONT"):
                os.killpg(os.getpgid(os.getpid()), signal.SIGCONT)
        except Exception:
            pass
        _force_exit(f"Received terminal stop signal ({signum})")

    if hasattr(signal, "SIGTSTP"):
        signal.signal(signal.SIGTSTP, _handle_stop_signal)

    try:
        controller.run()
    finally:
        try:
            controller.request_shutdown()
        except Exception:
            pass
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
        try:
            # Release instance lock last.
            app_lock.release()
        except Exception:
            pass
        logging.info("Shutdown complete.")


if __name__ == "__main__":
    main()
