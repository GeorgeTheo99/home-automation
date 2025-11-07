"""Environment configuration for the voice gateway."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


def _get_env(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.getenv(name, None)
    if value is None:
        return default
    stripped = value.strip()
    if stripped == "":
        return default
    return stripped


def _get_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _get_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    try:
        return int(float(value))
    except ValueError as exc:
        raise ValueError(f"Environment value for {name!r} must be an integer") from exc


def _get_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"Environment value for {name!r} must be a float") from exc


def _get_optional_float(name: str) -> Optional[float]:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return None
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"Environment value for {name!r} must be a float") from exc


def _get_optional_int(name: str) -> Optional[int]:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return None
    lowered = value.strip().lower()
    if lowered in {"none", "null", "default", "auto"}:
        return None
    try:
        return int(float(value))
    except ValueError as exc:
        raise ValueError(f"Environment value for {name!r} must be an integer") from exc


@dataclass(frozen=True)
class AudioInputConfig:
    sample_rate: int
    device_name: Optional[str]
    device_index: Optional[int]
    gain: float


@dataclass(frozen=True)
class AudioOutputConfig:
    sample_rate: int
    device_name: Optional[str]
    device_index: Optional[int]


@dataclass(frozen=True)
class FollowUpConfig:
    enabled: bool
    arm_mode: str
    window_sec: float
    window_normal_sec: float
    guard_ms: float
    min_rms: Optional[float]
    min_peak: Optional[int]


@dataclass(frozen=True)
class WakeConfig:
    model_path_raw: str
    model_path: Path
    phrases: List[str]
    confidence_threshold: float
    debounce_ms: float


@dataclass(frozen=True)
class VADConfig:
    speech_prob_threshold: float
    min_speech_ms: int
    min_silence_ms: int
    max_segment_ms: int
    window_ms: int
    pad_ms: int
    model_path_raw: str
    model_path: Path


@dataclass(frozen=True)
class RealtimeConfig:
    api_key: str
    model: str
    modalities: List[str]
    temperature: float
    # Maximum seconds to wait for initial websocket connect
    connect_timeout: float
    session_timeout: float
    instructions_prompt: str


@dataclass(frozen=True)
class HomeAssistantConfig:
    url: Optional[str]
    token: Optional[str]
    timeout: float


@dataclass(frozen=True)
class WeatherConfig:
    base_url: str
    default_location: Optional[str]
    timeout: float


@dataclass(frozen=True)
class AppPaths:
    log_dir: Path


@dataclass(frozen=True)
class AppConfig:
    realtime: RealtimeConfig
    audio_input: AudioInputConfig
    audio_output: AudioOutputConfig
    follow_up: FollowUpConfig
    wake: WakeConfig
    vad: VADConfig
    home_assistant: HomeAssistantConfig
    weather: WeatherConfig
    paths: AppPaths

    def validate(self) -> None:
        if not self.realtime.api_key:
            raise ValueError("OPENAI_API_KEY must be set.")
        if not self.wake.model_path_raw:
            raise ValueError("VOSK_MODEL_PATH must be set.")
        if not self.wake.model_path.exists():
            raise FileNotFoundError(f"VOSK_MODEL_PATH does not exist: {self.wake.model_path}")
        if not self.wake.model_path.is_dir():
            raise NotADirectoryError(f"VOSK_MODEL_PATH must point to a directory: {self.wake.model_path}")
        if not self.wake.phrases:
            raise ValueError("WAKEWORD_PHRASES must include at least one phrase.")
        if not 0.0 <= self.wake.confidence_threshold <= 1.0:
            raise ValueError("WAKE_CONFIDENCE_THRESHOLD must be between 0.0 and 1.0.")
        if self.wake.debounce_ms < 0:
            raise ValueError("WAKE_DEBOUNCE_MS must be non-negative.")
        if self.follow_up.arm_mode not in {"question_only", "always"}:
            raise ValueError("FOLLOWUP_ARM_MODE must be 'question_only' or 'always'.")
        if self.follow_up.min_rms is not None and not 0.0 <= self.follow_up.min_rms <= 1.0:
            raise ValueError("FOLLOWUP_MIN_RMS must be between 0.0 and 1.0.")
        if not self.vad.model_path_raw:
            raise ValueError("VAD_MODEL_PATH must be provided (point to silero_vad.jit).")
        if not self.vad.model_path.exists():
            raise FileNotFoundError(f"VAD_MODEL_PATH does not exist: {self.vad.model_path}")
        if not self.vad.model_path.is_file():
            raise FileNotFoundError(f"VAD_MODEL_PATH must point to a file: {self.vad.model_path}")


INSTRUCTIONS_PROMPT = (
    "You are a helpful, concise home assistant. Always respond in English. Speak succinctly. "
    "Use tools when appropriate to control the home or fetch weather. "
    "If you don't understand the user's input or cannot help, call the signal_confusion tool "
    "immediately instead of speaking - do not provide a verbal response. "
    "If you need more information, ask a direct question. "
    "The system will allow a brief follow-up window after your response; otherwise the user must "
    "say the wake word again."
)


def load_config() -> AppConfig:
    realtime_modalities = [
        modality.strip()
        for modality in os.getenv("REALTIME_MODALITIES", "audio,text").split(",")
        if modality.strip()
    ]
    if not realtime_modalities:
        realtime_modalities = ["audio"]

    wake_model_raw = _get_env("VOSK_MODEL_PATH", "models/vosk-model-en-us-0.22-lgraph") or "models/vosk-model-en-us-0.22-lgraph"
    wake_phrases_raw = _get_env("WAKEWORD_PHRASES", "hey computer") or "hey computer"
    vad_model_raw = _get_env("VAD_MODEL_PATH", "models/silero_vad.jit") or "models/silero_vad.jit"

    config = AppConfig(
        realtime=RealtimeConfig(
            api_key=_get_env("OPENAI_API_KEY", "") or "",
            # Default to pinned latest Realtime model (verified via /v1/models)
            model=_get_env("REALTIME_MODEL", "gpt-realtime-2025-08-28") or "gpt-realtime-2025-08-28",
            modalities=realtime_modalities,
            temperature=_get_float("REALTIME_TEMPERATURE", 0.6),
            connect_timeout=_get_float("REALTIME_CONNECT_TIMEOUT", 8.0),
            session_timeout=_get_float("REALTIME_SESSION_TIMEOUT", 12.0),
            instructions_prompt=_get_env("REALTIME_INSTRUCTIONS_PROMPT", INSTRUCTIONS_PROMPT) or INSTRUCTIONS_PROMPT,
        ),
        audio_input=AudioInputConfig(
            sample_rate=_get_int("INPUT_SAMPLE_RATE", 16000),
            device_name=_get_env("MIC_DEVICE_NAME"),
            device_index=_get_optional_int("MIC_DEVICE_INDEX"),
            gain=_get_float("INPUT_GAIN", 1.0),
        ),
        audio_output=AudioOutputConfig(
            sample_rate=_get_int("OUTPUT_SAMPLE_RATE", 24000),
            device_name=_get_env("SPEAKER_DEVICE_NAME"),
            device_index=_get_optional_int("SPEAKER_DEVICE_INDEX"),
        ),
        follow_up=FollowUpConfig(
            enabled=_get_bool("FOLLOWUP_ENABLED", True),
            arm_mode=_get_env("FOLLOWUP_ARM_MODE", "always") or "always",
            window_sec=_get_float("FOLLOWUP_WINDOW_SEC", 6.0),
            window_normal_sec=_get_float("FOLLOWUP_WINDOW_NORMAL_SEC", 3.0),
            guard_ms=_get_float("FOLLOWUP_GUARD_MS", 400.0),
            min_rms=_get_optional_float("FOLLOWUP_MIN_RMS"),
            min_peak=_get_optional_int("FOLLOWUP_MIN_PEAK"),
        ),
        wake=WakeConfig(
            model_path_raw=wake_model_raw,
            model_path=Path(wake_model_raw).expanduser(),
            phrases=[
                phrase.strip().lower()
                for phrase in wake_phrases_raw.split(",")
                if phrase.strip()
            ],
            confidence_threshold=_get_float("WAKE_CONFIDENCE_THRESHOLD", 0.6),
            debounce_ms=_get_float("WAKE_DEBOUNCE_MS", 1200.0),
        ),
        vad=VADConfig(
            speech_prob_threshold=_get_float("VAD_SPEECH_PROB_THRESHOLD", 0.5),
            min_speech_ms=_get_int("VAD_MIN_SPEECH_MS", 400),
            min_silence_ms=_get_int("VAD_MIN_SILENCE_MS", 600),
            max_segment_ms=_get_int("VAD_MAX_SEGMENT_MS", 7000),
            window_ms=_get_int("VAD_WINDOW_MS", 30),
            pad_ms=_get_int("VAD_PAD_MS", 120),
            model_path_raw=vad_model_raw,
            model_path=Path(vad_model_raw).expanduser(),
        ),
        home_assistant=HomeAssistantConfig(
            url=_get_env("HA_URL"),
            token=_get_env("HA_TOKEN"),
            timeout=_get_float("HA_TIMEOUT", 6.0),
        ),
        weather=WeatherConfig(
            base_url=_get_env("WEATHER_API_URL", "https://wttr.in") or "https://wttr.in",
            default_location=_get_env("WEATHER_DEFAULT_LOCATION"),
            timeout=_get_float("WEATHER_TIMEOUT", 6.0),
        ),
        paths=AppPaths(
            log_dir=Path(_get_env("LOG_DIR", "logs")).expanduser(),
        ),
    )
    config.validate()
    return config
