"""Silero voice activity detector wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class VADProcessResult:
    probability: float
    speech_started: bool
    speech_ended: bool
    forced: bool
    speech_active: bool
    speech_duration_ms: float
    silence_duration_ms: float


class VoiceActivityDetector:
    """Streaming endpoint detector built on top of the Silero VAD model."""

    def __init__(
        self,
        model_path: str | Path,
        threshold: float = 0.5,
        min_speech_ms: int = 400,
        min_silence_ms: int = 600,
        max_segment_ms: int = 7000,
        pad_ms: int = 120,
        sample_rate: int = 16000,
    ) -> None:
        if sample_rate not in (8000, 16000):
            raise ValueError("Silero VAD supports 8 kHz or 16 kHz audio.")

        model_path_obj = Path(model_path)
        if not model_path_obj.exists():
            raise FileNotFoundError(f"Silero VAD model file not found: {model_path_obj}")
        torch.set_num_threads(1)
        try:
            self._model = torch.jit.load(str(model_path_obj), map_location="cpu")
        except Exception as exc:
            raise RuntimeError(f"Failed to load Silero VAD model from {model_path_obj}: {exc}") from exc
        self._model.eval()
        self._threshold = float(threshold)
        self._neg_threshold = max(self._threshold - 0.15, 0.01)
        self._min_speech_samples = int(sample_rate * min_speech_ms / 1000)
        self._min_silence_samples = int(sample_rate * min_silence_ms / 1000)
        self._max_segment_samples = int(sample_rate * max_segment_ms / 1000)
        self._pad_samples = int(sample_rate * pad_ms / 1000)
        self._sample_rate = sample_rate

        self._window_samples = 512 if sample_rate == 16000 else 256
        self._buffer = np.zeros((0,), dtype=np.int16)

        self._candidate_speech = 0
        self._speech_active = False
        self._samples_since_start = 0
        self._silence_samples = 0
        self._total_samples = 0
        self._last_probability = 0.0

        self.reset()

    def reset(self) -> None:
        self._model.reset_states()
        self._buffer = np.zeros((0,), dtype=np.int16)
        self._candidate_speech = 0
        self._speech_active = False
        self._samples_since_start = 0
        self._silence_samples = 0
        self._total_samples = 0
        self._last_probability = 0.0

    @property
    def speech_active(self) -> bool:
        return self._speech_active

    def process(self, pcm16: np.ndarray) -> VADProcessResult:
        """Consume audio and update endpointing state."""
        if pcm16.size == 0:
            return VADProcessResult(
                probability=self._last_probability,
                speech_started=False,
                speech_ended=False,
                forced=False,
                speech_active=self._speech_active,
                speech_duration_ms=self._samples_since_start * 1000 / self._sample_rate,
                silence_duration_ms=self._silence_samples * 1000 / self._sample_rate,
            )

        if pcm16.dtype != np.int16:
            pcm16 = pcm16.astype(np.int16)

        if self._buffer.size:
            pcm16 = np.concatenate((self._buffer, pcm16))

        usable = (pcm16.size // self._window_samples) * self._window_samples
        frames = pcm16[:usable].reshape(-1, self._window_samples)
        self._buffer = pcm16[usable:]

        speech_started = False
        speech_ended = False
        forced_end = False
        duration_samples = self._samples_since_start
        silence_samples = self._silence_samples

        for idx, frame in enumerate(frames):
            audio = torch.from_numpy(frame.astype(np.float32) / 32768.0)
            with torch.no_grad():
                prob = float(self._model(audio, self._sample_rate).item())
            self._last_probability = prob
            self._total_samples += self._window_samples

            if not self._speech_active:
                if prob >= self._threshold:
                    self._candidate_speech += self._window_samples
                else:
                    self._candidate_speech = max(0, self._candidate_speech - self._window_samples)

                if self._candidate_speech >= self._min_speech_samples:
                    self._speech_active = True
                    speech_started = True
                    self._samples_since_start = self._candidate_speech
                    duration_samples = self._samples_since_start
                    self._silence_samples = 0
                    silence_samples = 0
                continue

            self._samples_since_start += self._window_samples
            duration_samples = self._samples_since_start

            if prob >= self._threshold:
                self._silence_samples = 0
            elif prob < self._neg_threshold:
                self._silence_samples += self._window_samples
            else:
                self._silence_samples += self._window_samples // 2
            silence_samples = self._silence_samples

            if self._samples_since_start >= self._max_segment_samples:
                forced_end = True
                speech_ended = True
            elif self._silence_samples >= self._min_silence_samples + self._pad_samples:
                speech_ended = True

            if speech_ended:
                duration_samples = self._samples_since_start
                silence_samples = self._silence_samples
                self._speech_active = False
                self._candidate_speech = 0
                self._samples_since_start = 0
                self._silence_samples = 0

                if idx + 1 < len(frames):
                    remainder = frames[idx + 1 :].reshape(-1)
                    if remainder.size:
                        self._buffer = np.concatenate((remainder, self._buffer))
                break

        return VADProcessResult(
            probability=self._last_probability,
            speech_started=speech_started,
            speech_ended=speech_ended,
            forced=forced_end,
            speech_active=self._speech_active,
            speech_duration_ms=duration_samples * 1000 / self._sample_rate,
            silence_duration_ms=silence_samples * 1000 / self._sample_rate,
        )
