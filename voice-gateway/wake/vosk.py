"""Vosk-based wake word detector."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Iterable, List, Sequence

import numpy as np
import vosk


def _normalize_phrase(phrase: str) -> str:
    return " ".join(phrase.lower().strip().split())


@dataclass
class VoskWakeWordDetector:
    """Streaming wake word detector built on top of Vosk recognizer."""

    model_path: str
    sample_rate: int
    phrases: Sequence[str]
    confidence_threshold: float = 0.6
    debounce_ms: float = 1200.0
    frame_length: int = field(init=False)

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive.")
        if not self.phrases:
            raise ValueError("phrases must not be empty.")
        self._phrases_normalized = [_normalize_phrase(p) for p in self.phrases if p.strip()]
        if not self._phrases_normalized:
            raise ValueError("phrases must contain at least one non-empty value.")

        self._model = vosk.Model(self.model_path)
        grammar = json.dumps(list(self._phrases_normalized))
        self._recognizer = vosk.KaldiRecognizer(self._model, self.sample_rate, grammar)
        self._recognizer.SetWords(True)

        # 80 ms chunks provide a balance between responsiveness and CPU load.
        self.frame_length = max(1, int(self.sample_rate * 0.08))
        self._confidence_threshold = float(self.confidence_threshold)
        self._debounce_sec = max(0.0, float(self.debounce_ms) / 1000.0)
        self._last_triggered: float = 0.0

    def reset(self) -> None:
        """Reset the recognizer state and debounce timer."""
        self._recognizer.Reset()
        self._last_triggered = 0.0

    def close(self) -> None:
        """Release recognizer resources."""
        # The Vosk API does not expose an explicit close; rely on GC.
        pass

    def __enter__(self) -> "VoskWakeWordDetector":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def process(self, pcm16: np.ndarray) -> bool:
        """Process PCM16 audio and return True when the wake word is detected."""
        if pcm16.size == 0:
            return False
        if pcm16.dtype != np.int16:
            pcm16 = pcm16.astype(np.int16)

        now = time.monotonic()
        if self._debounce_sec and now - self._last_triggered < self._debounce_sec:
            # Within debounce window; consume audio but don't trigger again.
            self._recognizer.AcceptWaveform(pcm16.tobytes())
            return False

        triggered = False
        if self._recognizer.AcceptWaveform(pcm16.tobytes()):
            triggered = self._handle_result(self._recognizer.Result())
        else:
            triggered = self._handle_partial(self._recognizer.PartialResult())

        if triggered:
            self._last_triggered = time.monotonic()
            self._recognizer.Reset()
        return triggered

    def _handle_result(self, result_json: str) -> bool:
        if not result_json:
            return False
        try:
            payload = json.loads(result_json)
        except json.JSONDecodeError:
            return False
        text = _normalize_phrase(payload.get("text", ""))
        if not text:
            return False
        if text not in self._phrases_normalized:
            return False
        confidence = self._estimate_confidence(payload.get("result", []))
        return confidence >= self._confidence_threshold

    def _handle_partial(self, partial_json: str) -> bool:
        if not partial_json:
            return False
        try:
            payload = json.loads(partial_json)
        except json.JSONDecodeError:
            return False
        text = _normalize_phrase(payload.get("partial", ""))
        if not text or text not in self._phrases_normalized:
            return False
        # Partial results do not provide confidence values. Treat them as tentative
        # detections only when no explicit confidence threshold is requested.
        if self._confidence_threshold <= 0.0:
            return True
        return False

    @staticmethod
    def _estimate_confidence(tokens: Iterable[dict]) -> float:
        confidences: List[float] = []
        for token in tokens or []:
            try:
                conf = float(token.get("conf", 0.0))
            except (TypeError, ValueError):
                conf = 0.0
            if conf > 0.0:
                confidences.append(conf)
        if not confidences:
            return 0.0
        return float(sum(confidences) / len(confidences))
