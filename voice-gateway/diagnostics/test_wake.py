#!/usr/bin/env python3
"""Wake word diagnostic tool.

This utility runs only the wake-word step using the same Vosk grammar
and confidence logic as the gateway. It supports microphone input or
an optional WAV file for offline testing.

Examples:
  # Mic test with defaults from .env or environment
  python diagnostics/test_wake.py

  # Override threshold and phrases
  WAKEWORD_PHRASES="hey computer,ok computer" \
  python diagnostics/test_wake.py --threshold 0.55 --debounce-ms 800

  # Offline WAV test
  python diagnostics/test_wake.py --wav samples/hey_computer.wav --phrases "hey computer"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from dotenv import load_dotenv

try:
    import vosk  # type: ignore
except Exception as exc:  # pragma: no cover - diagnostic environment
    print(f"vosk import failed: {exc}", file=sys.stderr)
    sys.exit(2)

try:
    import sounddevice as sd  # type: ignore
except Exception:
    sd = None  # Allow WAV-only runs without sounddevice


def _normalize_phrase(phrase: str) -> str:
    return " ".join(phrase.lower().strip().split())


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


@dataclass
class WakeHarness:
    model_path: str
    sample_rate: int
    phrases: Sequence[str]
    confidence_threshold: float = 0.6
    debounce_ms: float = 1200.0
    chunk_ms: Optional[float] = None

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if not self.phrases:
            raise ValueError("phrases must not be empty")
        self._phrases_normalized = [_normalize_phrase(p) for p in self.phrases if p.strip()]
        if not self._phrases_normalized:
            raise ValueError("phrases must include a non-empty value")

        self._model = vosk.Model(self.model_path)
        grammar = json.dumps(list(self._phrases_normalized))
        self._recognizer = vosk.KaldiRecognizer(self._model, self.sample_rate, grammar)
        self._recognizer.SetWords(True)

        base_chunk = float(self.chunk_ms) / 1000.0 if self.chunk_ms is not None else 0.08
        self.frame_length = max(1, int(self.sample_rate * base_chunk))
        self._threshold = float(self.confidence_threshold)
        self._debounce_sec = max(0.0, float(self.debounce_ms) / 1000.0)
        self._last_triggered: float = 0.0

    def process(self, pcm16: np.ndarray, *, verbose: bool = False) -> Tuple[bool, Optional[float]]:
        """Feed audio into recognizer. Returns (triggered, confidence or None)."""
        if pcm16.size == 0:
            return False, None
        if pcm16.dtype != np.int16:
            pcm16 = pcm16.astype(np.int16)

        now = time.monotonic()
        if self._debounce_sec and now - self._last_triggered < self._debounce_sec:
            # Consume audio but ignore triggers during debounce.
            self._recognizer.AcceptWaveform(pcm16.tobytes())
            if verbose:
                print(f"[debounce active for {self._debounce_sec:.2f}s]")
            return False, None

        triggered = False
        confidence: Optional[float] = None
        if self._recognizer.AcceptWaveform(pcm16.tobytes()):
            result_raw = self._recognizer.Result()
            try:
                payload = json.loads(result_raw or "{}")
            except json.JSONDecodeError:
                payload = {}
            text = _normalize_phrase(payload.get("text", ""))
            tokens = payload.get("result", []) or []
            confidence = _estimate_confidence(tokens)

            matched = bool(text) and text in self._phrases_normalized
            triggered = matched and (self._threshold <= 0.0 or confidence >= self._threshold)

            # Pretty print the full result
            token_s = ", ".join(
                f"{t.get('word','?')}:{float(t.get('conf',0.0)):.2f}" for t in tokens
            )
            print(
                f"[RESULT] text='{text}' matched={matched} conf={confidence:.3f} "
                f"threshold={self._threshold:.3f} tokens=[{token_s}]"
            )
        else:
            partial_raw = self._recognizer.PartialResult()
            if verbose:
                try:
                    p = json.loads(partial_raw or "{}")
                except json.JSONDecodeError:
                    p = {}
                partial_text = _normalize_phrase(p.get("partial", ""))
                if partial_text:
                    print(f"[PARTIAL] '{partial_text}'")

        if triggered:
            self._last_triggered = time.monotonic()
            self._recognizer.Reset()
        return triggered, confidence


def list_input_devices() -> None:
    if sd is None:
        print("sounddevice is not available; cannot list input devices", file=sys.stderr)
        return
    try:
        default = sd.default.device
    except Exception:
        default = None
    print("Index  Name                                   In  Out  Default")
    print("-----  -------------------------------------  --  ---  -------")
    for idx, dev in enumerate(sd.query_devices()):
        mark = ""
        if isinstance(default, (tuple, list)) and len(default) >= 1:
            in_idx = default[0]
            if idx == in_idx:
                mark = "<in>"
        elif default is not None and idx == default:
            mark = "<def>"
        name = dev.get("name", "") or ""
        print(
            f"{idx:5d}  {name[:37]:37s}  {dev.get('max_input_channels', 0):2d}  {dev.get('max_output_channels', 0):3d}  {mark}"
        )


def _read_wav_chunks(path: Path, chunk_frames: int):
    """Yield (pcm16_mono_chunk, samplerate) for the WAV file."""
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        if sampwidth != 2:
            raise ValueError("WAV must be 16-bit PCM")
        if channels not in (1, 2):
            raise ValueError("WAV must be mono or stereo")

        total_frames = wf.getnframes()
        frames_read = 0
        while frames_read < total_frames:
            n = min(chunk_frames, total_frames - frames_read)
            data = wf.readframes(n)
            frames_read += n
            if not data:
                break
            pcm = np.frombuffer(data, dtype=np.int16)
            if channels == 2:
                # Downmix stereo to mono
                pcm = pcm.reshape(-1, 2).mean(axis=1).astype(np.int16)
            yield pcm, framerate


def _resolve_model_path(raw_path: str) -> Tuple[Path, bool]:
    """Resolve a model directory, trying CWD and project root for relatives."""
    path = Path(raw_path).expanduser()
    candidates: List[Path] = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.append(Path.cwd() / path)
        repo_root = Path(__file__).resolve().parents[1]
        candidates.append(repo_root / path)

    seen: set[str] = set()
    filtered: List[Path] = []
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        filtered.append(candidate)

    for candidate in filtered:
        if candidate.exists():
            return candidate.resolve(), True

    fallback = filtered[0].resolve()
    return fallback, False


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Test the wake word detector with detailed prints.")
    parser.add_argument("--model-path", type=str, default=None, help="Path to Vosk model directory (VOSK_MODEL_PATH)")
    parser.add_argument("--phrases", type=str, default=None, help="Comma-separated wake phrases (WAKEWORD_PHRASES)")
    parser.add_argument("--threshold", type=float, default=None, help="Confidence threshold (WAKE_CONFIDENCE_THRESHOLD)")
    parser.add_argument("--debounce-ms", type=float, default=None, help="Debounce window in ms (WAKE_DEBOUNCE_MS)")
    parser.add_argument("--rate", type=int, default=None, help="Sample rate in Hz (INPUT_SAMPLE_RATE)")
    parser.add_argument("--chunk-ms", type=float, default=None, help="Chunk size in ms (default 80)")
    parser.add_argument("--device-index", type=int, help="Input device index to use")
    parser.add_argument("--device-name", type=str, help="Substring match for input device name")
    parser.add_argument("--gain", type=float, default=None, help="Linear capture gain 0.0-? (INPUT_GAIN)")
    parser.add_argument("--wav", type=str, help="Optional WAV file path to process instead of mic")
    parser.add_argument("--list", action="store_true", help="List input devices and exit")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print partial transcripts")
    parser.add_argument("--stop-on-detect", action="store_true", help="Exit after first trigger")
    args = parser.parse_args()

    if args.list:
        list_input_devices()
        return

    # Gather defaults from environment
    model_path_raw = args.model_path or (os.getenv("VOSK_MODEL_PATH") or "models/vosk-model-small-en-us-0.15")
    phrases_raw = args.phrases or (os.getenv("WAKEWORD_PHRASES") or "hey computer")
    threshold = args.threshold if args.threshold is not None else float(os.getenv("WAKE_CONFIDENCE_THRESHOLD", 0.6))
    debounce_ms = args.debounce_ms if args.debounce_ms is not None else float(os.getenv("WAKE_DEBOUNCE_MS", 1200.0))
    sample_rate = args.rate if args.rate is not None else int(float(os.getenv("INPUT_SAMPLE_RATE", 16000)))
    gain = args.gain if args.gain is not None else float(os.getenv("INPUT_GAIN", 1.0))

    resolved_model_path, model_exists = _resolve_model_path(model_path_raw)
    if not model_exists:
        print(f"Model directory not found: {resolved_model_path}", file=sys.stderr)
        print("Set VOSK_MODEL_PATH or pass --model-path to the extracted Vosk model directory.", file=sys.stderr)
        sys.exit(2)
    model_path = str(resolved_model_path)

    phrases = [p.strip().lower() for p in phrases_raw.split(",") if p.strip()]
    norm_phrases = [
        _normalize_phrase(p) for p in phrases
    ]

    print("Wake Word Diagnostic")
    print("---------------------")
    print(f"model_path     : {model_path}")
    print(f"sample_rate    : {sample_rate}")
    print(f"phrases        : {phrases}")
    print(f"normalized     : {norm_phrases}")
    print(f"threshold      : {threshold}")
    print(f"debounce_ms    : {debounce_ms}")
    print(f"chunk_ms       : {args.chunk_ms if args.chunk_ms is not None else 80}")
    if args.wav:
        print(f"wav            : {args.wav}")

    harness = WakeHarness(
        model_path=model_path,
        sample_rate=sample_rate,
        phrases=phrases,
        confidence_threshold=threshold,
        debounce_ms=debounce_ms,
        chunk_ms=args.chunk_ms,
    )

    # WAV path: offline processing.
    if args.wav:
        wav_path = Path(args.wav)
        if not wav_path.exists():
            print(f"WAV file not found: {wav_path}", file=sys.stderr)
            sys.exit(2)

        total_frames = 0
        for chunk, rate in _read_wav_chunks(wav_path, harness.frame_length):
            if rate != sample_rate:
                print(
                    f"[warn] WAV rate {rate} != requested {sample_rate}; results may be unreliable",
                    file=sys.stderr,
                )
            total_frames += int(chunk.size)
            if gain != 1.0 and chunk.size:
                amplified = np.clip(chunk.astype(np.float32) * gain, -32768, 32767)
                chunk = amplified.astype(np.int16)
            triggered, conf = harness.process(chunk, verbose=args.verbose)
            if triggered:
                t_sec = total_frames / float(sample_rate)
                print(f"[DETECTED] t={t_sec:.3f}s conf={conf if conf is not None else 'n/a'}")
                if args.stop_on_detect:
                    break
        return

    # Mic path
    if sd is None:
        print("sounddevice unavailable; install it or provide --wav", file=sys.stderr)
        sys.exit(2)

    # Use sounddevice directly (avoids requiring the full AudioInput stack here)
    try:
        sd.check_input_settings(
            device=args.device_index or args.device_name,
            samplerate=sample_rate,
            channels=1,
            dtype="int16",
        )
    except Exception as exc:
        print(f"Input device rejected settings: {exc}", file=sys.stderr)
        sys.exit(2)

    q: List[bytes] = []
    pending: bytes = b""

    def _callback(indata, frames, time_info, status):  # type: ignore[no-redef]
        nonlocal q
        try:
            q.append(indata.tobytes())
        except Exception:
            pass

    print("Opening input stream… (Ctrl+C to quit)")
    try:
        with sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="int16",
            blocksize=0,
            device=args.device_index or args.device_name,
            callback=_callback,
        ):
            block = harness.frame_length
            last_print = time.monotonic()
            while True:
                # Repack queued bytes into exact-size frames
                need = block * 2  # int16 mono
                buf = bytearray()
                if pending:
                    take = min(len(pending), need)
                    buf.extend(pending[:take])
                    pending = pending[take:]
                while len(buf) < need and q:
                    buf.extend(q.pop(0))
                if len(buf) < need:
                    time.sleep(0.01)
                    continue
                if len(buf) > need:
                    pending = bytes(buf[need:]) + pending
                    buf = buf[:need]
                pcm = np.frombuffer(bytes(buf), dtype=np.int16)
                if gain != 1.0 and pcm.size:
                    amplified = np.clip(pcm.astype(np.float32) * gain, -32768, 32767)
                    pcm = amplified.astype(np.int16)
                triggered, conf = harness.process(pcm, verbose=args.verbose)
                if triggered:
                    print(f"[DETECTED] conf={conf if conf is not None else 'n/a'}")
                    if args.stop_on_detect:
                        break
                # Periodic heartbeat
                now = time.monotonic()
                if now - last_print > 5.0:
                    print("…listening (press Ctrl+C to stop)…")
                    last_print = now
    except KeyboardInterrupt:
        print("Interrupted; exiting.")


if __name__ == "__main__":
    main()
