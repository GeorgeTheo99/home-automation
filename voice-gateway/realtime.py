from __future__ import annotations

import asyncio
import base64
import json
import time
import threading
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional

import numpy as np
import websockets


JsonDict = Dict[str, Any]


@dataclass
class RealtimeConfig:
    api_key: str
    model: str
    modalities: List[str]
    voice: Optional[str]
    temperature: float
    session_timeout: float
    max_audio_secs: float
    send_audio: bool
    expect_audio_output: bool
    input_sample_rate: int
    stream_sample_rate: int = 24000
    post_command_mute: float = 0.0
    playback_guard: float = 0.0
    server_vad_enabled: bool = False
    server_vad_threshold: float = 0.5
    server_vad_silence_ms: int = 200
    server_vad_prefix_padding_ms: int = 300
    server_vad_idle_timeout_ms: Optional[int] = None
    turn_create_response: bool = True
    turn_interrupt_response: bool = True
    chunk_duration_ms: float = 100.0
    min_input_audio_ms: int = 600
    noise_threshold: int = 0
    force_create_response: bool = True
    force_response_delay_ms: int = 150
    fallback_no_speech: Optional[str] = "Sorry, I didn't catch that."
    fallback_no_response: Optional[str] = "Sorry, something went wrong responding."


@dataclass
class RealtimeResult:
    handled: bool = False
    action_summaries: List[str] = field(default_factory=list)
    response_text: str = ""
    transcript: str = ""
    audio_played: bool = False
    audio_duration: float = 0.0
    tool_results: List[JsonDict] = field(default_factory=list)
    action_detected: bool = False
    informational_detected: bool = False
    intent: str = "unknown"
    captured_audio: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=np.int16))
    input_samples: int = 0
    input_duration_ms: float = 0.0
    min_input_samples: int = 0
    min_input_duration_ms: float = 0.0
    error_code: Optional[str] = None
    error_message: str = ""
    response_id: Optional[str] = None
    noise_detected: bool = False
    voice_activity_detected: bool = False
    peak_amplitude: int = 0
    speech_segments: int = 0


class RealtimeError(Exception):
    pass


class RealtimeConversation:
    """Manage a persistent realtime session across multiple turns."""

    def __init__(
        self,
        config: RealtimeConfig,
        entity_context: List[JsonDict],
        instructions: str,
        ha_executor: Callable[[str, str, Any, Optional[Dict[str, Any]]], Any],
        confirm_message: Callable[[Dict[str, Any]], str],
        speaker: Callable[[str], None],
        mute_microphone: Callable[[float], None],
        logger: Callable[[str, str], None],
    ) -> None:
        self.config = config
        self.entity_context = entity_context
        self.instructions = instructions
        self.ha_executor = ha_executor
        self.confirm_message = confirm_message
        self.speaker = speaker
        self.mute_microphone = mute_microphone
        self.logger = logger

        self._loop = asyncio.new_event_loop()
        self._loop_ready = threading.Event()
        self._thread = threading.Thread(target=self._loop_runner, name="RealtimeConversation", daemon=True)
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._closed = False
        self._first_turn = True
        self._tools: List[Dict[str, Any]] = []
        self._full_instructions: str = instructions

        self._thread.start()
        self._loop_ready.wait()

        try:
            future = asyncio.run_coroutine_threadsafe(self._connect(), self._loop)
            future.result()
        except Exception:
            self._closed = True
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=1.0)
            raise

    def _loop_runner(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop_ready.set()
        self._loop.run_forever()

    async def _connect(self) -> None:
        if not self.config.api_key:
            raise RealtimeError("Realtime enabled but OPENAI_API_KEY missing.")

        url = f"wss://api.openai.com/v1/realtime?model={self.config.model}"
        headers = [
            ("Authorization", f"Bearer {self.config.api_key}"),
            ("OpenAI-Beta", "realtime=v1"),
        ]
        self._ws = await websockets.connect(url, additional_headers=headers, max_size=20_000_000)
        self._tools = _build_tools()
        self._full_instructions = _compose_instructions(self.instructions, self.entity_context)
        await _send_session_update(self._ws, self.config, self._full_instructions, self._tools)

    def send_turn(
        self,
        pcm16: np.ndarray,
        live_audio_reader: Optional[Callable[[int], np.ndarray]] = None,
        live_audio_block: int = 0,
    ) -> RealtimeResult:
        if self._closed:
            raise RuntimeError("Realtime conversation already closed.")
        future = asyncio.run_coroutine_threadsafe(
            self._handle_turn(pcm16, live_audio_reader, live_audio_block), self._loop
        )
        return future.result()

    async def _handle_turn(
        self,
        pcm16: np.ndarray,
        live_audio_reader: Optional[Callable[[int], np.ndarray]],
        live_audio_block: int,
    ) -> RealtimeResult:
        if self._ws is None:
            raise RealtimeError("Realtime session not connected.")

        result = await _process_realtime_turn(
            self._ws,
            pcm16,
            self.config,
            self._full_instructions,
            self._tools,
            self.ha_executor,
            self.confirm_message,
            self.speaker,
            self.mute_microphone,
            self.logger,
            live_audio_reader,
            live_audio_block,
            clear_buffer_before=not self._first_turn,
        )
        self._first_turn = False
        return result


    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            future = asyncio.run_coroutine_threadsafe(self._close(), self._loop)
            future.result()
        finally:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=1.0)

    async def _close(self) -> None:
        if self._ws is not None:
            try:
                await self._ws.send(json.dumps({"type": "session.close"}))
            except Exception:
                pass
            try:
                await self._ws.close()
            except Exception:
                pass
        self._ws = None



def _read_frames(
    pcm16: np.ndarray,
    sample_rate: int,
    chunk_ms: float,
    max_audio_secs: float,
) -> List[bytes]:
    if pcm16.dtype != np.int16:
        pcm16 = pcm16.astype(np.int16)
    chunk_samples = max(int(sample_rate * (chunk_ms / 1000.0)), 1)
    max_samples = int(max_audio_secs * sample_rate) if max_audio_secs > 0 else len(pcm16)
    max_samples = min(len(pcm16), max_samples) if max_samples > 0 else len(pcm16)
    chunks: List[bytes] = []
    offset = 0
    while offset < max_samples:
        end = min(max_samples, offset + chunk_samples)
        data = pcm16[offset:end].tobytes()
        if data:
            chunks.append(data)
        offset = end
    return chunks


def _resample_pcm16(pcm16: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    if pcm16.size == 0 or from_rate == to_rate:
        return pcm16.astype(np.int16, copy=False)

    duration = pcm16.shape[0] / float(from_rate)
    target_len = int(round(duration * to_rate))
    if target_len <= 0:
        return np.empty((0,), dtype=np.int16)

    src_index = np.linspace(0.0, pcm16.shape[0] - 1, num=pcm16.shape[0], dtype=np.float32)
    tgt_index = np.linspace(0.0, pcm16.shape[0] - 1, num=target_len, dtype=np.float32)
    resampled = np.interp(tgt_index, src_index, pcm16.astype(np.float32))
    resampled = np.clip(np.round(resampled), -32768, 32767).astype(np.int16)
    return resampled


def _build_tools() -> List[Dict[str, Any]]:
    return [
        {
            "type": "function",
            "name": "call_home_assistant",
            "description": "Call a Home Assistant service using domain/service/entity_id/data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "domain": {"type": "string"},
                    "service": {"type": "string"},
                    "entity_id": {
                        "oneOf": [
                            {"type": "string"},
                            {"type": "array", "items": {"type": "string"}, "minItems": 1},
                        ]
                    },
                    "data": {"type": "object"},
                },
                "required": ["domain", "service", "entity_id"],
            },
        }
    ]


def _compose_instructions(base_instructions: str, entity_context: List[JsonDict]) -> str:
    entity_json = json.dumps(entity_context, ensure_ascii=False)
    return base_instructions + "\n\nValid entities JSON:" + entity_json


async def _send_session_update(
    ws: websockets.WebSocketClientProtocol,
    config: RealtimeConfig,
    full_instructions: str,
    tools: List[Dict[str, Any]],
) -> None:
    session_body: Dict[str, Any] = {
        "modalities": config.modalities,
        "temperature": max(0.6, config.temperature),
        "instructions": full_instructions,
        "tools": tools,
        "tool_choice": "auto",
    }
    if config.server_vad_enabled:
        turn_detection = {
            "type": "server_vad",
            "threshold": config.server_vad_threshold,
            "silence_duration_ms": config.server_vad_silence_ms,
            "prefix_padding_ms": config.server_vad_prefix_padding_ms,
            "idle_timeout_ms": config.server_vad_idle_timeout_ms,
            "create_response": config.turn_create_response,
            "interrupt_response": config.turn_interrupt_response,
        }
        session_body["turn_detection"] = {k: v for k, v in turn_detection.items() if v is not None}
    if config.send_audio:
        session_body["input_audio_format"] = "pcm16"
        if config.stream_sample_rate:
            session_body["input_audio_transcription"] = {"model": "gpt-4o-mini-transcribe"}
    if "audio" in config.modalities:
        session_body["output_audio_format"] = "pcm16"
    if config.voice and "audio" in config.modalities:
        session_body["voice"] = config.voice

    await ws.send(json.dumps({"type": "session.update", "session": session_body}))


async def _process_realtime_turn(
    ws: websockets.WebSocketClientProtocol,
    pcm16: np.ndarray,
    config: RealtimeConfig,
    full_instructions: str,
    tools: List[Dict[str, Any]],
    ha_executor: Callable[[str, str, Any, Optional[Dict[str, Any]]], Any],
    confirm_message: Callable[[Dict[str, Any]], str],
    speaker: Callable[[str], None],
    mute_microphone: Callable[[float], None],
    logger: Callable[[str, str], None],
    live_audio_reader: Optional[Callable[[int], np.ndarray]],
    live_audio_block: int,
    *,
    clear_buffer_before: bool,
) -> RealtimeResult:
    result = RealtimeResult()
    if pcm16 is None:
        pcm16 = np.empty((0,), dtype=np.int16)
    else:
        pcm16 = np.asarray(pcm16, dtype=np.int16).flatten()

    loop = asyncio.get_running_loop()
    target_rate = config.stream_sample_rate or config.input_sample_rate
    chunk_ms = config.chunk_duration_ms if config.chunk_duration_ms > 0 else 100.0
    if chunk_ms < 20.0:
        chunk_ms = 20.0
    chunk_samples_target = max(1, int(target_rate * (chunk_ms / 1000.0)))
    chunk_samples_input = max(1, int(config.input_sample_rate * (chunk_ms / 1000.0)))
    if config.min_input_audio_ms and config.min_input_audio_ms > 0:
        min_audio_ms = config.min_input_audio_ms
    else:
        min_audio_ms = 0
    if not config.server_vad_enabled:
        min_audio_ms = max(min_audio_ms, 3000)
    if min_audio_ms > 0:
        min_required_samples = max(1, int((min_audio_ms / 1000.0) * config.input_sample_rate))
    else:
        min_required_samples = 0
    max_input_samples = int(config.max_audio_secs * config.input_sample_rate) if config.max_audio_secs > 0 else None

    pending_resampled = np.empty((0,), dtype=np.int16)
    captured_parts: List[np.ndarray] = []
    total_input_samples = 0
    peak_value = 0
    total_chunks_sent = 0
    total_bytes_sent = 0

    def prepare_chunks(samples: Optional[np.ndarray]) -> List[bytes]:
        nonlocal pending_resampled, total_input_samples, peak_value
        if samples is None:
            return []
        arr = np.asarray(samples, dtype=np.int16).flatten()
        if arr.size == 0:
            return []
        captured_parts.append(arr.copy())
        total_input_samples += arr.size
        peak_value = max(peak_value, int(np.abs(arr).max())) if arr.size else peak_value
        resampled = arr if target_rate == config.input_sample_rate else _resample_pcm16(arr, config.input_sample_rate, target_rate)
        if resampled.size == 0:
            return []
        pending_resampled = np.concatenate([pending_resampled, resampled])
        chunks: List[bytes] = []
        while pending_resampled.size >= chunk_samples_target:
            chunk = pending_resampled[:chunk_samples_target]
            pending_resampled = pending_resampled[chunk_samples_target:]
            chunks.append(chunk.tobytes())
        return chunks

    def flush_chunks() -> List[bytes]:
        nonlocal pending_resampled
        if pending_resampled.size == 0:
            return []
        chunk = pending_resampled
        pending_resampled = np.empty((0,), dtype=np.int16)
        return [chunk.tobytes()]

    speech_stopped = asyncio.Event()
    committed = asyncio.Event()
    send_exception: Optional[Exception] = None
    voice_activity_detected = False
    speech_segments = 0

    async def audio_sender() -> None:
        nonlocal send_exception, total_chunks_sent, total_bytes_sent
        try:
            if not config.send_audio:
                if pcm16.size:
                    captured_parts.append(pcm16.copy())
                committed.set()
                return

            initial_chunks = prepare_chunks(pcm16)
            for chunk_bytes in initial_chunks:
                audio_b64 = base64.b64encode(chunk_bytes).decode("ascii")
                await ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": audio_b64}))
                total_chunks_sent += 1
                total_bytes_sent += len(chunk_bytes)

            read_size = live_audio_block if live_audio_block > 0 else chunk_samples_input
            read_size = max(1, read_size)

            while not committed.is_set():
                if speech_stopped.is_set():
                    break
                if max_input_samples is not None and total_input_samples >= max_input_samples:
                    break
                if live_audio_reader is None:
                    break

                samples = await loop.run_in_executor(None, live_audio_reader, read_size)
                arr = np.asarray(samples, dtype=np.int16).flatten()
                if arr.size == 0:
                    await asyncio.sleep(0.01)
                    continue

                for chunk_bytes in prepare_chunks(arr):
                    audio_b64 = base64.b64encode(chunk_bytes).decode("ascii")
                    await ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": audio_b64}))
                    total_chunks_sent += 1
                    total_bytes_sent += len(chunk_bytes)

                if max_input_samples is not None and total_input_samples >= max_input_samples:
                    break

            if not committed.is_set():
                for chunk_bytes in flush_chunks():
                    audio_b64 = base64.b64encode(chunk_bytes).decode("ascii")
                    await ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": audio_b64}))
                    total_chunks_sent += 1
                    total_bytes_sent += len(chunk_bytes)

            if committed.is_set():
                return

            # If server VAD is enabled but we never saw speech, avoid committing
            # an empty buffer which causes commit_empty errors and redundant turns.
            if config.server_vad_enabled and not voice_activity_detected:
                send_exception = RealtimeError("insufficient_audio")
                result.error_code = "insufficient_audio"
                result.error_message = "No speech detected after wake word."
                result.noise_detected = True
                committed.set()
                return

            if total_input_samples < min_required_samples:
                send_exception = RealtimeError(
                    f"Captured audio below minimum length ({min_audio_ms} ms minimum)."
                )
                result.error_code = "insufficient_audio"
                result.error_message = f"Captured audio below minimum length ({min_audio_ms} ms minimum)."
                logger("WARN", f"Realtime session: insufficient audio (<{min_audio_ms} ms); skipping.")
                if not voice_activity_detected:
                    result.noise_detected = True
                committed.set()
                return

            await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
            committed.set()
            logger("INFO", "Realtime input buffer committed")
        except Exception as exc:  # pragma: no cover
            send_exception = exc
            committed.set()

    if clear_buffer_before:
        try:
            await ws.send(json.dumps({"type": "input_audio_buffer.clear"}))
        except Exception:
            pass

    response_id: Optional[str] = None
    pending_tool_args: Dict[str, Dict[str, Any]] = {}
    collected_text: List[str] = []
    collected_audio: bytearray = bytearray()
    response_requested = False
    response_create_task: Optional[asyncio.Task] = None
    transcript_fragments: List[str] = []

    def _flatten_text(fragment: Any) -> List[str]:
        pieces: List[str] = []

        def _visit(obj: Any) -> None:
            if isinstance(obj, str):
                if obj:
                    pieces.append(obj)
            elif isinstance(obj, dict):
                for key in ("text", "value", "content", "delta"):
                    if key in obj:
                        _visit(obj[key])
                if "items" in obj:
                    _visit(obj["items"])
            elif isinstance(obj, list):
                for item in obj:
                    _visit(item)

        if fragment is not None:
            _visit(fragment)
        return pieces

    def _append_text(text_fragment: Any) -> None:
        for part in _flatten_text(text_fragment):
            collected_text.append(part)

    def _flatten_audio(fragment: Any) -> List[str]:
        chunks: List[str] = []

        def _visit(obj: Any) -> None:
            if isinstance(obj, str):
                if obj:
                    chunks.append(obj)
            elif isinstance(obj, dict):
                for key in ("audio", "chunk", "value", "delta", "data", "content"):
                    if key in obj:
                        _visit(obj[key])
                if "chunks" in obj:
                    _visit(obj["chunks"])
            elif isinstance(obj, list):
                for item in obj:
                    _visit(item)

        if fragment is not None:
            _visit(fragment)
        return chunks

    def _append_audio(audio_fragment: Any) -> None:
        for audio_b64 in _flatten_audio(audio_fragment):
            try:
                collected_audio.extend(base64.b64decode(audio_b64))
            except Exception:
                continue

    def _build_response_payload() -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "modalities": config.modalities,
            "temperature": max(0.6, config.temperature),
            "instructions": full_instructions,
            "tools": tools,
        }
        if "audio" in config.modalities:
            payload["output_audio_format"] = "pcm16"
        if config.voice and "audio" in config.modalities:
            payload["voice"] = config.voice
        return payload

    def apply_fallback(message: Optional[str], reason: str) -> None:
        if result.handled or not message:
            return
        result.response_text = message
        result.handled = True
        result.intent = "informational"
        logger("INFO", f"Realtime local fallback ({reason}): {message}")

    async def send_tool_result(tool_call_id: str, output: JsonDict) -> None:
        payload = {
            "type": "response.tool_result",
            "tool_call_id": tool_call_id,
            "output": json.dumps(output, ensure_ascii=False),
        }
        if response_id:
            payload["response_id"] = response_id
        await ws.send(json.dumps(payload))

    async def schedule_response_create(reason: str) -> None:
        nonlocal response_create_task, response_requested
        try:
            if not config.force_create_response:
                return
            delay = max(0, config.force_response_delay_ms) / 1000.0
            if delay > 0:
                await asyncio.sleep(delay)
            if response_requested:
                return
            try:
                await asyncio.wait_for(committed.wait(), timeout=max(0.5, config.session_timeout))
            except asyncio.TimeoutError:
                logger("WARN", f"Realtime response.create skipped ({reason}): input audio was never committed.")
                return
            if response_requested:
                return
            payload = _build_response_payload()
            await ws.send(json.dumps({"type": "response.create", "response": payload}))
            response_requested = True
            logger("DEBUG", f"Realtime response.create issued (reason={reason}).")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger("WARN", f"Realtime response.create failed ({reason}): {exc}")
        finally:
            response_create_task = None

    def trigger_response_create(reason: str) -> None:
        nonlocal response_create_task
        if not config.force_create_response or response_requested:
            return
        if response_create_task and not response_create_task.done():
            response_create_task.cancel()
        response_create_task = asyncio.create_task(schedule_response_create(reason))

    def mark_response_started() -> None:
        nonlocal response_requested, response_create_task
        if not response_requested:
            response_requested = True
        if response_create_task and not response_create_task.done():
            response_create_task.cancel()

    send_task = asyncio.create_task(audio_sender()) if config.send_audio else None

    if not config.turn_create_response:
        payload = _build_response_payload()
        await ws.send(json.dumps({"type": "response.create", "response": payload}))
        mark_response_started()

    start_time = time.time()

    try:
        while True:
            if send_exception:
                raise send_exception
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=config.session_timeout)
            except asyncio.TimeoutError:
                result.error_code = result.error_code or "session_timeout"
                raise RealtimeError("Realtime session timed out")
            data = json.loads(msg)
            event_type = data.get("type")

            if event_type == "response.created":
                logger("DEBUG", "Realtime event: response.created")
                mark_response_started()
                response = data.get("response", {})
                if isinstance(response, dict):
                    response_id = response.get("id", response_id)
            elif event_type in {"response.output_text.delta", "response.text.delta"}:
                delta_payload = data.get("delta")
                text_payload = delta_payload if delta_payload is not None else data.get("text")
                preview_parts = _flatten_text(text_payload)
                if preview_parts:
                    logger(
                        "DEBUG",
                        f"Realtime event: text.delta len={sum(len(p) for p in preview_parts)}",
                    )
                mark_response_started()
                response_id = data.get("response_id", response_id)
                _append_text(text_payload)
            elif event_type in {"response.output_text.done", "response.text.done"}:
                text_payload = data.get("text") or data.get("output") or data.get("delta")
                preview_parts = _flatten_text(text_payload)
                if preview_parts:
                    logger(
                        "DEBUG",
                        f"Realtime event: text.done len={sum(len(p) for p in preview_parts)}",
                    )
                mark_response_started()
                response_id = data.get("response_id", response_id)
                _append_text(text_payload)
            elif event_type == "response.output_audio.delta":
                audio_payload = data.get("audio")
                if audio_payload is None:
                    audio_payload = data.get("delta")
                chunk_lengths = [_len for _len in (len(chunk) for chunk in _flatten_audio(audio_payload)) if _len]
                if chunk_lengths:
                    logger("DEBUG", f"Realtime event: audio.delta chunks={len(chunk_lengths)} total_b64={sum(chunk_lengths)}")
                mark_response_started()
                response_id = data.get("response_id", response_id)
                _append_audio(audio_payload)
            elif event_type == "response.output_audio.done":
                audio_payload = data.get("audio") or data.get("output") or data.get("delta")
                chunk_lengths = [_len for _len in (len(chunk) for chunk in _flatten_audio(audio_payload)) if _len]
                if chunk_lengths:
                    logger("DEBUG", f"Realtime event: audio.done chunks={len(chunk_lengths)} total_b64={sum(chunk_lengths)}")
                mark_response_started()
                response_id = data.get("response_id", response_id)
                _append_audio(audio_payload)
            elif event_type in {"response.audio.delta"}:
                audio_payload = data.get("audio") or data.get("delta")
                chunk_lengths = [_len for _len in (len(chunk) for chunk in _flatten_audio(audio_payload)) if _len]
                if chunk_lengths:
                    logger("DEBUG", f"Realtime event: audio.delta chunks={len(chunk_lengths)} total_b64={sum(chunk_lengths)}")
                mark_response_started()
                response_id = data.get("response_id", response_id)
                _append_audio(audio_payload)
            elif event_type in {"response.audio.done"}:
                audio_payload = data.get("audio") or data.get("output") or data.get("delta")
                chunk_lengths = [_len for _len in (len(chunk) for chunk in _flatten_audio(audio_payload)) if _len]
                if chunk_lengths:
                    logger("DEBUG", f"Realtime event: audio.done chunks={len(chunk_lengths)} total_b64={sum(chunk_lengths)}")
                mark_response_started()
                response_id = data.get("response_id", response_id)
                _append_audio(audio_payload)
            elif event_type in {"response.audio_transcript.delta"}:
                text_payload = data.get("delta") or data.get("text") or data.get("transcript")
                preview_parts = _flatten_text(text_payload)
                if preview_parts:
                    logger("DEBUG", f"Realtime event: audio_transcript.delta len={sum(len(p) for p in preview_parts)}")
                mark_response_started()
                response_id = data.get("response_id", response_id)
                _append_text(text_payload)
            elif event_type in {"response.audio_transcript.done"}:
                text_payload = data.get("transcript") or data.get("text") or data.get("delta")
                preview_parts = _flatten_text(text_payload)
                if preview_parts:
                    logger("DEBUG", f"Realtime event: audio_transcript.done len={sum(len(p) for p in preview_parts)}")
                mark_response_started()
                response_id = data.get("response_id", response_id)
                _append_text(text_payload)
            elif event_type in {"response.content_part.added"}:
                part = data.get("part")
                mark_response_started()
                response_id = data.get("response_id", response_id)
                _append_text(part)
                _append_audio(part)
            elif event_type in {"response.content_part.delta"}:
                delta_payload = data.get("delta")
                mark_response_started()
                response_id = data.get("response_id", response_id)
                _append_text(delta_payload)
                _append_audio(delta_payload)
            elif event_type in {"response.content_part.done"}:
                part = data.get("part")
                mark_response_started()
                response_id = data.get("response_id", response_id)
                _append_text(part)
                _append_audio(part)
            elif event_type in {"response.output_item.added", "output_item.added"}:
                logger("DEBUG", "Realtime event: output_item.added")
                mark_response_started()
                response_id = data.get("response_id", response_id)
                item = data.get("item") or {}
                if item.get("type") == "function_call":
                    call_id = item.get("call_id")
                    if call_id:
                        info = pending_tool_args.setdefault(call_id, {"name": item.get("name"), "args": ""})
                        if item.get("arguments"):
                            info["args"] += item["arguments"]
            elif event_type in {"response.output_item.done", "output_item.done"}:
                logger("DEBUG", "Realtime event: output_item.done")
                response_id = data.get("response_id", response_id)
            elif event_type == "conversation.item.input_audio_transcription.completed":
                transcript = data.get("transcript")
                item_obj = data.get("item")
                if isinstance(item_obj, dict):
                    item_id = item_obj.get("id")
                else:
                    item_id = data.get("item_id")
                if transcript:
                    logger(
                        "DEBUG",
                        f"Realtime event: transcription len={len(transcript)} item_id={item_id!s}",
                    )
                    logger("INFO", f"Realtime heard: {transcript}")
                    result.transcript = transcript
                    trigger_response_create("transcription")
            elif event_type == "conversation.item.input_audio_transcription.delta":
                delta_text = data.get("delta")
                fragments = _flatten_text(delta_text)
                if fragments:
                    transcript_fragments.extend(fragments)
                    result.transcript = "".join(transcript_fragments)
                    logger("DEBUG", f"Realtime event: transcription.delta len={sum(len(p) for p in fragments)} item_id={data.get('item_id')!s}")
            elif event_type == "conversation.item.created":
                logger("DEBUG", "Realtime event: conversation.item.created")
            elif event_type == "input_audio_buffer.speech_started":
                logger("DEBUG", "Realtime event: speech_started")
                voice_activity_detected = True
                speech_segments += 1
            elif event_type == "input_audio_buffer.speech_stopped":
                logger("DEBUG", "Realtime event: speech_stopped")
                speech_stopped.set()
                trigger_response_create("speech_stopped")
            elif event_type == "input_audio_buffer.committed":
                logger("DEBUG", "Realtime event: buffer.committed")
                committed.set()
                trigger_response_create("commit_event")
            elif event_type == "input_audio_buffer.cleared":
                logger("DEBUG", "Realtime event: buffer.cleared")
            elif event_type == "rate_limits.updated":
                logger("DEBUG", "Realtime event: rate_limits.updated")
            elif event_type == "response.function_call_arguments.delta":
                logger("DEBUG", "Realtime event: function_call_arguments.delta")
                mark_response_started()
                response_id = data.get("response_id", response_id)
                call_id = data.get("call_id")
                if call_id:
                    info = pending_tool_args.setdefault(call_id, {"name": None, "args": ""})
                    if data.get("delta"):
                        info["args"] += data["delta"]
            elif event_type == "response.function_call_arguments.done":
                logger("DEBUG", "Realtime event: function_call_arguments.done")
                mark_response_started()
                response_id = data.get("response_id", response_id)
                call_id = data.get("call_id")
                if call_id and call_id in pending_tool_args:
                    info = pending_tool_args[call_id]
                    if data.get("name"):
                        info["name"] = data["name"]
                    name = info.get("name")
                    args_json: Dict[str, Any] = {}
                    if info.get("args"):
                        try:
                            args_json = json.loads(info["args"])
                        except json.JSONDecodeError as exc:
                            await send_tool_result(call_id, {"error": f"Invalid JSON args: {exc}"})
                            pending_tool_args.pop(call_id, None)
                            continue
                    else:
                        args_json = {}
                    if name == "call_home_assistant":
                        try:
                            ha_result = ha_executor(
                                args_json.get("domain"),
                                args_json.get("service"),
                                args_json.get("entity_id"),
                                args_json.get("data"),
                            )
                            summary = confirm_message({
                                "domain": args_json.get("domain"),
                                "service": args_json.get("service"),
                                "entity_id": args_json.get("entity_id"),
                                "data": args_json.get("data", {}),
                            })
                            if config.post_command_mute > 0:
                                mute_microphone(config.post_command_mute)
                            call_record = {
                                "domain": args_json.get("domain"),
                                "service": args_json.get("service"),
                                "entity_id": args_json.get("entity_id"),
                                "data": args_json.get("data", {}),
                                "result": ha_result,
                            }
                            result.tool_results.append(call_record)
                            await send_tool_result(call_id, {
                                "ok": True,
                                "result": ha_result,
                                "summary": summary,
                            })
                            if summary:
                                result.action_summaries.append(summary)
                                logger("INFO", f"Realtime action: {summary}")
                            if summary and not config.expect_audio_output:
                                speaker(summary)
                        except Exception as exc:
                            await send_tool_result(call_id, {"ok": False, "error": str(exc)})
                    else:
                        await send_tool_result(call_id, {"error": f"Unknown tool {name}"})
                    pending_tool_args.pop(call_id, None)
            elif event_type in {"response.completed", "response.done"}:
                logger("DEBUG", "Realtime event: response.completed")
                response_id = data.get("response_id", response_id)
                if config.force_create_response and not response_requested:
                    if response_create_task and not response_create_task.done():
                        response_create_task.cancel()
                    response_create_task = None
                    payload = _build_response_payload()
                    await ws.send(json.dumps({"type": "response.create", "response": payload}))
                    response_requested = True
                    logger("DEBUG", "Realtime response.create issued (reason=auto_complete).")
                    continue
                break
            elif event_type in {"error", "invalid_request_error"}:
                logger("DEBUG", f"Realtime event: error payload={data}")
                err_payload = data.get("error") if isinstance(data, dict) else None
                if isinstance(err_payload, dict):
                    code = err_payload.get("code")
                    message = err_payload.get("message") or ""
                elif isinstance(data, dict):
                    code = data.get("code")
                    message = data.get("message") or ""
                    err_payload = data
                else:
                    code = None
                    message = ""
                    err_payload = data

                result.error_code = code
                result.error_message = message
                if code == "input_audio_buffer_commit_empty":
                    logger("INFO", "Realtime session: no speech detected after wake word.")
                    result.voice_activity_detected = voice_activity_detected
                    result.speech_segments = speech_segments
                    result.peak_amplitude = peak_value
                    if not result.transcript and total_input_samples <= 0 and not voice_activity_detected:
                        result.noise_detected = True
                    break

                raise RealtimeError(str(err_payload))
            else:
                logger(
                    "DEBUG",
                    f"Realtime event: unhandled type={event_type!r} keys={sorted(data.keys())}",
                )

        if time.time() - start_time > config.session_timeout:
            raise RealtimeError("Realtime session exceeded timeout")

        if send_task:
            speech_stopped.set()
            committed.set()
            await send_task
            if send_exception:
                if isinstance(send_exception, RealtimeError) and result.error_code == "insufficient_audio":
                    logger("INFO", "Realtime session: no speech captured after wake word.")
                else:
                    raise send_exception

        if config.send_audio and total_chunks_sent:
            total_ms = (total_input_samples / float(config.input_sample_rate)) * 1000.0 if config.input_sample_rate else 0.0
            logger(
                "INFO",
                (
                    f"Realtime capture: {total_ms:.0f} ms audio "
                    f"({total_chunks_sent} chunks, {total_bytes_sent // 1024} KiB, peak={peak_value})"
                ),
            )

        play_audio = config.expect_audio_output and (not result.tool_results or collected_text)
        if collected_audio and play_audio:
            try:
                import sounddevice as sd

                samples = np.frombuffer(bytes(collected_audio), dtype=np.int16)
                if samples.size:
                    playback_rate = max(1, config.stream_sample_rate)
                    duration = samples.size / float(playback_rate)
                    mute_duration = duration + max(0.0, config.playback_guard)
                    if mute_duration > 0:
                        mute_microphone(mute_duration)
                    sd.play(samples / 32768.0, samplerate=playback_rate)
                    sd.wait()
                    result.audio_played = True
                    result.audio_duration = duration
            except Exception as exc:
                logger("WARN", f"Realtime audio playback failed: {exc}")
        elif collected_audio and config.expect_audio_output and result.tool_results and not collected_text:
            logger("INFO", "Realtime audio suppressed for action intent.")

        if collected_text:
            response_text = "".join(collected_text)
            result.response_text = response_text
            logger("INFO", f"Realtime response text: {response_text}")

        result.response_id = response_id
        result.voice_activity_detected = voice_activity_detected
        result.peak_amplitude = peak_value
        result.speech_segments = speech_segments

        should_cancel_noise = (
            not result.tool_results
            and not collected_text
            and not collected_audio
            and not result.audio_played
            and not voice_activity_detected
            and not result.transcript
            and config.noise_threshold > 0
            and peak_value <= config.noise_threshold
        )

        if should_cancel_noise:
            result.noise_detected = True
            result.error_code = result.error_code or "no_speech"
            result.error_message = result.error_message or "No speech detected after wake word."
            if response_id:
                try:
                    await ws.send(json.dumps({"type": "response.cancel", "response_id": response_id}))
                    logger("INFO", "Realtime response cancelled due to noise-only input.")
                except Exception as exc:
                    logger("WARN", f"Realtime cancel failed: {exc}")

        if result.audio_played or result.response_text or result.action_summaries or result.tool_results:
            result.handled = True
        elif not result.noise_detected:
            if result.transcript:
                result.error_code = result.error_code or "no_response"
                result.error_message = result.error_message or "Realtime model produced no output."
                apply_fallback(config.fallback_no_response, "no_response")
            else:
                result.error_code = result.error_code or "no_transcript"
                result.error_message = result.error_message or "Realtime model could not understand the speech."
                apply_fallback(config.fallback_no_speech, "no_transcript")
            if not result.handled:
                logger("INFO", "Realtime session completed with no output and no fallback message configured.")

        result.action_detected = bool(result.tool_results)
        result.informational_detected = bool(
            result.response_text or (result.audio_played and not result.tool_results)
        )
        if result.action_detected and result.informational_detected:
            result.intent = "mixed"
        elif result.action_detected:
            result.intent = "action"
        elif result.informational_detected:
            result.intent = "informational"
        else:
            result.intent = "unknown"

    except Exception as exc:
        logger("ERROR", f"Realtime session error: {exc}")
        error_code = result.error_code or "exception"
        result.error_code = error_code
        result.error_message = result.error_message or str(exc)
        if error_code == "insufficient_audio":
            if voice_activity_detected or total_input_samples > 0 or result.transcript:
                apply_fallback(config.fallback_no_speech, error_code)
            else:
                result.noise_detected = True
                logger("INFO", "Realtime session: insufficient audio with no speech; suppressing fallback.")
        elif error_code == "input_audio_buffer_commit_empty":
            result.noise_detected = True
            result.error_message = result.error_message or "No speech detected after wake word."
        elif error_code == "session_timeout":
            if voice_activity_detected or total_input_samples > 0 or result.transcript:
                apply_fallback(config.fallback_no_response, "session_timeout")
            else:
                logger("INFO", "Realtime session timed out with no speech; suppressing fallback.")
        else:
            apply_fallback(config.fallback_no_response, "exception")
    finally:
        if response_create_task and not response_create_task.done():
            response_create_task.cancel()
            with suppress(asyncio.CancelledError):
                await response_create_task

        if captured_parts:
            try:
                result.captured_audio = np.concatenate(captured_parts).astype(np.int16, copy=False)
            except ValueError:
                result.captured_audio = pcm16.copy()
        else:
            result.captured_audio = pcm16.copy()

        result.input_samples = total_input_samples
        if config.input_sample_rate:
            result.input_duration_ms = (total_input_samples / float(config.input_sample_rate)) * 1000.0
        else:
            result.input_duration_ms = 0.0
        result.min_input_samples = min_required_samples
        result.min_input_duration_ms = float(min_audio_ms)
        result.response_id = result.response_id or response_id
        result.voice_activity_detected = result.voice_activity_detected or voice_activity_detected
        result.peak_amplitude = peak_value
        result.speech_segments = result.speech_segments or speech_segments

        if config.send_audio:
            try:
                await ws.send(json.dumps({"type": "input_audio_buffer.clear"}))
            except Exception:
                pass

    return result



async def run_realtime_session(
    pcm16: np.ndarray,
    config: RealtimeConfig,
    entity_context: List[JsonDict],
    instructions: str,
    ha_executor: Callable[[str, str, Any, Optional[Dict[str, Any]]], Any],
    confirm_message: Callable[[Dict[str, Any]], str],
    speaker: Callable[[str], None],
    mute_microphone: Callable[[float], None],
    logger: Callable[[str, str], None],
    live_audio_reader: Optional[Callable[[int], np.ndarray]] = None,
    live_audio_block: int = 0,
) -> RealtimeResult:
    result = RealtimeResult()
    if pcm16 is None:
        pcm16 = np.empty((0,), dtype=np.int16)
    else:
        pcm16 = np.asarray(pcm16, dtype=np.int16).flatten()

    if not config.api_key:
        logger("ERROR", "Realtime enabled but OPENAI_API_KEY missing.")
        return result

    tools = _build_tools()
    full_instructions = _compose_instructions(instructions, entity_context)

    url = f"wss://api.openai.com/v1/realtime?model={config.model}"
    headers = [("Authorization", f"Bearer {config.api_key}"), ("OpenAI-Beta", "realtime=v1")]

    async with websockets.connect(url, additional_headers=headers, max_size=20_000_000) as ws:
        try:
            await _send_session_update(ws, config, full_instructions, tools)
            result = await _process_realtime_turn(
                ws,
                pcm16,
                config,
                full_instructions,
                tools,
                ha_executor,
                confirm_message,
                speaker,
                mute_microphone,
                logger,
                live_audio_reader,
                live_audio_block,
                clear_buffer_before=False,
            )
        finally:
            try:
                await ws.send(json.dumps({"type": "session.close"}))
            except Exception:
                pass
    return result
