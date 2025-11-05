"""Minimal OpenAI Realtime websocket client."""

from __future__ import annotations

import asyncio
import base64
import json
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional

import numpy as np
import websockets
import logging

from config import RealtimeConfig
from tools.registry import ToolRegistry


class RealtimeError(RuntimeError):
    """Raised when realtime operations fail."""


@dataclass
class RealtimeResponse:
    text: str = ""
    tool_results: List[Dict[str, Any]] = field(default_factory=list)
    response_id: Optional[str] = None


def resample_linear(pcm16: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Resample PCM16 audio using linear interpolation."""
    if source_rate == target_rate or pcm16.size == 0:
        return pcm16.astype(np.int16, copy=False)
    x_old = np.arange(pcm16.size, dtype=np.float32)
    duration = pcm16.size / float(source_rate)
    new_size = int(round(duration * target_rate))
    if new_size <= 1:
        return pcm16.astype(np.int16, copy=True)
    x_new = np.linspace(0, pcm16.size - 1, new_size, dtype=np.float32)
    interpolated = np.interp(x_new, x_old, pcm16.astype(np.float32))
    return np.clip(interpolated, -32768, 32767).astype(np.int16)


class RealtimeClient:
    """Thread-safe wrapper around the OpenAI Realtime websocket API."""

    def __init__(
        self,
        config: RealtimeConfig,
        tool_registry: ToolRegistry,
        output_audio_format: str = "pcm16",
        output_sample_rate: int = 24000,
    ) -> None:
        self._config = config
        self._tool_registry = tool_registry
        self._output_audio_format = output_audio_format
        self._output_sample_rate = output_sample_rate
        self._closed = False

        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(target=self._run_loop, daemon=True)
        self._loop_thread.start()

        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._lock = asyncio.Lock()
        # Start connecting in the background, but do not block startup.
        # We'll ensure connection (with timeout) on first use.
        self._bg_connect_future: Optional[asyncio.Future] = None
        self._schedule_background_connect()

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    async def _connect(self) -> None:
        url = f"wss://api.openai.com/v1/realtime?model={self._config.model}"
        headers = [
            ("Authorization", f"Bearer {self._config.api_key}"),
            ("OpenAI-Beta", "realtime=v1"),
        ]
        # Apply an open/connect timeout if configured
        open_timeout: Optional[float] = None
        try:
            open_timeout = float(self._config.connect_timeout)
            if open_timeout <= 0:
                open_timeout = None
        except Exception:
            open_timeout = None

        logging.info("Realtime: connecting to OpenAI…")
        self._ws = await websockets.connect(
            url,
            additional_headers=headers,
            max_size=20_000_000,
            open_timeout=open_timeout,
        )
        await self._send_session_update()
        logging.info("Realtime: handshake successful.")

    def _schedule_background_connect(self) -> None:
        # Schedule a background connect attempt if not already scheduled/connected.
        if self._ws is not None:
            return
        if self._bg_connect_future is not None and not self._bg_connect_future.done():
            return

        async def _try_connect_bg() -> None:
            try:
                await self._connect()
            except Exception as exc:
                logging.warning("Realtime: background connect failed: %s", exc)

        self._bg_connect_future = asyncio.run_coroutine_threadsafe(_try_connect_bg(), self._loop)

    async def _send_session_update(self) -> None:
        if self._ws is None:
            raise RealtimeError("Realtime websocket is not connected.")
        payload = {
            "type": "session.update",
            "session": {
                "modalities": self._config.modalities,
                "temperature": self._config.temperature,
                "instructions": self._config.instructions_prompt,
                "tools": self._tool_registry.schemas(),
            },
        }
        await self._ws.send(json.dumps(payload))

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        async def _shutdown() -> None:
            if self._ws is not None:
                await self._ws.close()
                self._ws = None
            self._loop.stop()

        future = asyncio.run_coroutine_threadsafe(_shutdown(), self._loop)
        future.result()
        self._loop_thread.join(timeout=1.0)

    def send_utterance(
        self,
        pcm16_audio: np.ndarray,
        source_sample_rate: int,
        audio_consumer: Callable[[np.ndarray], None],
    ) -> RealtimeResponse:
        future = asyncio.run_coroutine_threadsafe(
            self._send_utterance_async(pcm16_audio, source_sample_rate, audio_consumer),
            self._loop,
        )
        return future.result()

    def request_shutdown(self) -> None:
        """Initiate a best-effort shutdown of any in-flight realtime activity."""
        if self._closed:
            return

        if self._bg_connect_future is not None and not self._bg_connect_future.done():
            self._bg_connect_future.cancel()

        async def _close_ws() -> None:
            if self._ws is not None:
                try:
                    await self._ws.close()
                except Exception:
                    pass

        try:
            asyncio.run_coroutine_threadsafe(_close_ws(), self._loop)
        except RuntimeError:
            pass

    async def _send_utterance_async(
        self,
        pcm16_audio: np.ndarray,
        source_sample_rate: int,
        audio_consumer: Callable[[np.ndarray], None],
    ) -> RealtimeResponse:
        # Ensure connected before sending (respecting configured timeout)
        await self._ensure_connected(self._config.connect_timeout)

        async with self._lock:
            resampled = resample_linear(pcm16_audio.astype(np.int16), source_sample_rate, self._output_sample_rate)
            duration = pcm16_audio.size / float(source_sample_rate) if source_sample_rate else 0.0
            logging.info(
                "Realtime: streaming utterance (duration=%.2fs, input_samples=%d -> output_samples=%d).",
                duration,
                pcm16_audio.size,
                resampled.size,
            )
            await self._ws.send(json.dumps({"type": "input_audio_buffer.clear"}))
            for chunk in self._chunk_audio(resampled):
                audio_b64 = base64.b64encode(chunk.tobytes()).decode("ascii")
                await self._ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": audio_b64}))
            await self._ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
            response_payload = {
                "type": "response.create",
                "response": {
                    "modalities": self._config.modalities,
                    "temperature": self._config.temperature,
                },
            }
            if "audio" in self._config.modalities:
                response_payload["response"]["output_audio_format"] = self._output_audio_format
            await self._ws.send(json.dumps(response_payload))
            logging.debug("Realtime: request sent; awaiting events.")

            return await self._consume_events(audio_consumer)

    async def _ensure_connected(self, timeout: Optional[float]) -> None:
        if self._ws is not None:
            return
        # If a background connect is already inflight, wait for it up to timeout; else, connect now.
        if self._bg_connect_future is not None and not self._bg_connect_future.done():
            try:
                await asyncio.wait_for(asyncio.wrap_future(self._bg_connect_future), timeout=timeout)
            except Exception:
                # Fall through to an explicit connect attempt below to surface errors
                pass
        if self._ws is None:
            # Explicit connect with timeout to surface failures to the caller
            try:
                await asyncio.wait_for(self._connect(), timeout=timeout if (timeout or 0) > 0 else None)
            except asyncio.TimeoutError as exc:
                raise RealtimeError(f"Realtime connect timed out after {timeout} seconds") from exc
            except Exception as exc:
                raise RealtimeError(f"Realtime connect failed: {exc}") from exc

    def _chunk_audio(self, pcm16: np.ndarray, chunk_ms: float = 50.0) -> Iterable[np.ndarray]:
        chunk_samples = max(1, int(self._output_sample_rate * chunk_ms / 1000.0))
        total = pcm16.size
        for start in range(0, total, chunk_samples):
            yield pcm16[start : start + chunk_samples]

    async def _consume_events(self, audio_consumer: Callable[[np.ndarray], None]) -> RealtimeResponse:
        if self._ws is None:
            raise RealtimeError("Realtime websocket is not connected.")

        text_fragments: List[str] = []
        tool_buffers: Dict[str, Dict[str, Any]] = {}
        tool_results: List[Dict[str, Any]] = []
        response_id: Optional[str] = None

        logger = logging.getLogger(__name__)

        while True:
            try:
                raw = await self._recv_event()
            except websockets.ConnectionClosed as exc:
                raise RealtimeError(f"Realtime connection closed: {exc.code}") from exc

            event = json.loads(raw)
            event_type = event.get("type")
            if event_type is None:
                continue

            if logger.isEnabledFor(logging.DEBUG):
                if event_type in {"response.output_audio.delta", "response.audio.delta"}:
                    audio_field = event.get("audio") or event.get("delta") or event.get("chunk")
                    logger.debug(
                        "Realtime event: %s (audio payload size=%s)",
                        event_type,
                        len(audio_field) if isinstance(audio_field, (bytes, str)) else "complex",
                    )
                else:
                    logger.debug("Realtime event: %s", event_type)

            if "response_id" in event:
                response_id = event.get("response_id") or response_id

            if event_type in {"response.output_text.delta", "response.text.delta"}:
                delta = event.get("delta") or event.get("text")
                text_fragments.extend(self._extract_text(delta))
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug("Realtime text delta: %s", delta)
            elif event_type in {"response.output_text.done", "response.text.done"}:
                delta = event.get("text") or event.get("output")
                text_fragments.extend(self._extract_text(delta))
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug("Realtime text done: %s", delta)
            elif event_type in {"response.output_audio.delta", "response.audio.delta"}:
                chunks = list(self._extract_audio(event))
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "Realtime audio delta: %d chunk(s) (sizes=%s)",
                        len(chunks),
                        [chunk.size for chunk in chunks],
                    )
                for chunk in chunks:
                    audio_consumer(chunk)
            elif event_type == "response.output_item.added":
                item = event.get("item") or {}
                if item.get("type") == "function_call":
                    call_id = item.get("call_id")
                    if call_id:
                        tool_buffers.setdefault(call_id, {"name": item.get("name"), "args": ""})
            elif event_type == "response.function_call_arguments.delta":
                call_id = event.get("tool_call_id") or event.get("call_id")
                if not call_id:
                    continue
                info = tool_buffers.setdefault(call_id, {"name": event.get("name"), "args": ""})
                info["args"] += event.get("arguments", "")
            elif event_type == "response.function_call_arguments.done":
                call_id = event.get("tool_call_id") or event.get("call_id")
                if not call_id:
                    continue
                info = tool_buffers.get(call_id)
                if not info:
                    continue
                tool_name = info.get("name")
                args_text = info.get("args", "")
                try:
                    arguments = json.loads(args_text) if args_text else {}
                except json.JSONDecodeError:
                    arguments = {}
                logging.info("Realtime: invoking tool %s with arguments=%s", tool_name, arguments)
                output = self._tool_registry.dispatch(tool_name, arguments)
                logging.info("Realtime: tool %s returned %s", tool_name, output)
                tool_results.append(
                    {"tool_call_id": call_id, "name": tool_name, "arguments": arguments, "output": output}
                )
                payload = {
                    "type": "response.tool_result",
                    "tool_call_id": call_id,
                    "output": json.dumps(output, ensure_ascii=False),
                }
                if response_id:
                    payload["response_id"] = response_id
                await self._ws.send(json.dumps(payload))
            elif event_type in {"response.completed", "response.done"}:
                break
            elif event_type == "response.error":
                message = event.get("error", {}).get("message", "Unknown realtime error")
                raise RealtimeError(message)
            else:
                logging.debug("Unhandled realtime event: %s", event_type)

        return RealtimeResponse(text="".join(text_fragments).strip(), tool_results=tool_results, response_id=response_id)

    def _extract_text(self, payload: Any) -> List[str]:
        fragments: List[str] = []

        def _visit(value: Any) -> None:
            if isinstance(value, str) and value:
                fragments.append(value)
            elif isinstance(value, dict):
                for key in ("text", "value", "delta", "content"):
                    if key in value:
                        _visit(value[key])
                if "items" in value:
                    _visit(value["items"])
            elif isinstance(value, list):
                for item in value:
                    _visit(item)

        if payload is not None:
            _visit(payload)
        return fragments

    def _extract_audio(self, event: Dict[str, Any]) -> Iterable[np.ndarray]:
        audio_field = event.get("audio") or event.get("delta") or event.get("chunk")
        if audio_field is None:
            return []

        def _flatten(node: Any) -> Iterable[str]:
            if isinstance(node, str):
                yield node
            elif isinstance(node, dict):
                for key in ("audio", "chunk", "value", "delta", "data"):
                    if key in node:
                        yield from _flatten(node[key])
                if "chunks" in node:
                    yield from _flatten(node["chunks"])
            elif isinstance(node, list):
                for item in node:
                    yield from _flatten(item)

        chunks: List[np.ndarray] = []
        for audio_b64 in _flatten(audio_field):
            try:
                raw = base64.b64decode(audio_b64)
                chunk = np.frombuffer(raw, dtype=np.int16)
                if chunk.size:
                    chunks.append(chunk)
            except Exception:
                continue
        return chunks

    async def _recv_event(self) -> str:
        if self._ws is None:
            raise RealtimeError("Realtime websocket is not connected.")
        timeout = getattr(self._config, "session_timeout", 0) or 0
        if timeout <= 0:
            return await self._ws.recv()
        try:
            return await asyncio.wait_for(self._ws.recv(), timeout=timeout)
        except asyncio.TimeoutError as exc:
            raise RealtimeError(f"Realtime response timed out after {timeout} seconds") from exc
