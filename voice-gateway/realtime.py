from __future__ import annotations

import asyncio
import base64
import json
import time
from dataclasses import dataclass
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


class RealtimeError(Exception):
    pass


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


async def run_realtime_session(
    pcm16: np.ndarray,
    config: RealtimeConfig,
    entity_context: List[JsonDict],
    instructions: str,
    ha_executor: Callable[[str, str, Any, Optional[Dict[str, Any]]], Any],
    confirm_message: Callable[[Dict[str, Any]], str],
    speaker: Callable[[str], None],
    logger: Callable[[str, str], None],
) -> bool:
    """Send the captured audio to OpenAI realtime API and handle tool calls."""
    if not config.api_key:
        logger("ERROR", "Realtime enabled but OPENAI_API_KEY missing.")
        return False

    url = f"wss://api.openai.com/v1/realtime?model={config.model}"
    headers = [
        ("Authorization", f"Bearer {config.api_key}"),
        ("OpenAI-Beta", "realtime=v1"),
    ]

    loop = asyncio.get_event_loop()
    target_rate = config.stream_sample_rate or config.input_sample_rate

    def prepare_audio():
        audio = pcm16
        if config.send_audio and target_rate != config.input_sample_rate:
            audio = _resample_pcm16(audio, config.input_sample_rate, target_rate)
        chunks = _read_frames(
            audio,
            sample_rate=target_rate,
            chunk_ms=100.0,
            max_audio_secs=config.max_audio_secs,
        )
        return audio, chunks

    resampled_audio, audio_chunks = await loop.run_in_executor(
        None,
        prepare_audio,
    )
    if not audio_chunks:
        logger("WARN", "Realtime session: no audio chunks produced")
        return False
    total_ms = (len(resampled_audio) / float(target_rate)) * 1000.0 if target_rate else 0.0
    total_bytes = sum(len(chunk) for chunk in audio_chunks)
    peak = int(np.abs(resampled_audio).max()) if resampled_audio.size else 0
    logger(
        "INFO",
        (
            f"Realtime capture: {total_ms:.0f} ms audio "
            f"({len(audio_chunks)} chunks, {total_bytes // 1024} KiB, peak={peak})"
        ),
    )

    tools = [
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
                            {
                                "type": "array",
                                "items": {"type": "string"},
                                "minItems": 1,
                            },
                        ]
                    },
                    "data": {"type": "object"},
                },
                "required": ["domain", "service", "entity_id"],
            },
        }
    ]

    entity_json = json.dumps(entity_context, ensure_ascii=False)
    full_instructions = instructions + "\n\nValid entities JSON:" + entity_json

    response_id: Optional[str] = None
    pending_tool_args: Dict[str, Dict[str, Any]] = {}
    collected_text: List[str] = []
    collected_audio: bytearray = bytearray()

    async def send_tool_result(ws, tool_call_id: str, output: JsonDict) -> None:
        nonlocal response_id
        payload = {
            "type": "response.tool_result",
            "tool_call_id": tool_call_id,
            "output": json.dumps(output, ensure_ascii=False),
        }
        if response_id:
            payload["response_id"] = response_id
        await ws.send(json.dumps(payload))

    try:
        async with websockets.connect(url, additional_headers=headers, max_size=20_000_000) as ws:
            session_body = {
                "modalities": config.modalities,
                "temperature": max(0.6, config.temperature),
            }
            if config.send_audio:
                session_body["input_audio_format"] = "pcm16"
                if target_rate:
                    session_body["input_audio_transcription"] = {"model": "gpt-4o-mini-transcribe"}
            session_body["turn_detection"] = None
            if config.voice and "audio" in config.modalities:
                session_body["voice"] = config.voice
            await ws.send(json.dumps({
                "type": "session.update",
                "session": session_body,
            }))

            for chunk in audio_chunks:
                if not config.send_audio:
                    break
                b64 = base64.b64encode(chunk).decode("ascii")
                await ws.send(json.dumps({
                    "type": "input_audio_buffer.append",
                    "audio": b64,
                }))
            if config.send_audio:
                await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
                logger("INFO", "Realtime input buffer committed")

            await ws.send(json.dumps({
                "type": "response.create",
                "response": {
                    "modalities": config.modalities,
                    "temperature": max(0.6, config.temperature),
                    "instructions": full_instructions,
                    "tools": tools,
                },
            }))

            start_time = time.time()

            while True:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=config.session_timeout)
                except asyncio.TimeoutError:
                    raise RealtimeError("Realtime session timed out")
                data = json.loads(msg)
                event_type = data.get("type")

                if event_type == "response.created":
                    response_id = data.get("response", {}).get("id", response_id)
                elif event_type == "response.output_text.delta":
                    delta = data.get("delta") or data.get("text")
                    if isinstance(delta, str):
                        collected_text.append(delta)
                elif event_type == "response.output_audio.delta":
                    audio_b64 = data.get("delta") or data.get("audio")
                    if isinstance(audio_b64, str):
                        collected_audio.extend(base64.b64decode(audio_b64))
                elif event_type == "response.output_item.added":
                    item = data.get("item") or {}
                    if item.get("type") == "function_call":
                        call_id = item.get("call_id")
                        if call_id:
                            info = pending_tool_args.setdefault(call_id, {"name": item.get("name"), "args": ""})
                            if item.get("arguments"):
                                info["args"] += item["arguments"]
                elif event_type == "conversation.item.input_audio_transcription.completed":
                    transcript = data.get("transcript")
                    if transcript:
                        logger("INFO", f"Realtime heard: {transcript}")
                elif event_type == "response.function_call_arguments.delta":
                    call_id = data.get("call_id")
                    if call_id:
                        info = pending_tool_args.setdefault(call_id, {"name": None, "args": ""})
                        if data.get("delta"):
                            info["args"] += data["delta"]
                elif event_type == "response.function_call_arguments.done":
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
                                await send_tool_result(ws, call_id, {"error": f"Invalid JSON args: {exc}"})
                                pending_tool_args.pop(call_id, None)
                                continue
                        else:
                            args_json = {}
                        if name == "call_home_assistant":
                            try:
                                result = ha_executor(
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
                                await send_tool_result(ws, call_id, {
                                    "ok": True,
                                    "result": result,
                                    "summary": summary,
                                })
                                if summary:
                                    logger("INFO", f"Realtime action: {summary}")
                                if summary and not config.expect_audio_output:
                                    speaker(summary)
                            except Exception as exc:
                                await send_tool_result(ws, call_id, {"ok": False, "error": str(exc)})
                        else:
                            await send_tool_result(ws, call_id, {"error": f"Unknown tool {name}"})
                        pending_tool_args.pop(call_id, None)
                elif event_type in {"response.completed", "response.done"}:
                    break
                elif event_type == "error":
                    raise RealtimeError(str(data.get("error")))

                if time.time() - start_time > config.session_timeout:
                    raise RealtimeError("Realtime session exceeded timeout")

            if collected_audio and config.expect_audio_output:
                try:
                    import sounddevice as sd

                    samples = np.frombuffer(bytes(collected_audio), dtype=np.int16)
                    if samples.size:
                        sd.play(samples / 32768.0, samplerate=16000)
                        sd.wait()
                except Exception as exc:
                    logger("WARN", f"Realtime audio playback failed: {exc}")

            if collected_text:
                logger("INFO", f"Realtime response text: {''.join(collected_text)}")
            return True
    except Exception as exc:
        logger("ERROR", f"Realtime session error: {exc}")
        return False
