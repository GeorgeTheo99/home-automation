# Voice Gateway Overhaul Plan

This plan describes a clean redesign of the voice gateway to rely exclusively on OpenAI’s GPT‑5 Realtime streaming API for speech-in/speech-out and tool calls, with local wake word (Vosk) and local VAD for end-of-utterance detection. It replaces the current Whisper fallback and legacy pipeline.

Goals
- Use OpenAI Realtime WebSocket API for all ASR + NLU + TTS.
- Use wake word (“Hey Computer” via Vosk) only to start the first turn; after playback, arm a follow-up window driven by local VAD (longer if the assistant asked a question; short otherwise) to allow another turn without the wake word.
- Use local Silero VAD (PyTorch) to detect end-of-utterance and to detect start-of-speech during the follow-up window; then send audio to Realtime.
- Integrate Home Assistant via REST as a single tool; also provide a simple weather tool.
- Keep the runtime simple and robust; no server-side VAD; action-completion chime only (no general turn-end chimes).
- Separate concerns into small modules with minimal state.

Non‑Goals (v1)
- No Whisper fallback STT.
- No client-side TTS.
- No AEC/barge-in.
- No server-side VAD.

State Machine
- COLD_IDLE → WAIT_WAKE → CAPTURE → END_DETECTED → STREAM → PLAYBACK → (FOLLOWUP_ARMED | COLD_IDLE)
  - WAIT_WAKE: Vosk listens for the wake word “Hey Computer” (or configured phrases) using a small grammar.
  - CAPTURE: After wake (or follow-up start), capture mic audio (16 kHz mono PCM16).
  - END_DETECTED: Silero VAD detects end-of-utterance (silence threshold, max length guard).
  - STREAM: Up-sample to 24 kHz and stream to Realtime via input_audio_buffer.append; commit and request response.
  - PLAYBACK: Play audio from Realtime; handle tool calls during response generation.
  - FOLLOWUP_ARMED: After playback, if an action completed and no follow-up question was asked, play a short action chime and return to COLD_IDLE. Otherwise, arm a follow-up window: if the final sentence ends with a question mark, use `FOLLOWUP_WINDOW_SEC` (e.g., 6 s); else use `FOLLOWUP_INFO_WINDOW_SEC` (e.g., 2.5 s). Within this window, Silero VAD can start the next turn without the wake word. If no speech is detected before timeout, return to COLD_IDLE.

Multi‑Turn Follow‑up Policy (Final)
- Policy
  - If an action completes (e.g., Home Assistant service success): play a short action chime and end the turn (no follow-up window).
  - Else if the final sentence ends with “?”: arm a full follow-up window (`FOLLOWUP_WINDOW_SEC`, default 6.0 s).
  - Else: arm a short passive follow-up window (`FOLLOWUP_INFO_WINDOW_SEC`, default 2.5 s) to catch quick add-ons.
  - Use VAD gating with conservative thresholds; if no speech arrives, end quietly.
- Implementation
  - In `main.py` STREAM state, after receiving `RealtimeResponse`:
    - Detect action completion via tool results; if completed and no question, chime → return to WAIT_WAKE.
    - Otherwise set `followup_deadline` based on `asked_question(response.text)` (full vs short window) and transition to FOLLOWUP_ARMED.
  - Keep `FOLLOWUP_GUARD_MS` delay post-playback before listening; in FOLLOWUP_ARMED, capture via VAD; on timeout/no speech → WAIT_WAKE.
  - Add a tiny earcon helper to play the action-completion chime only.
- Heuristics
  - `asked_question(text)`: `text.strip().rstrip(' "'”’)]').endswith('?')`.
- Defaults
  - `FOLLOWUP_WINDOW_SEC` (question): 6.0; `FOLLOWUP_INFO_WINDOW_SEC` (informational): 2.5; `FOLLOWUP_GUARD_MS`: 400.

High-Level Architecture
- main.py: Orchestrator and state machine runner.
- config.py: Centralized env parsing and typed configuration.
- audio/
  - input.py: Mic capture at 16 kHz, mono, PCM16. Provides a blocking reader with gain and device selection.
  - output.py: PCM16 playback from Realtime audio stream (speaker device selection).
- wake/
  - vosk.py: Vosk-based wake word detector using a constrained grammar/keywords.
- vad/
  - silero.py: Silero VAD wrapper + endpointing logic (speech/silence decision, min/max durations, padding).
- gpt/
  - realtime_client.py: Minimal Realtime websocket client: connect/update session with tools, stream audio, commit, create response, receive events, surface tool calls and audio chunks.
- tools/
  - registry.py: Tool JSON schemas for the session and a dispatcher for tool call handling.
  - home_assistant.py: REST client (call_service).
  - weather.py: Lightweight weather lookup (e.g., wttr.in), used by the weather tool.

File Layout
- main.py (orchestrator)
- config.py
- audio/input.py
- audio/output.py
- wake/vosk.py
- vad/silero.py
- gpt/realtime_client.py
- tools/registry.py
- tools/home_assistant.py
- tools/weather.py
- requirements.txt (updated)
- README.md (update later)
- plan.md (this file)

External Dependencies
- vosk (offline ASR; used only for wake word detection)
- numpy, sounddevice
- websockets, requests, python-dotenv
- torch (CPU), silero (model code) – for Silero VAD
- Optional: resampy or simple linear resampler (we will implement a simple linear resample to 24 kHz to avoid extra deps)

Configuration (env)
- OPENAI_API_KEY: required.
- REALTIME_MODEL: default `gpt-realtime-2025-08-28`.
- REALTIME_MODALITIES: default `audio,text`.
- REALTIME_TEMPERATURE: default `0.2`.
- REALTIME_SESSION_TIMEOUT: default `12.0` (seconds).

- MIC_DEVICE_NAME / MIC_DEVICE_INDEX: optional capture device selection.
- SPEAKER_DEVICE_NAME / SPEAKER_DEVICE_INDEX: optional playback device selection.
- INPUT_GAIN: linear gain (float, default `1.0`).
- INPUT_SAMPLE_RATE: default `16000` (fixed for VAD, mono only).

- FOLLOWUP_ENABLED: default `true`.
- FOLLOWUP_ARM_MODE: `question_only` (default) or `always`.
- FOLLOWUP_WINDOW_SEC: default `6.0`.
- FOLLOWUP_INFO_WINDOW_SEC: default `2.5`.
- FOLLOWUP_GUARD_MS: default `400` (ignore mic after playback to avoid leakage).
- FOLLOWUP_MIN_RMS: optional float in [0,1]; if set, require at least this RMS to trigger start-of-speech in follow-up.
- FOLLOWUP_MIN_PEAK: optional int PCM16 peak threshold (e.g., 4500) for start-of-speech gating.

- VOSK_MODEL_PATH: path to the local Vosk model directory (e.g., a subfolder under `models/`). Required.
- WAKEWORD_PHRASES: comma-separated list of phrases; default `hey computer`.
- WAKE_CONFIDENCE_THRESHOLD: default `0.6` (0–1), minimum confidence for detection.
- WAKE_DEBOUNCE_MS: default `1200` (suppress re-triggers within this window).

- VAD_SPEECH_PROB_THRESHOLD: default `0.5` (0–1), speech decision threshold.
- VAD_MIN_SPEECH_MS: default `400` (minimum speech duration to treat as valid).
- VAD_MIN_SILENCE_MS: default `600` (silence to end utterance after speech started).
- VAD_MAX_SEGMENT_MS: default `7000` (max utterance length safety cutoff).
- VAD_WINDOW_MS: default `30` (frame window size given to Silero).
- VAD_PAD_MS: default `120` (trailing padding kept after end-of-utterance).

- HA_URL: e.g., `http://127.0.0.1:8123`.
- HA_TOKEN: long-lived token for REST.

- WEATHER_API_URL: default `https://wttr.in`.
- WEATHER_DEFAULT_UNITS: `metric` or `imperial`.
- WEATHER_LANG: default `en`.

Main Control Flow (detailed)
1) Startup
   - Load env via config.py; validate required values (OPENAI_API_KEY, VOSK_MODEL_PATH).
   - Initialize Vosk recognizer with constrained grammar for `WAKEWORD_PHRASES` at 16 kHz.
   - Initialize Silero VAD (Torch) model; warm up once to avoid first-turn latency.
   - Initialize GPT‑5 Realtime client (connect, update session with tools, set instructions, set voice, modalities, turn_detection: none).
   - Initialize mic and speaker devices.

2) Cold idle and Wake
   - In COLD_IDLE, continuously read microphone frames (16 kHz) and feed to Vosk.
   - On detection event (keyword/phrase match with confidence ≥ threshold), transition to CAPTURE and start a new utterance buffer (clear previous).

3) Capture + VAD (for any turn)
   - Keep appending frames to an in-memory buffer.
   - Run Silero VAD per frame (or small chunk) and track:
     - When first exceeding `VAD_SPEECH_PROB_THRESHOLD`, mark speech_started_ts.
     - Once speech started, if we see >= `VAD_MIN_SILENCE_MS` of consecutive silence, mark end.
     - Enforce `VAD_MIN_SPEECH_MS` to avoid false starts.
     - If we hit `VAD_MAX_SEGMENT_MS`, force end.
   - At end, include `VAD_PAD_MS` of trailing samples (if available), and move to STREAM.

4) Stream to Realtime
   - Up-sample the captured 16 kHz buffer to 24 kHz PCM16.
   - Send base64-encoded chunk(s) via `input_audio_buffer.append` events.
   - Send `input_audio_buffer.commit` then `response.create`.
   - While awaiting response, handle `conversation.updated` deltas; if a function_call appears, route to tools/registry, send `conversation.item.create` with `function_call_output`, then continue.

5) Playback and Follow-up arming
   - As audio deltas arrive (Int16 PCM), write them to output device in near-real time.
   - On response completion, accumulate the final assistant text and inspect tool results.
     - If a Home Assistant action completed and no follow-up question was asked: play a short action chime and return to COLD_IDLE (no follow-up window).
     - Else determine window: if the final sentence ends with a question mark, use `FOLLOWUP_WINDOW_SEC`; otherwise use `FOLLOWUP_INFO_WINDOW_SEC`.
     - Arm FOLLOWUP_ARMED with the chosen deadline.
   - In FOLLOWUP_ARMED: after a short guard (`FOLLOWUP_GUARD_MS`), monitor Silero VAD for start-of-speech within the window; if speech is detected, transition to CAPTURE (without wake word). If timeout or only noise, transition to COLD_IDLE.

Instructions/Persona
- Provide a short instruction block focused on:
  - Being concise 
  - Preferring direct, actionable responses
  - Using available tools for HA control or weather
  - Asking a direct question when more info is needed; the system will allow a brief follow-up window after responses

Modules and Interfaces (prescriptive)

config.py
- `class Config`: structured config with attributes for all env values.
- `def load_config() -> Config`.

audio/input.py
- `class Microphone:`
  - `__init__(sample_rate: int, channels: int, device_name: Optional[str], device_index: Optional[int], gain: float)`
  - `start()` / `stop()`
  - `read_frames(frame_samples: int) -> np.ndarray`  (blocking, returns int16 mono)
- Frame size guidance: 160 samples (10 ms at 16 kHz), 320 (20 ms), or 480 (30 ms). Use 480 by default to match VAD window.

audio/output.py
- `class Speaker:`
  - `__init__(sample_rate: int = 24000, channels: int = 1, device_name: Optional[str], device_index: Optional[int])`
  - `start()` / `stop()`
  - `play_pcm16(pcm: np.ndarray) -> None` (non-blocking enqueue)

wake/vosk.py
- `class VoskWake:`
  - `__init__(model_path: str, phrases: List[str], confidence_threshold: float = 0.6, debounce_ms: int = 1200)`
  - `process(frame: np.ndarray) -> bool` (feed PCM16 at 16 kHz; returns True on wake phrase match with confidence ≥ threshold, honoring debounce)
  - `close()`
- Notes:
  - Use a small grammar constrained to `phrases` for fast keyword spotting.
  - Ensure input is mono, PCM16, 16 kHz; 10–30 ms frames are acceptable (e.g., 480 samples per frame).

vad/silero.py
- `class SileroVAD:`
  - `__init__(sample_rate: int = 16000, speech_prob_threshold: float, min_speech_ms: int, min_silence_ms: int, max_segment_ms: int, pad_ms: int, window_ms: int = 30)`
  - `reset_turn()`
  - `process(frame: np.ndarray) -> Tuple[bool, bool]`  -> (speech_detected, end_of_utterance)
  - `get_buffered_audio() -> np.ndarray` (returns PCM16 for the current turn including padding)
- Implementation details:
  - Load Silero with Torch once: `torch.hub.load('snakers4/silero-vad', 'silero_vad')` (or from the `silero` package if available).
  - Keep a ring buffer of frames with timestamps to compute windows.
  - Convert PCM16 -> float32 in [-1, 1] for model input; compute prob per window.

gpt/realtime_client.py
- `class RealtimeClient:`
  - `__init__(api_key: str, model: str, modalities: List[str], temperature: float, session_timeout: float)`
  - `async connect()` -> establish ws to `wss://api.openai.com/v1/realtime?model=...` with headers.
  - `async update_session(instructions: str, tools: List[dict], voice: Optional[str] = None)` -> send `session.update`.
  - `async send_turn(pcm16_24k: np.ndarray, on_tool_call: Callable[[str, dict], Awaitable[dict]], on_audio_chunk: Callable[[np.ndarray], None]) -> dict`:
    - Append chunks via `input_audio_buffer.append` (base64 of PCM16), then `input_audio_buffer.commit`, then `response.create`.
    - Handle events until response completion. On tool call, await `on_tool_call` and send `conversation.item.create` with `function_call_output`.
    - Return a summary dict (transcript text, tool results, status).
  - `async close()`.
- Audio format: PCM16 mono at 24 kHz.

tools/registry.py
- `def get_session_tools() -> List[dict]` (JSON schemas for tools).
- `async def handle_tool_call(name: str, args: dict) -> dict` (dispatch to module handlers and return JSON-serializable result).

tools/home_assistant.py
- `class HomeAssistant:`
  - `__init__(base_url: str, token: str)`
  - `def call_service(self, domain: str, service: str, entity_id: Union[str, List[str]], data: Optional[dict] = None) -> dict`
  - Implementation: POST `/api/services/{domain}/{service}` with JSON body; returns JSON response or raises with HTTP details.

tools/weather.py
- `def get_weather(location: Optional[str], units: Optional[str]) -> dict` (use existing wttr.in integration style; small timeout; friendly error messages).

Session Tools (schemas)
- call_home_assistant
  - name: `call_home_assistant`
  - description: Call a Home Assistant service using domain/service/entity_id/data.
  - parameters:
    - type: object
    - properties: domain (string), service (string), entity_id (string or array of string), data (object)
    - required: [domain, service, entity_id]

- get_weather
  - name: `get_weather`
  - description: Retrieve current weather and short forecast for a location.
  - parameters:
    - type: object
    - properties: location (string, default from config), units (enum: metric|imperial)

Main Orchestrator (main.py) – prescriptive loop
- Synchronous high-level loop using one background task for the websocket client:
  1) Load config; init modules (VoskWake, SileroVAD, Microphone, Speaker, RealtimeClient, HA client; register tools).
  2) Update Realtime session with instructions + tools (tool_choice: auto; modalities: audio,text; temperature as configured).
  3) IDLE/WAIT_WAKE: read frames (e.g., 480 samples at 16 kHz) and call `VoskWake.process(frame)`; if detected, `SileroVAD.reset_turn()` and start CAPTURE.
  4) CAPTURE: continue reading frames (e.g., 480-sample 30 ms windows) -> push to `SileroVAD.process(frame)` and also append to an utterance buffer.
  5) If `end_of_utterance` -> copy buffered audio (including pad), up-sample to 24 kHz, and `await realtime.send_turn(...)` to get response while streaming playback via `on_audio_chunk` -> `Speaker.play_pcm16`.
  6) After response completes, return to IDLE (reset internal buffers/states), repeat.

Resampling
- Implement a simple linear resampler from 16 kHz int16 to 24 kHz int16 (float intermediate + np.interp), identical to current approach.

Instructions Prompt (short)
- “You are a helpful, concise home assistant. Speak succinctly. Use tools when appropriate to control the home or fetch weather. If you need more information, ask a direct question. The system will allow a brief follow-up window after your response; otherwise the user must say the wake word again.”

Deletions / Removals (explicit)
- Remove all Whisper/legacy STT code-paths and related env vars.
- Remove OpenWakeWord and all references.
- Remove Porcupine and all references; disable it entirely.
- Remove server VAD code paths and env flags.
- Remove barge-in logic.
- Trim main.py to delegate to new modules; remove echo canceller and complex channel selection heuristics for now (single-channel mono pipeline).

 Acceptance Criteria
- Startup succeeds with valid env; logs confirm Vosk and Silero initialized; Realtime connected.
- Saying “Hey Computer …” triggers capture; speaking for > `VAD_MIN_SPEECH_MS` then pausing for ≥ `VAD_MIN_SILENCE_MS` ends the turn.
- The utterance is streamed to Realtime and a spoken response plays back without noticeable gaps.
- “Hey Computer, turn on the kitchen lights” calls HA service (domain: light, service: turn_on, entity_id: light.kitchen) and the response confirms.
- “Hey Computer, what’s the weather” returns a concise weather summary.
- If a Home Assistant action completes successfully, a short action chime plays and the turn ends (no follow-up window).
- If the assistant’s final sentence ends with a question mark, a follow-up window of `FOLLOWUP_WINDOW_SEC` is armed after playback; speaking within that window triggers another turn without the wake word.
- If the assistant provides an informational answer (no trailing question), a short follow-up window of `FOLLOWUP_INFO_WINDOW_SEC` is armed; quick add-ons within this window trigger another turn; otherwise the system returns to COLD_IDLE and requires the wake word.

Manual Test Plan
1) Environment
   - Set `OPENAI_API_KEY`, `HA_URL`, `HA_TOKEN`.
   - Ensure a Vosk model is present under `models/` and set `VOSK_MODEL_PATH` to that directory.
   - Optionally set `WAKEWORD_PHRASES` (default `hey computer`).
2) Launch
   - `python main.py`.
3) Wake and command
   - Speak: “Hey Computer, turn on the living room lights”. Observe HA call and voice confirmation.
4) Weather
   - Speak: “Hey Computer, what’s the weather today?” Listen for concise spoken summary.
5) Follow-up window
   - Ask a question that elicits a question in response, e.g., “Hey Computer, set a reminder”. The assistant should ask “for when?” or similar. After playback ends, wait ~500 ms and answer “tomorrow at 9am” without saying the wake word. The follow-up should trigger immediately via VAD and proceed as a new turn. If you remain silent for longer than `FOLLOWUP_WINDOW_SEC`, the system should require the wake word again.
6) Robustness
   - Speak a very short utterance; VAD should ignore if shorter than `VAD_MIN_SPEECH_MS`.
   - Speak in noise; ensure wake required each time; minimal false starts.

Performance/Latency Targets
- Wake-to-capture start: < 50 ms.
- End-of-utterance to first audio: ~300–800 ms depending on network.
- End-of-utterance cutoff: controlled by `VAD_MIN_SILENCE_MS` (~600 ms default) + `VAD_PAD_MS`.

Logging & Observability
- INFO-level: state transitions, tool calls, HA responses (status code only), response summary text length.
- DEBUG-level: VAD probabilities (min/avg/max per segment), audio chunk counts, websocket event names.
- Optional: rotate text logs in `logs/conversations/` with timestamped files.

Security
- Do not log raw tokens or secrets.
- Redact HA token; optionally redact entity_id if privacy is a concern.

Implementation Steps (for the next agent)
1) Dependencies
   - Update `requirements.txt`: add `vosk`, `torch`, `numpy`, `websockets`, `requests`, `sounddevice`, `python-dotenv` (and remove `openwakeword` and `pvporcupine`).
2) Create modules per file layout with minimal code skeletons and docstrings.
3) Implement `config.py` parsing with validation for required values.
4) Implement `wake/vosk.py` using `vosk` with a constrained grammar for `WAKEWORD_PHRASES` and confidence threshold/debounce.
5) Implement `vad/silero.py` (model load, per-frame prob, endpointing).
6) Implement `audio/input.py` and `audio/output.py` with sounddevice.
7) Implement `tools/home_assistant.py` and `tools/weather.py`.
8) Implement `tools/registry.py` (schemas + dispatcher).
9) Implement `gpt/realtime_client.py` (WS connect, session.update, append/commit/create, event handling, tool call plumbing).
10) Implement `main.py` orchestrator with the state machine and glue between modules, including FOLLOWUP_ARMED handling, question-heuristic arming, post-playback guard, and follow-up VAD start-of-speech gating.
11) Remove legacy code from `main.py` and `realtime.py`; delete unused env var references; update README.
12) Manual test pass; iterate VAD thresholds and mic gain as needed.

Open Questions
- Voice selection: prefer a specific Realtime voice (e.g., `verse`)?
- Should we save short text conversation transcripts to `logs/conversations/`?

Notes
- Vosk runs offline and requires a local model directory. Place the model under `models/` and point `VOSK_MODEL_PATH` to it.
