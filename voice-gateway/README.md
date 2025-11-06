# Voice Gateway (Realtime)

Local wake word + local VAD + OpenAI Realtime (gpt-streaming) for ASR, NLU, TTS, and tool calls. The pipeline:

```
Vosk wake word → Silero VAD endpointing → OpenAI Realtime WebSocket → Home Assistant / Weather tools → PCM audio playback
```

## Requirements

- Linux host with microphone and speaker access.
- Python 3.10+ with `pip` (or [`uv`](https://docs.astral.sh/uv/latest/)).
- Local Vosk model directory (e.g. `vosk-model-small-en-us-0.15`).
- OpenAI API key with access to the Realtime API.
- Optional: Home Assistant instance and long-lived access token if you want voice control.

## Setup

1. **Create a Python environment and install dependencies**

   ```bash
   cd /home/georgetheo/home_automation/voice-gateway
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

   (If you prefer `uv`, run `uv venv` and `uv sync` instead.)

2. **Provision Vosk**

   - Download a lightweight Vosk English model (for example, [`vosk-model-small-en-us-0.15`](https://alphacephei.com/vosk/models)).
   - Extract it somewhere accessible, such as `~/models/vosk-model-small-en-us-0.15`.
   - Point `VOSK_MODEL_PATH` at the extracted directory.

3. **Create `.env` (or export variables)**

   ```bash
   cp .env.example .env  # if you keep one checked in
   ```

   Required environment variables:

   - `OPENAI_API_KEY` – Realtime WebSocket API key.
   - `VOSK_MODEL_PATH` – directory containing the Vosk acoustic model (defaults to `models/vosk-model-small-en-us-0.15`).
   - `HA_URL` / `HA_TOKEN` – Home Assistant base URL and long-lived token (optional but required for home control).

   Useful optional variables:

   | Variable | Default | Description |
   | --- | --- | --- |
   | `REALTIME_MODEL` | `gpt-realtime-2025-08-28` | Realtime model ID. |
   | `REALTIME_MODALITIES` | `audio,text` | Comma separated modalities for responses. |
   | `REALTIME_TEMPERATURE` | `0.2` | Sampling temperature. |
   | `MIC_DEVICE_NAME` / `MIC_DEVICE_INDEX` | | Pin a capture device for sounddevice. |
   | `SPEAKER_DEVICE_NAME` / `SPEAKER_DEVICE_INDEX` | | Pin a playback device. |
   | `INPUT_GAIN` | `1.0` | Linear gain applied to mic PCM before VAD/wake. |
   | `FOLLOWUP_ENABLED` | `true` | Enable follow-up window after responses. |
   | `FOLLOWUP_ARM_MODE` | `question_only` | `question_only` or `always` to arm follow-ups. |
   | `FOLLOWUP_WINDOW_SEC` | `6.0` | Length of the follow-up window. |
   | `FOLLOWUP_GUARD_MS` | `400` | Guard period after playback before re-arming (ms). |
   | `FOLLOWUP_MIN_RMS` / `FOLLOWUP_MIN_PEAK` | | Optional gating thresholds for follow-up speech. |
   | `WAKEWORD_PHRASES` | `hey computer` | Comma-separated wake word phrases (e.g., `hey computer,ok computer`). |
   | `WAKE_CONFIDENCE_THRESHOLD` | `0.6` | Minimum Vosk confidence to accept a wake detection (recommend 0.5–0.6 with a constrained grammar). |
   | `WAKE_DEBOUNCE_MS` | `1200` | Debounce period after a trigger (milliseconds). |
   | `WEATHER_API_URL` | `https://wttr.in` | Weather lookup base URL. |
   | `WEATHER_DEFAULT_LOCATION` | | Default location for weather tool. |

## Running

Activate your environment (if required) and launch the gateway:

```bash
source .venv/bin/activate
python main.py
```

On startup the logs will confirm Vosk, Silero VAD, and the Realtime websocket session have initialised. Say “hey computer …” or “ok computer …” (or your configured wake phrase) to begin a turn. After each spoken answer, the gateway arms a six second follow-up window (when enabled) so you can continue speaking without the wake word if the assistant asked a question.

## Tools

Two tools are registered with the Realtime session:

- `home_assistant_call_service` – calls Home Assistant REST services (`domain`, `service`, optional `entity_id`, `data`).
- `get_weather_summary` – fetches a concise weather summary using `wttr.in` (honours `WEATHER_DEFAULT_LOCATION`).

The tool registry validates arguments and raises descriptive errors if Home Assistant isn’t configured.

## Manual Test Plan

1. Set `OPENAI_API_KEY`, optionally override `VOSK_MODEL_PATH` (defaults to `models/vosk-model-small-en-us-0.15`), configure `WAKEWORD_PHRASES` if desired, and (optionally) `HA_URL`/`HA_TOKEN`.
2. `python main.py`
3. Say “computer, turn on the kitchen lights” – confirm a Home Assistant service call and spoken confirmation.
4. Say “computer, what’s the weather?” – confirm a concise spoken summary.
5. Trigger a follow-up by asking a question that elicits one (“computer, set a reminder”). After the assistant replies, answer the follow-up question within the six second window without using the wake word.

## Troubleshooting

- **No wake detection** – ensure `VOSK_MODEL_PATH` points to a valid Vosk model directory and the wake phrases are audible. Adjust `WAKE_CONFIDENCE_THRESHOLD` (lower it slightly) or `INPUT_GAIN` if detections are missed.
- **Realtime errors** – check the log for `Realtime processing failed` messages. Network connectivity is required while streaming.
- **Follow-up false triggers** – increase `FOLLOWUP_MIN_RMS` (e.g. `0.02`) or `FOLLOWUP_MIN_PEAK` (e.g. `4500`) to demand stronger speech energy.
- **Device selection** – run `python -c "import sounddevice as sd; print(sd.query_devices())"` to inspect indices, then set the `*_DEVICE_*` env vars.

## Project Structure

```
config.py                   # environment parsing
main.py                     # thin entrypoint: config + wiring + signals
pipeline/controller.py      # pipeline finite-state machine (wake → capture → stream → follow-up)
pipeline/capture.py         # VAD-driven utterance capture helper
audio/                      # microphone and speaker abstractions
wake/vosk.py                # wake word detector
vad/silero.py               # streaming VAD + endpointing
gpt/realtime_client.py      # OpenAI Realtime websocket client
tools/                      # Home Assistant + weather tools
```

Logs (INFO level) show state transitions, tool results, and assistant summaries. For more detailed diagnostics, adjust logging in `main.py` (or within `pipeline/controller.py`).
