# Voice Gateway (Path A)

Wake word (local) → STT (OpenAI Whisper API) → NLU (GPT‑5 JSON) → Home Assistant REST.

## Prerequisites
- Ubuntu/Linux host with microphone access
- Home Assistant running (Container is fine) and a Long‑Lived Access Token
- Python 3.10+
- System packages
  - `sudo apt update && sudo apt install -y python3-venv python3-pip portaudio19-dev ffmpeg espeak-ng alsa-utils`

## Setup (uv recommended)
Install uv (once):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# then restart your shell or: source ~/.local/bin/env
```

Install a Python toolchain that works with OpenWakeWord (Python 3.11 recommended):
```bash
uv python install 3.11
```

Create environment and install deps:
```bash
cd /home/georgetheo/home_automation/voice-gateway
uv venv -p 3.11  # creates .venv with Python 3.11
uv sync  # installs dependencies from pyproject.toml

cp .env.example .env
# Edit .env: set HA_URL, HA_TOKEN, OPENAI_API_KEY
```

Run (no activation needed):
```bash
uv run python main.py
```

## Audio Setup (PipeWire recommended)
We configured PipeWire (user session) and set Built‑in Analog Stereo as the default sink/source.

1) Install/enable PipeWire + WirePlumber
```bash
sudo apt install -y pipewire pipewire-audio-client-libraries wireplumber
systemctl --user enable --now pipewire pipewire-pulse wireplumber
```

2) Set defaults and unmute via `wpctl`
```bash
wpctl status
# Identify your Built-in Analog Stereo sink and source IDs (e.g., sink=48, source=49)
wpctl set-default 48
wpctl set-default 49
wpctl set-mute 49 0
wpctl set-volume 49 0.85
```

3) Verify mic levels (RMS should rise when you speak)
```bash
uv run python - <<'PY'
import numpy as np, sounddevice as sd
def rms(idx):
    a = sd.rec(int(0.5*16000), samplerate=16000, channels=1, dtype='int16', device=idx); sd.wait()
    x = a.flatten().astype('float32')/32768.0
    return float(np.sqrt((x*x).mean()))
print('Default devices (in,out):', sd.default.device)
for i in [13,5]:
    try: print(i, 'RMS=', rms(i))
    except Exception as e: print(i, 'error:', e)
PY
```

If you prefer ALSA‑only (no PipeWire), use `alsamixer -c 0` to enable/unmute Capture and `arecord -D default` to test.

### Silent Input Troubleshooting
- If logs show `Mic peak=0 rms=0.0000` constantly, your default source may not be set.
  - Run `wpctl status`, note the Sources section, then set a default: `wpctl set-default <source-id>`.
  - If using AUX/line-in, ensure the input port is selected: `wpctl set-port <source-id> analog-input-linein` (or `analog-input-internal-mic`).
- You can let the gateway perform these at startup by setting envs in `voice-gateway/.env`:
  - `WPCTL_SOURCE_ID=47` and `WPCTL_SET_DEFAULT=true`
  - `WPCTL_SET_PORT=analog-input-linein`
- As an alternative, explicitly pin a device in `.env` without changing system defaults:
  - `MIC_DEVICE_INDEX=<index>` or `MIC_DEVICE_NAME=<substring>`
  - With `LOG_LEVEL=DEBUG` the gateway lists input-capable devices with indices at startup.

## Alternative (pip/venv)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Optional: list microphones with indices and defaults
```bash
uv run python - <<'PY'
import sounddevice as sd
for i, d in enumerate(sd.query_devices()):
    print(i, d['name'], 'in_channels=', d.get('max_input_channels', 0))
print('Default devices (in,out):', sd.default.device)
PY
# Then set MIC_DEVICE_INDEX or MIC_DEVICE_NAME in .env
```

## Run
Using uv:
```bash
uv run python main.py
```
or with an activated venv:
```bash
source .venv/bin/activate
python main.py
```
Speak the wake word (e.g., "hey jarvis"), then try:
- "kitchen lights on"
- "kitchen lights 50 percent"
- "living room lights off"

If your entity IDs differ, the gateway builds aliases from HA friendly names automatically at startup. You can also add/override regexes in `ALIASES_STATIC` inside `main.py`.

## Systemd (optional)
Create a user service:
```bash
mkdir -p ~/.config/systemd/user
cat > ~/.config/systemd/user/ha-voice-gateway.service <<'UNIT'
[Unit]
Description=HA Voice Gateway (wake word -> STT -> GPT-5 -> HA)
After=network-online.target sound.target

[Service]
Type=simple
WorkingDirectory=%h/home_automation/voice-gateway
Environment=PYTHONUNBUFFERED=1
ExecStart=%h/home_automation/voice-gateway/.venv/bin/python %h/home_automation/voice-gateway/main.py
Restart=on-failure

[Install]
WantedBy=default.target
UNIT

systemctl --user daemon-reload
systemctl --user enable ha-voice-gateway
systemctl --user start ha-voice-gateway
journalctl --user -u ha-voice-gateway -f
```

## Notes
- This gateway avoids HA’s Assist/Wyoming UI entirely.
- STT uses OpenAI's `gpt-4o-mini-transcribe` API (successor to Whisper); no local model cache required.
- NLU uses `MODEL_NAME` (default: `gpt-5`). If unavailable, set to `gpt-4o-mini` or `gpt-4.1`.
  - Some models enforce default temperature; set `NLU_TEMPERATURE=1.0` (default) or let the app auto-retry without temperature.
- The realtime path falls back to the classic Whisper+GPT pipeline only when you explicitly set `LEGACY_PIPELINE_ENABLED=true`.
- The gateway allow‑lists HA services for safety (lights/switches by default).
- Local speak-back uses `espeak-ng` by default. You can adjust voice/rate via `.env`.
- GPT NLU gets a JSON array of valid entities (friendly names, room/area, aliases) extracted from HA at startup. Keep entity names in HA up to date for best results.

## Realtime Mode (Experimental)
- Set `REALTIME_ENABLED=true` in `.env` to use OpenAI’s realtime WebSocket API instead of Whisper+chat.
- Requires access to a realtime-capable model (e.g., `gpt-realtime-preview`). Configure:
  - Existing configs that still list the deprecated `gpt-voice-40` will be remapped automatically to `gpt-realtime-preview`.
  - `REALTIME_MODEL`, `REALTIME_MODALITIES` (`audio,text` for voiced replies), `REALTIME_VOICE`, `REALTIME_TEMPERATURE`.
  - `REALTIME_MAX_AUDIO_SECS` limits how much post-wake audio is sent.
  - `REALTIME_SESSION_TIMEOUT` aborts slow sessions; the pipeline falls back to Whisper + GPT on failure.
  - `REALTIME_PLAYBACK_GUARD_SEC` adds padding to the mic mute window while GPT audio is playing.
  - When server VAD is enabled we now allow short (~600 ms) follow-ups; raise or lower `REALTIME_MIN_INPUT_AUDIO_MS` (default 900 ms) to require more audio before committing.
- Server-side VAD is enabled by default (`REALTIME_SERVER_VAD=true`). Tune detection via `REALTIME_VAD_THRESHOLD`, `REALTIME_VAD_SILENCE_MS` (default 350 ms), `REALTIME_VAD_PREFIX_MS`, and optional `REALTIME_VAD_IDLE_TIMEOUT_MS`.
- `REALTIME_PREFETCH_SEC` (default 0.35 s) captures a short lead-in while the realtime session negotiates. `REALTIME_MIN_INPUT_AUDIO_MS` gates the minimum capture length (default 900 ms with server VAD, otherwise at least `POST_WAKE_RECORD_SECS`).
- Current OpenAI policy requires `temperature >= 0.6`; the code clamps lower values automatically, but set it to ~0.7 for best results.
- `REALTIME_FORCE_CREATE_RESPONSE` (default `true`) nudges the model with a manual `response.create` if it stays silent; tune `REALTIME_FORCE_RESPONSE_DELAY_MS` (ms) to wait longer before issuing the nudge.
- Local fallbacks stay silent unless you opt in. Set `REALTIME_FALLBACK_NO_SPEECH` or `REALTIME_FALLBACK_NO_RESPONSE` if you want the gateway to apologize when realtime can’t understand or respond.
- `FOLLOWUP_ENABLED` (default `true`) opens a follow-up window after each assistant reply so you can keep talking without repeating the wake word. Adjust `FOLLOWUP_WINDOW_SEC` (default 5 s), `FOLLOWUP_MIN_RMS`, `FOLLOWUP_MIN_PEAK`, `FOLLOWUP_TRIGGER_BLOCKS`, `FOLLOWUP_MIN_ACTIVE_MS`, `FOLLOWUP_SILENCE_SEC`, and `FOLLOWUP_MAX_CAPTURE_SEC` if you need a longer (or stricter) conversational tail.
- The gateway defines a `call_home_assistant` tool. GPT streams a tool call, we execute it via HA REST, and send a tool result back so the model can confirm.
- Actions detected via the tool call trigger a local chime instead of GPT speech. Configure the file/tone via `ACTION_CHIME_*`, or set `ACTION_CHIME_ENABLED=false` to fall back to TTS confirmations.
- Informational answers stream GPT audio (`REALTIME_MODALITIES` includes `audio`). If no audio arrives, the gateway falls back to local `speak()`.
- Troubleshooting:
  - Check logs for `Realtime session error` messages.
  - Some events/field names may evolve; adjust in `realtime.py` if OpenAI changes their schema.
- Tweak `REALTIME_NOISE_PEAK` if quiet speech is being discarded or background hum still produces prompts.
- The fallback path (Whisper + GPT) runs only if `LEGACY_PIPELINE_ENABLED=true`. If you want that behavior, opt in explicitly.
- For faster wake-word rearm in realtime mode, lower `WAKE_COOLDOWN_SEC` (e.g. `0.4–0.6`) and adjust `WAKE_RETRIGGER_SUPPRESS_SEC` (default `0.6`) to balance latency vs. false re-triggers.
- Realtime responses are forced to English; when only noise is captured (peak below `REALTIME_NOISE_PEAK` with no transcript), the session is dropped politely.

### Latency Tips
- Shorten `POST_WAKE_RECORD_SECS` and `REALTIME_MAX_AUDIO_SECS` (e.g., 1.5–2.0 seconds) to ship less audio.
- For realtime sessions, keep `REALTIME_MODALITIES` to `text` if you don’t need voice replies.
- If you stick with the classic pipeline, switch `MODEL_NAME` to `gpt-4o-mini` or enable fast-paths (`NLU_MODE=fast_first`) so simple light commands avoid GPT entirely.
- Local STT (`faster-whisper`) can still be enabled later if you need a fully offline path.

### NLU Modes
- `fast_only`: Only use local regex fast-paths (on/off, brightness %, simple colors) with HA entity aliasing from friendly names.
- `fast_first`: Try fast-paths, fall back to GPT JSON if no match. (default)
- `gpt_only`: Always call GPT for NLU; disables fast-paths.
  - Enable logs with `LOG_NLU=true` to see which path was used and the final JSON.

## Wake Word Troubleshooting
- Default wakeword: `hey_jarvis`. Saying “weather” will not trigger; only the configured wakeword (or any `hey_*` model in fallback) does.
- To narrow detection, set `WAKEWORD_MODEL` to the bundled model:
  - `voice-gateway/.venv/lib/python3.11/site-packages/openwakeword/resources/models/hey_jarvis_v0.1.tflite`
- Tune sensitivity: try `WAKE_THRESHOLD=0.10..0.55`, `WAKE_BLOCK_SEC=1.0`, `STREAM_BLOCK_SEC=0.2`.
- With `LOG_LEVEL=DEBUG`, you’ll see `RMS` and top scores. RMS should rise when you speak.
- If RMS is low and scores stay tiny, increase `INPUT_GAIN` (e.g., 2.0–6.0). Keep RMS generally below ~0.3 to avoid heavy clipping.
- Avoid double triggers: increase `WAKE_COOLDOWN_SEC` (e.g., >= `POST_WAKE_RECORD_SECS + 0.5`) and/or require a short streak with `WAKE_STREAK=2`.
- Add hysteresis: `WAKE_HYSTERESIS_SEC` adds extra guard time after cooldown to ignore trailing echoes/background voice.
- Fine-tune streak logic: `WAKE_STREAK_WINDOW_SEC` keeps the detection burst tight while `WAKE_FINAL_THRESHOLD` asks for a stronger final score.
- Post-command mute: bump `POST_COMMAND_MUTE_SEC` to keep the mic ignored briefly after HA/GPT responses.
- Clamp noisy playback: `WAKE_RMS_DECAY`, `WAKE_RMS_CLAMP_RATIO`, `WAKE_RMS_CLAMP_ABS`, and `WAKE_NOISE_HOLD_SEC` hold the detector while room audio is loud.
- Capture the very start of speech: `PRE_WAKE_BUFFER_SEC` keeps ~0.5–0.8 s of audio before the wake hit so the first words (“turn on…”) aren’t clipped.

## Remove old Whisper cache (manual)
If present, remove the legacy cache directory used for offline STT:
```bash
rm -rf /home/georgetheo/home_automation/whisper-cache
```
